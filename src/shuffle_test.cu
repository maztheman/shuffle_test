#include "shuffle_test.h"
#include "crypto/blake.hpp"
#include "stratum/primitives/block.hpp"
#include "stratum/streams.hpp"
#include "stratum/arith/uint256.hpp"

#include <cstdint>
#include <ctime>
#include <execution>

using namespace crypto;

#include "sols.h"

#include "param.h"

typedef unsigned char uchar;
typedef uint64_t ulong;
typedef uint32_t uint;

#define checkCudaErrors(call)								\
do {														\
	cudaError_t err = call;									\
	if (cudaSuccess != err) {								\
		char errorBuff[512];								\
        sprintf(errorBuff, 			\
			"CUDA error '%s' in func '%s' line %d",			\
			cudaGetErrorString(err), __FUNCTION__, __LINE__);	\
		fprintf(stderr, "<error> %s\n", errorBuff); \
		}														\
} while (0)

static void internalSolve(mtm_context& ctx, const char* header, unsigned int header_len, const char* nonce, unsigned int nonce_len,
	std::function<bool()> cancelf,
	std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
	std::function<void(void)> hashdonef);

#define ENCODE_INPUTS(row, slot0, slot1) ((row << 20) | ((slot1 & 0x3ff) << 10) | (slot0 & 0x3ff))
#define DECODE_ROW(REF)   (REF >> 20)
#define DECODE_SLOT1(REF) ((REF >> 10) & 0x3ff)
#define DECODE_SLOT0(REF) (REF & 0x3ff)


typedef struct slot32_s
{
	uint4 x;//16 bytes
	uint4 y;//16 bytes
} slot32_t;

typedef struct row32_s
{
	slot32_t slots[NR_SLOTS];
} row32_t;

typedef struct row16_s
{
	uint4 slots[NR_SLOTS];
} row16_t;

typedef struct row8_s
{
	uint2 slots[NR_SLOTS];
} row8_t;

typedef struct table32_s
{
	row32_t	rows[NR_ROWS];
} table32_t;

typedef struct table16_s
{
	row16_t	rows[NR_ROWS];
} table16_t;

typedef struct table8_s
{
	row8_t	rows[NR_ROWS];
} table8_t;

typedef struct data_s
{
	
	candidate_t		candidates;
	table32_t		round0;
	table32_t		round1;
	table32_t		round2;
	table32_t		round3;
	table32_t		round4;
	table16_t		round5;
	table16_t		round6;
	table16_t		round7;
	table8_t		round8;
	uint			rowCounter0[NR_ROWS];
	uint			rowCounter1[NR_ROWS];
	uint			bin_counter[NR_ROWS * 512];
	uint			sols[1024];
	ulong			blake[16];
} data_t;

struct mtm_context
{
	mtm_context(int deviceNo)
	: m_device_no(deviceNo)
	{
		checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
		init();
	}

	void init()
	{
		checkCudaErrors(cudaSetDevice(m_device_no));
		checkCudaErrors(cudaDeviceReset());
		checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(data_t)));
		checkCudaErrors(cudaMalloc((void**)&d_blake_data, 128));
		checkCudaErrors(cudaMemset(d_data, 0, sizeof(data_t)));
		checkCudaErrors(cudaMallocHost(&h_candidates, sizeof(candidate_t)));
	}

	void destroy()
	{
		checkCudaErrors(cudaSetDevice(m_device_no));
		checkCudaErrors(cudaDeviceReset());
		//checkCudaErrors(cudaFreeHost(h_candidates));
	}

	int				m_device_no;
	data_t*			d_data;
	uint4*			d_blake_data;
	candidate_t*	h_candidates;
};

__device__ __forceinline__ void rotate32(ulong value, ulong* retval)
{
	uint2* ret = (uint2*)&value;
	uint2* ret2 = (uint2*)retval;
	ret2->y = ret->x;
	ret2->x = ret->y;
}

__device__ __forceinline__ void rotate40(ulong value, ulong* retval)
{
	uint2* ret = (uint2*)&value;
	uint2* ret2 = (uint2*)retval;
	ret2->y = __byte_perm(ret->x, ret->y, 0x2107);
	ret2->x = __byte_perm(ret->x, ret->y, 0x6543);
}

__device__ __forceinline__ void rotate48(ulong value, ulong* retval)
{
	uint2* ret = (uint2*)&value;
	uint2* ret2 = (uint2*)retval;
	ret2->y = __byte_perm(ret->x, ret->y, 0x1076);
	ret2->x = __byte_perm(ret->x, ret->y, 0x5432);
}

union ab
{
	ulong a; 
	uint2 b;
};

__device__ __forceinline__ ulong ROR2(const ulong aa, const uint offset) 
{
	uint2* aa_ = (uint2*)&aa;

	ab result;
	{
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.b.x) : "r"(aa_->y), "r"(aa_->x), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.b.y) : "r"(aa_->x), "r"(aa_->y), "r"(offset));
	}
	return result.a;
}

__device__ __forceinline__ uint get_lane_id()
{
	uint ret;
	asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}



#define rotate(a, bits) ((a) << (bits)) | ((a) >> (64 - (bits)))

#define mix(va, vb, vc, vd, x, y) \
va = (va + vb + x); \
rotate32(va ^ vd, &vd); \
vc = (vc + vd); \
rotate40(vb ^ vc, &vb); \
va = (va + vb + y); \
rotate48(vd ^ va, &vd); \
vc = (vc + vd); \
vb = ROR2(vb ^ vc, 63U);

//rotate((vb ^ vc), (ulong)64 - 63);

/*

asm volatile ("{\n\t"
".reg .v4 .u32 v1,v2,v3,v4,v5,v6,v7,v8;\n\t"
"ld.global.v4.u32 v1, [%16];\n\t"
"ld.global.v4.u32 v2, [%16+16];\n\t"
"ld.global.v4.u32 v3, [%16+32];\n\t"
"ld.global.v4.u32 v4, [%16+48];\n\t"
"ld.global.v4.u32 v5, [%16+64];\n\t"
"ld.global.v4.u32 v6, [%16+80];\n\t"
"ld.global.v4.u32 v7, [%16+96];\n\t"
"ld.global.v4.u32 v8, [%16+112];\n\t"
"mov.b64 %0, {v1.x, v1.y};\n\t"
"mov.b64 %1, {v1.z, v1.w};\n\t"
"mov.b64 %2, {v2.x, v2.y};\n\t"
"mov.b64 %3, {v2.z, v2.w};\n\t"
"mov.b64 %4, {v3.x, v3.y};\n\t"
"mov.b64 %5, {v3.z, v3.w};\n\t"
"mov.b64 %6, {v4.x, v4.y};\n\t"
"mov.b64 %7, {v4.z, v4.w};\n\t"
"mov.b64 %8, {v5.x, v5.y};\n\t"
"mov.b64 %9, {v5.z, v5.w};\n\t"
"mov.b64 %10, {v6.x, v6.y};\n\t"
"mov.b64 %11, {v6.z, v6.w};\n\t"
"mov.b64 %12, {v7.x, v7.y};\n\t"
"mov.b64 %13, {v7.z, v7.w};\n\t"
"mov.b64 %14, {v8.x, v8.y};\n\t"
"mov.b64 %15, {v8.z, v8.w};\n\t"
"}\n"
:
"=l"(sv[0]), "=l"(sv[1]), "=l"(sv[2]), "=l"(sv[3]),
"=l"(sv[4]), "=l"(sv[5]), "=l"(sv[6]), "=l"(sv[7]),
"=l"(sv[8]), "=l"(sv[9]), "=l"(sv[10]), "=l"(sv[11]),
"=l"(sv[12]), "=l"(sv[13]), "=l"(sv[14]), "=l"(sv[15])
: "l"(blake_data)
);
*/

/*
__device__ uint cnt0[4096];


template<int SIZE>
__global__
void test(const ulong (&blakey)[SIZE])
{
	ulong v[16];
	v[0] = blakey[0];

	uint idx = blockIdx.x;

	asm volatile (
		"ld.param.u64 %0, [%1];\n"
		: "=l"(v[0]) : "l"(blakey)
		);

	uint cnnnt;
	uint* cnt = cnt0 + idx;

	asm volatile (
		"ldu.global.u32 %0, [%1];\n"
		: "=r"(cnnnt) : "l"(cnt)
		);

}*/

__global__
__launch_bounds__(256)
void kernel_round0(data_t* data, const uint4 *  bla)
{
	uint* rowCounter = data->rowCounter0;
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;//1-1m for inputs
	uint laneid = get_lane_id();
	__shared__ uint4 s_sv[8];

	if (tid < 3)
	{
		data->candidates.sol_nr[tid] = 0;
	}

	ulong* sv = (ulong*)s_sv;
	
	union
	{
		ulong v[16];
		uint  v32[32];
		uint4 v_ui4[8];
	};

	if (threadIdx.x < 8)
	{
		s_sv[threadIdx.x] = bla[threadIdx.x];
	}

	__syncthreads();

	v_ui4[0] = s_sv[0];
	v_ui4[1] = s_sv[1];
	v_ui4[2] = s_sv[2];
	v_ui4[3] = s_sv[3];
	v_ui4[4] = s_sv[4];
	v_ui4[5] = s_sv[5];
	v_ui4[6] = s_sv[6];
	v_ui4[7] = s_sv[7];

	ulong word1 = ((ulong)tid) << 32;

	// round 1
	mix(v[0], v[4], v[8], v[12], 0, word1);
	mix(v[1], v[5], v[9], v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8], v[13], 0, 0);
	mix(v[3], v[4], v[9], v[14], 0, 0);
	// round 2
	mix(v[0], v[4], v[8], v[12], 0, 0);
	mix(v[1], v[5], v[9], v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], word1, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8], v[13], 0, 0);
	mix(v[3], v[4], v[9], v[14], 0, 0);
	// round 3
	mix(v[0], v[4], v[8], v[12], 0, 0);
	mix(v[1], v[5], v[9], v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8], v[13], 0, word1);
	mix(v[3], v[4], v[9], v[14], 0, 0);
	// round 4
	mix(v[0], v[4], v[8], v[12], 0, 0);
	mix(v[1], v[5], v[9], v[13], 0, word1);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8], v[13], 0, 0);
	mix(v[3], v[4], v[9], v[14], 0, 0);
	// round 5
	mix(v[0], v[4], v[8], v[12], 0, 0);
	mix(v[1], v[5], v[9], v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, word1);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8], v[13], 0, 0);
	mix(v[3], v[4], v[9], v[14], 0, 0);
	// round 6
	mix(v[0], v[4], v[8], v[12], 0, 0);
	mix(v[1], v[5], v[9], v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8], v[13], 0, 0);
	mix(v[3], v[4], v[9], v[14], word1, 0);
	// round 7
	mix(v[0], v[4], v[8], v[12], 0, 0);
	mix(v[1], v[5], v[9], v[13], word1, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8], v[13], 0, 0);
	mix(v[3], v[4], v[9], v[14], 0, 0);
	// round 8
	mix(v[0], v[4], v[8], v[12], 0, 0);
	mix(v[1], v[5], v[9], v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, word1);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8], v[13], 0, 0);
	mix(v[3], v[4], v[9], v[14], 0, 0);
	// round 9
	mix(v[0], v[4], v[8], v[12], 0, 0);
	mix(v[1], v[5], v[9], v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8], v[13], word1, 0);
	mix(v[3], v[4], v[9], v[14], 0, 0);
	// round 10
	mix(v[0], v[4], v[8], v[12], 0, 0);
	mix(v[1], v[5], v[9], v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], word1, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8], v[13], 0, 0);
	mix(v[3], v[4], v[9], v[14], 0, 0);
	// round 11
	mix(v[0], v[4], v[8], v[12], 0, word1);
	mix(v[1], v[5], v[9], v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8], v[13], 0, 0);
	mix(v[3], v[4], v[9], v[14], 0, 0);
	// round 12
	mix(v[0], v[4], v[8], v[12], 0, 0);
	mix(v[1], v[5], v[9], v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], word1, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8], v[13], 0, 0);
	mix(v[3], v[4], v[9], v[14], 0, 0);

	// compress v into the blake state; this produces the 50-byte hash
	// (two Xi values)
	v[0] = sv[0] ^ v[0] ^ v[8];
	v[1] = sv[1] ^ v[1] ^ v[9];
	v[2] = sv[2] ^ v[2] ^ v[10];
	v[3] = sv[3] ^ v[3] ^ v[11];
	v[4] = sv[4] ^ v[4] ^ v[12];
	v[5] = sv[5] ^ v[5] ^ v[13];
	v[6] = (sv[6] ^ v[6] ^ v[14]) & 0xffff;

	    uint row;

   //input li | hi
   //aaaa aaaa bbbb bbbb cccc cccc dddd dddd | eeee eeee ffff ffff gggg gggg hhhh hhhh
   //output
   //eeee eeee eeee eeee dddd dddd cccc cccc
   //row 
   //0000 eeee eeee eeee eeee dddd dddd cccc

	row = __byte_perm(v32[0], 0, 0x4401);
	row >>= 4;

	uint rowCnt = atomicAdd(rowCounter + row, 1);

	if (rowCnt < NR_SLOTS) {
		slot32_t slot;
		
		//input lo | hi
		//aaaa aaaa bbbb bbbb cccc cccc dddd dddd | eeee eeee ffff ffff gggg gggg hhhh hhhh
		//output
		//cccc cccc bbbb bbbb aaaa aaaa eeee eeee
		//0000 cccc bbbb bbbb aaaa aaaa eeee eeee

		slot.x.x = __byte_perm(v32[0], v32[1], 0x1234) & 0xFFFFFFF; 
		slot.x.y = __byte_perm(v32[1], v32[2], 0x1234);
		slot.x.z = __byte_perm(v32[2], v32[3], 0x1234);
		slot.x.w = __byte_perm(v32[3], v32[4], 0x1234);
		
		slot.y.x = __byte_perm(v32[4], v32[5], 0x1234);
		slot.y.y = __byte_perm(v32[5], v32[6], 0x1234);
		slot.y.z = tid << 1;
		slot.y.w = 0;

		data->round0.rows[row].slots[rowCnt].x = slot.x;
		data->round0.rows[row].slots[rowCnt].y = slot.y;
	}

	//input lo | hi
	//aaaa aaaa bbbb bbbb cccc cccc dddd dddd | eeee eeee ffff ffff gggg gggg hhhh hhhh
	//output
	//eeee eeee eeee eeee cccc cccc bbbb bbbb
	//row
	//0000 0000 0000 0000 0000 cccc cccc bbbb
	//we skip v32[6] dddd dddd because its used in previous thingy

	row = __byte_perm(v32[6], 0, 0x4412);
	row >>= 4;

	rowCnt = atomicAdd(rowCounter + row, 1);
	if (rowCnt < NR_SLOTS) {
		slot32_t slot;

		//input lo | hi
		//aaaa aaaa bbbb bbbb cccc cccc dddd dddd | eeee eeee ffff ffff gggg gggg hhhh hhhh
		//output
		//0000 bbbb aaaa aaaa eeee eeee ffff ffff
		//x is 28 bits
		slot.x.x = __byte_perm(v32[6], v32[7], 0x2345) & 0xFFFFFFF;
		slot.x.y = __byte_perm(v32[7], v32[8], 0x2345);
		slot.x.z = __byte_perm(v32[8], v32[9], 0x2345);
		slot.x.w = __byte_perm(v32[9], v32[10], 0x2345);
		
		slot.y.x = __byte_perm(v32[10], v32[11], 0x2345);
		slot.y.y = __byte_perm(v32[11], v32[12], 0x2345);
		slot.y.z = (tid << 1) + 1;
		slot.y.w = 0;
		data->round0.rows[row].slots[rowCnt].x = slot.x;
		data->round0.rows[row].slots[rowCnt].y = slot.y;
	}
}

__global__
__launch_bounds__(NR_SLOTS)
void kernel_round1(data_t* data)
{
    const uint BIN_CNT = 256;
    const uint MAX_COLL = 12;
    const uint MAX_COLL_IDX = MAX_COLL - 1;
    const uint COLL_SIZE = BIN_CNT * MAX_COLL;
	__shared__ uint16_t s_collisions[COLL_SIZE];
	__shared__ uint4 s_w0[NR_SLOTS];
	__shared__ uint2 s_w1[NR_SLOTS];
	__shared__ uint s_cnt[BIN_CNT];
	__shared__ uint s_count;

	uint count;
	uint idx = blockIdx.x;
	uint tid = threadIdx.x;
	uint laneid = get_lane_id();

	if (tid < BIN_CNT) {
		s_cnt[tid] = 0;
	}

	if (tid == 0)
	{
		s_count = min(NR_SLOTS, data->rowCounter0[idx]);
		data->rowCounter0[idx] = 0;
	}

    __syncthreads();

	if (laneid == 0) {
		count = s_count;
	}

	count = __shfl_sync(0xFFFFFFFF, count, 0);

	for (; tid < NR_SLOTS; tid += blockDim.x) {
		uint4 slot_0;
		uint2 slot_1;
		uint bin = 0;
		uint bin_idx = 0;
		if (tid < count) {
			slot_0 = data->round0.rows[idx].slots[tid].x;
			slot_1 = *(uint2*)&data->round0.rows[idx].slots[tid].y;
			s_w0[tid] = slot_0;
			s_w1[tid] = slot_1;
			//x is 28 bits
			//bin is upper 8 bits
			bin = slot_0.x >> 20;
			uint cnt = atomicAdd(&s_cnt[bin], 1);
			bin_idx = min(MAX_COLL_IDX, cnt);
			s_collisions[bin * MAX_COLL + bin_idx] = tid;
		}

		__syncthreads();

		if (bin_idx >= 1) {
			uint16_t* col_ptr = &s_collisions[bin * MAX_COLL];
			for (uint i = 0; i < bin_idx; i++, col_ptr++) {
				uint16_t col = *col_ptr;
				uint2 o_slot_1 = s_w1[col];
				if (slot_1.y != o_slot_1.y) {
					uint4 o_slot_0 = s_w0[col];
					//x will be (20 bits)
					//0000 0000 0000 AAAA bbbb BBBB cccc CCCC
					uint r36 = o_slot_0.x ^ slot_0.x;
					uint nextRow = __byte_perm(r36, 0, 0x4321) & 4095;
					uint nextSlot = atomicAdd(&data->rowCounter1[nextRow], 1);
					if (nextSlot < NR_SLOTS) {
						slot32_t to_slot;
						//x is 8 bits
						to_slot.x.x = r36 & 255;
						to_slot.x.y = o_slot_0.y ^ slot_0.y;
						to_slot.x.z = o_slot_0.z ^ slot_0.z;
						to_slot.x.w = o_slot_0.w ^ slot_0.w;
						data->round1.rows[nextRow].slots[nextSlot].x = to_slot.x;
						to_slot.y.y = o_slot_1.y ^ slot_1.y;
						to_slot.y.x = o_slot_1.x ^ slot_1.x;
						uint r76 = idx << 10;
						uint r77 = col | r76;
						uint r78 = r77 << 10;
						to_slot.y.z = r78 | tid;
						to_slot.y.w = 0;
						data->round1.rows[nextRow].slots[nextSlot].y = to_slot.y;
					}
				}
			}
		}
	}
}


__global__
__launch_bounds__(NR_SLOTS)
void kernel_round2(data_t* data)
{
    const uint BIN_CNT = 256;
    const uint MAX_COLL = 12;
    const uint MAX_COLL_IDX = MAX_COLL - 1;
    
	__shared__ uint16_t s_collisions[BIN_CNT * MAX_COLL];
	__shared__ uint4 s_w0[NR_SLOTS];
	__shared__ uint2 s_w1[NR_SLOTS];
	__shared__ uint s_cnt[BIN_CNT];
	__shared__ uint s_row_count;

	uint idx = blockIdx.x;
	uint count;
	uint tid = threadIdx.x;
	uint laneid = get_lane_id();

	if (tid < BIN_CNT)
	{
		s_cnt[tid] = 0;
	}

	if (tid == 0) {
		s_row_count = min(data->rowCounter1[idx], NR_SLOTS);
		data->rowCounter1[idx] = 0;
	}

	__syncthreads();

	if (laneid == 0)
	{
		count = s_row_count;
	}

	count = __shfl_sync(0xFFFFFFFF, count, 0);

	for (; tid < NR_SLOTS; tid += blockDim.x) {
		uint bin_idx = 0;
		uint4 slot_0;
		uint2 slot_1;
		uint bin = 0;

		if (tid < count) {
			slot_0 = data->round1.rows[idx].slots[tid].x;
			slot_1 = *(uint2*)&data->round1.rows[idx].slots[tid].y.x;
			s_w0[tid] = slot_0;
			s_w1[tid] = slot_1;
			//x is 8 bits
			//bin is x
			bin = slot_0.x;
			uint cnt = atomicAdd(&s_cnt[bin], 1);
			bin_idx = min(cnt, MAX_COLL_IDX);
			s_collisions[bin * MAX_COLL + bin_idx] = tid;
		}

		__syncthreads();

		if (bin_idx >= 1) {
			uint16_t* col_ptr = &s_collisions[bin * MAX_COLL];
			for (uint n = 0; n < bin_idx; n++, col_ptr++) {
				uint16_t col = *col_ptr;
				uint2 o_slot_1 = s_w1[col];
				if (slot_1.y != o_slot_1.y) {
					uint4 o_slot_0 = s_w0[col];
					//y is full 32 bits
					uint r33 = o_slot_0.y ^ slot_0.y;
					//use upper 12 bits for row
					uint nextRow = r33 >> 20;
					uint nextSlot = atomicAdd(&data->rowCounter0[nextRow], 1);
					if (nextSlot < NR_SLOTS) {
						//could be slot24_t last 8 bytes are not used
						slot32_t to_slot;
						//x is 20 bits, bottom of y
						to_slot.x.x = r33 & 0xFFFFFU;
						to_slot.x.y = o_slot_0.z ^ slot_0.z;
						to_slot.x.z = o_slot_0.w ^ slot_0.w;
						to_slot.x.w = o_slot_1.x ^ slot_1.x;
						data->round2.rows[nextRow].slots[nextSlot].x = to_slot.x;
						to_slot.y.x = o_slot_1.y ^ slot_1.y;
						uint r69 = idx << 10;
						uint r70 = col | r69;
						uint r71 = r70 << 10;
						to_slot.y.y = r71 | tid;
						to_slot.y.w = to_slot.y.z = 0;
						data->round2.rows[nextRow].slots[nextSlot].y = to_slot.y;
					}
				}
			}
		}
	}
}


__global__
__launch_bounds__(NR_SLOTS)
void kernel_round3(data_t* data)
{
    const uint BIN_CNT = 256;
    const uint MAX_COLL = 12;
    const uint MAX_COLL_IDX = MAX_COLL - 1;

	__shared__ uint16_t s_collisions[BIN_CNT * MAX_COLL];
	__shared__ uint4 s_w0[NR_SLOTS];
	__shared__ uint s_w1[NR_SLOTS];
	__shared__ uint s_count;
	__shared__ uint s_cnt[BIN_CNT];

	
	uint idx = blockIdx.x;
	uint count;
	uint tid = threadIdx.x;
	
	uint laneid = get_lane_id();
	if (tid < BIN_CNT) {
		s_cnt[tid] = 0;
	}

	if (tid == 0) 
	{
		s_count = min(data->rowCounter0[idx], NR_SLOTS);
		data->rowCounter0[idx] = 0;
	}

	__syncthreads();

	if (laneid == 0)
	{
		count = s_count;
	}

	count = __shfl_sync(0xFFFFFFFF, count, 0);

	for (; tid < NR_SLOTS; tid += blockDim.x)
	{
		uint4 slot_0;
		uint  slot_1;
		uint bin = 0;
		uint bin_idx = 0;

		if (tid < count)
		{
			slot_0 = data->round2.rows[idx].slots[tid].x;
			slot_1 = *(uint*)&data->round2.rows[idx].slots[tid].y;
			s_w0[tid] = slot_0;
			s_w1[tid] = slot_1;
			//x is 20 bits
			//bin is upper 8
			bin = slot_0.x >> 12;
			uint cnt = atomicAdd(&s_cnt[bin], 1);
			bin_idx = min(cnt, MAX_COLL_IDX);
			s_collisions[bin * MAX_COLL + bin_idx] = tid;
		}

		__syncthreads();

		if (bin_idx >= 1) 
		{
			uint16_t* col_ptr = &s_collisions[bin * MAX_COLL];
			for (uint n = 0; n < bin_idx; n++, col_ptr++)
			{
				uint16_t col = *col_ptr;
				//we could see if col is in our wave and pull it from that thread reg
				uint o_slot_1 = s_w1[col];
				if (slot_1 != o_slot_1)
				{
					uint4 o_slot_0 = s_w0[col];
					//x will be
					//0000 0000 0000 0000 0000 AAAA bbbb BBBB
					uint nextRow = o_slot_0.x ^ slot_0.x;
					uint nextSlot = atomicAdd(&data->rowCounter1[nextRow], 1);
					if (nextSlot < NR_SLOTS)
					{
						slot32_t to_slot;
						//x is 32 bits
						to_slot.x.x = o_slot_0.y ^ slot_0.y;
						to_slot.x.y = o_slot_0.z ^ slot_0.z;
						to_slot.x.z = o_slot_0.w ^ slot_0.w;
						to_slot.x.w = o_slot_1 ^ slot_1;
						data->round3.rows[nextRow].slots[nextSlot].x = to_slot.x;
						uint r62 = idx << 10;
						uint r63 = col | r62;
						uint r64 = r63 << 10;
						to_slot.y.x = r64 | tid;
						to_slot.y.y = to_slot.y.z = to_slot.y.w = 0;
						//could we get away with writing uint2? would that benefit anything?
						data->round3.rows[nextRow].slots[nextSlot].y = to_slot.y;
					}
				}
			}
		}
	}
}


__global__
__launch_bounds__(NR_SLOTS)
void kernel_round4(data_t* data)
{
    const uint BIN_CNT = 256;
    const uint MAX_COLL = 20;
    const uint MAX_COLL_IDX = MAX_COLL - 1;

	__shared__ uint16_t s_collisions[BIN_CNT * MAX_COLL];
	__shared__ uint4 s_w0[NR_SLOTS];
	__shared__ uint s_count;
	__shared__ uint s_cnt[BIN_CNT];

	uint idx = blockIdx.x;
	uint count;
	uint tid = threadIdx.x;
	uint laneid = get_lane_id();

    if (tid < BIN_CNT)
    {
		s_cnt[tid] = 0;
	}

	if (tid == 0)
	{
		s_count = min(data->rowCounter1[idx], NR_SLOTS);
		data->rowCounter1[idx] = 0;
	}

	__syncthreads();

	if (laneid == 0)
	{
		count = s_count;
	}

	count = __shfl_sync(0xFFFFFFFF, count, 0);

	for (; tid < NR_SLOTS; tid += blockDim.x) {
		uint4 slot_0;
		uint bin = 0;
		uint bin_idx = 0;

		if (tid < count) {
			slot_0 = data->round3.rows[idx].slots[tid].x;
			s_w0[tid] = slot_0;
			//x is 32 bits
			//bin is upper 8 bits
			bin = slot_0.x >> 24;
			uint cnt = atomicAdd(&s_cnt[bin], 1);
			bin_idx = min(cnt, MAX_COLL_IDX);
			s_collisions[bin * MAX_COLL + bin_idx] = tid;
		}

		__syncthreads();

		if (bin_idx >= 1) {
			uint col_idx = bin * MAX_COLL;
			for (uint n = 0; n < bin_idx; n++, col_idx++) {
				uint16_t col = s_collisions[col_idx];
				uint4 o_slot_0 = s_w0[col];
				if (o_slot_0.w != slot_0.w)
				{
					//x will be
					//0000 0000 aaaa AAAA bbbb BBBB cccc CCCC
					uint r28 = o_slot_0.x ^ slot_0.x;
					//row is upper 12 bits
					uint nextRow = r28 >> 12;
					uint nextSlot = atomicAdd(&data->rowCounter0[nextRow], 1);
					if (nextSlot < NR_SLOTS) {
						slot32_t to_slot;
						//x is 12 bits
						to_slot.x.x = r28 & 0xFFFU;
						to_slot.x.y = o_slot_0.y ^ slot_0.y;
						to_slot.x.z = o_slot_0.z ^ slot_0.z;
						to_slot.x.w = o_slot_0.w ^ slot_0.w;
						data->round4.rows[nextRow].slots[nextSlot].x = to_slot.x;
						uint r59 = idx << 10;
						uint r60 = col | r59;
						uint r61 = r60 << 10;
						to_slot.y.x = r61 | tid;
						to_slot.y.y = to_slot.y.z = to_slot.y.w = 0;
						data->round4.rows[nextRow].slots[nextSlot].y = to_slot.y;
					}
				}
			}
		}
	}
}

__global__
__launch_bounds__(NR_SLOTS)
void kernel_round5(data_t* data)
{
    const uint BIN_CNT = 256;
    const uint MAX_COLL = 12;
    const uint MAX_COLL_IDX = MAX_COLL - 1;

	__shared__ uint16_t s_collisions[BIN_CNT * MAX_COLL];
	__shared__ uint4 s_w0[NR_SLOTS];
	__shared__ uint s_count;
	__shared__ uint s_cnt[BIN_CNT];

	uint idx = blockIdx.x;
	uint count;
	uint tid = threadIdx.x;
	uint laneid = get_lane_id();

	if (tid < BIN_CNT)
	{
		s_cnt[tid] = 0;
	}

	if (tid == 0)
	{
		s_count = min(data->rowCounter0[idx], NR_SLOTS);
		data->rowCounter0[idx] = 0;
	}

	__syncthreads();

	if (laneid == 0)
	{
		count = s_count;
	}

	count = __shfl_sync(0xFFFFFFFF, count, 0);

	for (; tid < NR_SLOTS; tid += blockDim.x) {
		uint4 slot_0;
		uint bin = 0;
		uint bin_idx = 0;

		if (tid < count) {
			slot_0 = data->round4.rows[idx].slots[tid].x;
			s_w0[tid] = slot_0;
			//x is 12 bits
			//bin is upper 8 bits
			bin = slot_0.x >> 4;
			uint cnt = atomicAdd(&s_cnt[bin], 1);
			bin_idx = min(cnt, MAX_COLL_IDX);
			s_collisions[bin * MAX_COLL + bin_idx] = tid;
		}

		__syncthreads();

		if (bin_idx >= 1) {
			uint col_idx = bin * MAX_COLL;
			for (uint n = 0; n < bin_idx; n++, col_idx++) {
				uint16_t col = s_collisions[col_idx];
				uint4 o_slot_0 = s_w0[col];
				if (slot_0.w != o_slot_0.w) 
				{
					//x will be 
					//0000 0000 0000 0000 0000 0000 0000 aaaa
					uint r53 = o_slot_0.x ^ slot_0.x;
					uint r28 = o_slot_0.y ^ slot_0.y;
					//borrow a byte from y
					uint nextRow = __byte_perm(r53, r28, 0x3307);
					uint nextSlot = atomicAdd(&data->rowCounter1[nextRow], 1);
					if (nextSlot < NR_SLOTS) {
						uint4 out_slot;
						//x is 24 bits
						out_slot.x = r28 & 0xFFFFFFU;
						out_slot.y = o_slot_0.z ^ slot_0.z;
						out_slot.z = o_slot_0.w ^ slot_0.w;
						uint r60 = idx << 10;
						uint r61 = col | r60;
						uint r62 = r61 << 10;
						out_slot.w = r62 | tid;
						data->round5.rows[nextRow].slots[nextSlot] = out_slot;
					}
				}
			}
		}
	}
}

__global__
__launch_bounds__(NR_SLOTS)
void kernel_round6(data_t* data)
{
    const uint BIN_CNT = 256;
    const uint MAX_COLL = 12;
    const uint MAX_COLL_IDX = MAX_COLL - 1;

	__shared__ uint16_t s_collisions[BIN_CNT * MAX_COLL];
	__shared__ uint4 s_w0[NR_SLOTS];
	__shared__ uint s_row_count;
	__shared__ uint s_cnt[BIN_CNT];

	uint idx = blockIdx.x;
	//represents processing a slot
	uint tid = threadIdx.x;
	uint count;
	uint laneid = get_lane_id();

	if (tid < BIN_CNT)
	{
		s_cnt[tid] = 0;
	}

	if (tid == 0)
	{
		s_row_count = min(data->rowCounter1[idx], NR_SLOTS);
		data->rowCounter1[idx] = 0;
	}

	__syncthreads();

	if (laneid == 0)
	{
		count = s_row_count;
	}

	count = __shfl_sync(0xFFFFFFFF, count, 0);

	for (; tid < NR_SLOTS; tid += blockDim.x) {
		uint4 slot_0;
		uint bin = 0;
		uint bin_idx = 0;

		if (tid < count)
		{
			slot_0 = data->round5.rows[idx].slots[tid];
			s_w0[tid] = slot_0;
			//x is 24 bits
			//bin is upper 8 bits
			bin = slot_0.x >> 16;
			uint cnt = atomicAdd(&s_cnt[bin], 1);
			bin_idx = min(cnt, MAX_COLL_IDX);
			s_collisions[bin * MAX_COLL + bin_idx] = tid;
		}

		__syncthreads();

		if (bin_idx >= 1)
		{
			uint col_idx = bin * MAX_COLL;
			for (uint n = 0; n < bin_idx; n++, col_idx++)
			{
				uint16_t col = s_collisions[col_idx];
				uint4 o_slot_0 = s_w0[col];
				if (slot_0.z != o_slot_0.z)
				{
					//x will be:
					//0000 0000 0000 0000 aaaa AAAA bbbb BBBB
					uint r25 = o_slot_0.x ^ slot_0.x;
					//row is upper 12 bits leaving a nibble
					uint nextRow = r25 >> 4;
					uint nextSlot = atomicAdd(&data->rowCounter0[nextRow], 1);
					if (nextSlot < NR_SLOTS)
					{
						uint4 to_slot;
						uint r52 = r25 & 0xFU;
						uint r53 = r52 << 4;
						uint r54 = o_slot_0.y ^ slot_0.y;
						//x is 4 bits, borrow 4 bits from y, in total x is 8 bits
						uint r55 = r54 >> 28;
						to_slot.x = r53 | r55;
						//y is 28 bits
						to_slot.y = r54 & 0xFFFFFFFU;
						to_slot.z = o_slot_0.z  ^ slot_0.z;
						uint r58 = idx << 10;
						uint r59 = col | r58;
						uint r60 = r59 << 10;
						to_slot.w = r60 | tid;
						
						data->round6.rows[nextRow].slots[nextSlot] = to_slot;
					}
				}
			}
		}
	}
}

__global__
__launch_bounds__(NR_SLOTS)
void kernel_round7(data_t* data)
{
	//512,256 - 1200 sols/s
	//128 - 2000 sols/s (must be not correct)
    const uint BIN_CNT = 256;
    const uint MAX_COLL = 12;
    const uint MAX_COLL_IDX = MAX_COLL - 1;

	__shared__ uint16_t s_collisions[BIN_CNT * MAX_COLL];
	__shared__ uint4 s_w0[NR_SLOTS];
	__shared__ uint s_count;
	__shared__ uint s_cnt[BIN_CNT];

	uint idx = blockIdx.x;
	uint count;
	uint tid = threadIdx.x;
	uint laneid = get_lane_id();

	if (tid < BIN_CNT) 
	{
		s_cnt[tid] = 0;
	}

	if (tid == 0) 
	{
		s_count = min(data->rowCounter0[idx], NR_SLOTS);
		data->rowCounter0[idx] = 0;
	}

	__syncthreads();

	if (laneid == 0)
	{
		count = s_count;
	}

	count = __shfl_sync(0xFFFFFFFF, count, 0);

	for (; tid < NR_SLOTS; tid += blockDim.x) {
		uint4 slot_0;
		uint bin = 0;
		uint bin_idx = 0;

		if (tid < count) {
			slot_0 = data->round6.rows[idx].slots[tid];
			s_w0[tid] = slot_0;
			//x is 8 bits
			bin = slot_0.x;
			uint cnt = atomicAdd(&s_cnt[bin], 1);
			bin_idx = min(cnt, MAX_COLL_IDX);
			s_collisions[bin * MAX_COLL + bin_idx] = tid;
		}

		__syncthreads();

		if (bin_idx >= 1) {
			uint16_t* col_ptr = &s_collisions[bin * MAX_COLL];
			for (uint n = 0; n < bin_idx; n++, col_ptr++) {
				uint16_t col = *col_ptr;
				uint4 o_slot_0 = s_w0[col];
				if (slot_0.z != o_slot_0.z) {
					//y is 28 bits
					uint r22 = o_slot_0.y ^ slot_0.y;
					//row uses upper 12 bits of y
					uint nextRow = r22 >> 16;
					uint nextSlot = atomicAdd(&data->rowCounter1[nextRow], 1);
					if (nextSlot < NR_SLOTS) {
						uint4 to_slot;
						//x is 16 bits
						to_slot.x = r22 & 0xFFFFU;
						to_slot.y = o_slot_0.z ^ slot_0.z;
						uint r51 = idx << 10;
						uint r52 = col | r51;
						uint r53 = r52 << 10;
						to_slot.z = r53 | tid;
						to_slot.w = 0;
						data->round7.rows[nextRow].slots[nextSlot] = to_slot;
					}
				}
			}
		}
	}
}



__global__
__launch_bounds__(NR_SLOTS)
void kernel_round8(data_t* data)
{
	//512 - 320
	//256 - 1200
	//128 - 800
    const uint BIN_CNT = 256;
    const uint MAX_COLL = 32;
    const uint MAX_COLL_IDX = MAX_COLL - 1;
	const uint BIN_SHIFT = 8;

	__shared__ uint16_t s_collisions[BIN_CNT * MAX_COLL];
	__shared__ uint2 s_w0[NR_SLOTS];
	__shared__ uint s_cnt[BIN_CNT];
	__shared__ uint s_count;

	//represents a row 0-4095 (12 bit)
	uint idx = blockIdx.x;
	uint count;
	//represents a slot 0-607 (10 bit)
	uint tid = threadIdx.x;
	uint laneid = get_lane_id();

	if (tid < BIN_CNT)
	{
		s_cnt[tid] = 0;
	}

	if (tid == 0)
	{
		s_count = min(data->rowCounter1[idx], NR_SLOTS); 
		data->rowCounter1[idx] = 0;
	}

	__syncthreads();

	if (laneid == 0)
	{
		count = s_count;
	}

	count = __shfl_sync(0xFFFFFFFF, count, 0);

	for (; tid < NR_SLOTS; tid += blockDim.x) {
		uint2 slot_0;
		uint bin = 0;
		uint bin_idx = 0;

		if (tid < count) {
			slot_0 = *(uint2*)&data->round7.rows[idx].slots[tid];
			s_w0[tid] = slot_0;
			//x is 16 bits
			//0000 0000 0000 0000 aaaa AAAA bbbb BBBB
			//bin is
			//aaaa AAAA
			bin = slot_0.x >> BIN_SHIFT;
			uint cnt = atomicAdd(&s_cnt[bin], 1);
			bin_idx = min(cnt, MAX_COLL_IDX);
			s_collisions[bin * MAX_COLL + bin_idx] = tid;
		}

		__syncthreads();

		if (bin_idx >= 1)
		{
			uint16_t* col_ptr = &s_collisions[bin * MAX_COLL];
			for (uint n = 0; n < bin_idx; n++, col_ptr++) {
				//represents a slot 0-607 (10 bit)
				uint16_t col = *col_ptr;
				uint2 o_slot_0 = s_w0[col];
				if (slot_0.y != o_slot_0.y) {
					//x is 8 bits
					//0000 0000 0000 0000 0000 0000 aaaa AAAA
					uint r40 = o_slot_0.x ^ slot_0.x;
					uint r20 = o_slot_0.y ^ slot_0.y;
					uint r42 = r40 << 4;
					//borrow 4 bits from y to make a row
					uint r43 = r20 >> 28;
					uint nextRow = r42 | r43;
					uint nextSlot = atomicAdd(&data->rowCounter0[nextRow], 1);
					if (nextSlot < NR_SLOTS) {
						uint2 to_slot;
						//x is 28 bits
						to_slot.x = r20 & 0xFFFFFFFU;
						uint r47 = idx << 10;
						uint r48 = col | r47;
						uint r49 = r48 << 10;
						//y is r=row c=collison slot t=slot
						//rrrr rrrr rrrr cccc cccc cctt tttt tttt
						to_slot.y = r49 | tid;
						data->round8.rows[nextRow].slots[nextSlot] = to_slot;
					}
				}
			}
		}
	}
}

__global__
__launch_bounds__(NR_SLOTS)
  void kernel_round9(data_t* data)
{
	//256 - 1200
	//512 - 800
    const uint BIN_CNT = 256;
    const uint MAX_COLL = 32;
    const uint MAX_COLL_IDX = MAX_COLL - 1;


	__shared__ uint		s_bincount[BIN_CNT];//256 counters, each index is the value, count is the current index
	__shared__ uint2	s_slot1[NR_SLOTS];//first 16 bytes
	__shared__ uint16_t	s_collisions[BIN_CNT * MAX_COLL];
	__shared__ uint 	s_count;
	
	uint idx = blockIdx.x;
	uint tid = threadIdx.x;
	uint count;
	uint laneid = get_lane_id();

	if (tid < BIN_CNT)
	{
		s_bincount[tid] = 0;//reset bin count
	}

	if (tid == 0)
	{
		s_count = min(data->rowCounter0[idx], NR_SLOTS);
		data->rowCounter0[idx] = 0;//reset counter after we read it
	}

	__syncthreads();

	if (laneid == 0)
	{
		count = s_count;
	}

	count = __shfl_sync(0xFFFFFFFF, count, 0);

	for (; tid < NR_SLOTS; tid += blockDim.x)
	{
		uint bin_idx = 0;
		uint2 slot1;
		uint bin = 0;

		if (tid < count)
		{
			slot1 = data->round8.rows[idx].slots[tid];
			s_slot1[tid] = slot1;
			//x is 28 bits
			//0000 AAAA bbbb BBBB cccc CCCC dddd DDDD
			//so bin is
			//AAAA bbbb
			bin = slot1.x >> 20;
			uint cnt = atomicAdd(&s_bincount[bin], 1);
			bin_idx = min(cnt, MAX_COLL_IDX);
			s_collisions[bin * MAX_COLL + bin_idx] = tid;
		}

		__syncthreads();

		if (bin_idx >= 1)
		{
			uint col_idx = bin * MAX_COLL;
			for (uint i = 0; i < bin_idx; i++, col_idx++)
			{
				uint col = s_collisions[col_idx];
				uint2 other_slot1 = s_slot1[col];
				//by the time we are here x is:
				//0000 0000 0000 BBBB cccc CCCC dddd DDDD
				if (other_slot1.x == slot1.x)
				{
					//look for duplicates
					//y is r=row c=collison slot t=slot
					//rrrr rrrr rrrr cccc cccc cctt tttt tttt
					uint tmp = other_slot1.y ^ slot1.y;
					bool rc = true;
					//test if row matched, if they are then we need to check to make sure the slots are unique
					if (tmp < 0x100000U)
					{
						//part 1 - collision slot doesnt equal
						//part 2 - slot doesnt equal
						rc = ((tmp & 0xFFC00U) != 0) && ((tmp & 0x3FFU) != 0);
						//bonus if same row, then check to see if all combos are unique
						//test other slot with collision slot, make sure its unique
						rc = rc && ((other_slot1.y ^ (slot1.y >> 10)) & 0x3FF) != 0;
						//test other collision slot with slot
						rc = rc && ((slot1.y ^ (other_slot1.y >> 10)) & 0x3FF) != 0;
					}
					if (rc)
					{
						//increase sol match
						uint cnt = atomicAdd(&data->candidates.sol_nr[2], 1);
						if (cnt < 1024)
						{
							data->sols[cnt] = ((col | (idx << 10)) << 10) | tid;
						}
					}
				}
			}
		}
	}
}


__global__
__launch_bounds__(128)
void kernel_candidates(data_t* data)
{
	__shared__ uint s_candidate[512];
	__shared__ uint16_t s_test[2048];
	__shared__ uint s_cnts[64];
	__shared__ uint s_is_col;
	__shared__ uint s_sol_num;
	__shared__ uint s_cnt;
	__shared__ uint s_encoded_row;

	uint tid = threadIdx.x;
	uint idx = blockIdx.x;
	uint laneid = get_lane_id();
	uint cnt;
	uint encoded_row;

	if (tid == 0)
	{
		s_is_col = 0;
		uint tmpCnt = data->candidates.sol_nr[2];//r83;
		if (tmpCnt < idx)
		{
			s_encoded_row = data->sols[idx];
		}
		s_cnt = tmpCnt;
	}

	__syncthreads();

	if (laneid == 0)
	{//4 warp means 4 reads
		cnt = s_cnt;
		if (idx < cnt)
		{
			encoded_row = s_encoded_row;
		}
	}

	cnt = __shfl_sync(0xFFFFFFFF, cnt, 0);

	if (idx >= cnt) 
	{
		//all threads die here
		return;
	}

	for (int i = tid; i < 64; i += blockDim.x)
	{
		s_cnts[i] = 0;
	}

	encoded_row = __shfl_sync(0xFFFFFFFF, encoded_row, 0);

	uint sl_row = encoded_row >> 20;
	uint slot_a = encoded_row >> 10;
	//uint block_count = blockDim.x;
	//uint addr = row * 4864;
	//char* addr_p = (char*)data + (row * 4864);
	for(uint n = tid; n < 512; n += blockDim.x) 
	{
		uint sl_slot = (n < 256 ? slot_a : encoded_row) & 0x3FF;
		//round 8 is 8 bytes
		uint r8_enc = data->round8.rows[sl_row].slots[sl_slot].y;//4
		uint r8_row = r8_enc >> 20;
		uint r8_slot = ((n & 128) < 128 ? (r8_enc >> 10) : r8_enc) & 0x3FF;
		//round 7 is 16 bytes
		uint r7_enc = data->round7.rows[r8_row].slots[r8_slot].z;//8
		uint r7_row = r7_enc >> 20;
		uint r7_slot = ((n & 64) < 64 ? (r7_enc >> 10) : r7_enc) & 0x3FF;
		//round 6 is 16 bytes
		uint r6_enc = data->round6.rows[r7_row].slots[r7_slot].w;//12
		uint r6_row = r6_enc >> 20;
		uint r6_slot = ((n & 32) < 32 ? (r6_enc >> 10) : r6_enc) & 0x3FF;
		//round 5 is 16 bytes ?
		uint r5_enc = data->round5.rows[r6_row].slots[r6_slot].w;//12
		uint r5_row = r5_enc >> 20;
		uint r5_slot = ((n & 16) < 16 ? (r5_enc >> 10) : r5_enc) & 0x3FF;
		//round 4 is 32 bytes ?
		uint r4_enc = data->round4.rows[r5_row].slots[r5_slot].y.x;//16
		uint r4_row = r4_enc >> 20;
		uint r4_slot = ((n & 8) < 8 ? (r4_enc >> 10) : r4_enc) & 0x3FF;
		//round 3 is 32 bytes
		uint r3_enc = data->round3.rows[r4_row].slots[r4_slot].y.x;//16
		uint r3_row = r3_enc >> 20;
		uint r3_slot = ((n & 4) < 4 ? (r3_enc >> 10) : r3_enc) & 0x3FF;
		//round 2 is 32 bytes
		uint r2_enc = data->round2.rows[r3_row].slots[r3_slot].y.y;//20
		uint r2_row = r2_enc >> 20;
		uint r2_slot = ((n & 2) < 2 ? (r2_enc >> 10) : r2_enc) & 0x3FF;
		//round 1 is 32 bytes
		uint r1_enc = data->round1.rows[r2_row].slots[r2_slot].y.z;//24
		uint r1_row = r1_enc >> 20;
		uint r1_slot = (((n & 1) != 1) ? (r1_enc >> 10) : r1_enc) & 0x3FF;
		//round 0 is 32 bytes
		uint r0_enc = data->round0.rows[r1_row].slots[r1_slot].y.z;//24
		s_candidate[n] = r0_enc;
	}

	__syncthreads();

	for (uint n = tid; n < 512; n += blockDim.x)
	{
		uint cand = s_candidate[n];
		uint cand_xx = cand & 63;
		uint cnt = atomicAdd(&s_cnts[cand_xx], 1);

		if (cnt < 32) {
			s_test[cand_xx * 32 + cnt] = cand >> 6;
		}
	}

	__syncthreads();

	uint t1 = tid >> 31;
	uint t2 = tid + t1;
	uint t3 = t2 >> 1;
	uint l_cnt = s_cnts[t3];
	l_cnt = min(l_cnt, 32);
	uint cnt2 = l_cnt - 1;
	uint cnt3 = cnt2 * l_cnt;
	uint cnt4 = cnt3 >> 31;
	uint cnt5 = cnt3 + cnt4;
	uint cnt_max = cnt5 >> 1;
	uint t4 = t2 & 4294967294U;
	uint t5 = tid - t4;

	if (t5 < cnt_max) {
		for (uint n = t5; n < cnt_max; n += 2) {
			uint d1 = ((n + (n / l_cnt)) + 1) % l_cnt;
			uint d2 = n % l_cnt;
			uint16_t r1 = s_test[t3 * 32 + d1];
			uint16_t r2 = s_test[t3 * 32 + d2];
			if (r1 == r2) {
				atomicAdd(&s_is_col, 1);
			}
		}
	}

	__syncthreads();

	if (s_is_col != 0)
	{ 
		//all threads can die here
		return; 
	}

	const uint tid_idx = tid * 4;
	uint cand1 = s_candidate[tid_idx + 1];
	uint cand2 = s_candidate[tid_idx];

	if (cand2 > cand1) {
		s_candidate[tid_idx] = cand1;
		s_candidate[tid_idx + 1] = cand2;
	}
	cand1 = s_candidate[tid_idx + 3];
	cand2 = s_candidate[tid_idx + 2];

	if (cand2 > cand1) {
		s_candidate[tid_idx + 2] = cand1;
		s_candidate[tid_idx + 3] = cand2;
	}
	__syncthreads();

	cand1 = s_candidate[tid_idx + 2];
	cand2 = s_candidate[tid_idx];

	if (cand2 > cand1) {
		uint tid_idx2 = tid_idx + 2;
		for (uint n = tid * 4; n < tid_idx2; n++) {
			uint u1 = s_candidate[n + 2];
			s_candidate[n] = u1;
			s_candidate[n + 2] = cand2;
			cand2 = s_candidate[n + 1];
		}
	}

	__syncthreads();

	uint tt3 = t3 * 4;
	uint cand2_1 = s_candidate[tt3 + 4];
	uint cand2_2 = s_candidate[tt3];

	if (cand2_2 > cand2_1) {
		uint n_max = (t5 << 1) + tt3 + 1;
		for (uint n = (t3 * 8 + (t5 << 1)) - 1; n < n_max; n++) {
			uint tr1 = s_candidate[n];
			uint tr2 = s_candidate[n + 4];
			s_candidate[n] = tr2;
			s_candidate[n + 4] = tr1;
		}

	}

	__syncthreads();

	uint r165 = tid >> 31;
	uint r166 = r165 >> 30;
	uint r167 = tid + r166;
	uint r39 = r167 >> 2;
	uint r40 = r39 << 4;
	uint r168 = s_candidate[r40 + 8];
	uint r169 = s_candidate[r40];

	if (r169 > r168) {
		uint r173 = r167 & 2147483644;
		uint r174 = tid - r173;
		uint r175 = r174 << 1;
		uint r176 = r175 + r40;
		uint r41 = r176 + 1;
		uint r177 = r39 * 16 + r175;
		for (uint r269 = r177; r269 < r41; r269++) {
			uint r178 = s_candidate[r269];
			uint r179 = s_candidate[r269 + 8];
			s_candidate[r269] = r179;
			s_candidate[r269 + 8] = r178;
		}
	}

	__syncthreads();

	uint r181 = r165 >> 29;
	uint r182 = tid + r181;
	uint r45 = r182 >> 3;
	uint r46 = r45 << 5;
	uint r183 = s_candidate[r46 + 16];
	uint r184 = s_candidate[r46];

	if (r184 > r183) {
		uint r188 = r182 & 2147483640;
		uint r189 = tid - r188;
		uint r190 = r189 << 1;
		uint r191 = r190 + r46;
		uint r47 = r191 + 1;
		uint r192 = r45 * 32 + r190;
		for (uint r270 = r192; r270 < r47; r270++) {
			uint r193 = s_candidate[r270];
			uint r194 = s_candidate[r270 + 16];
			s_candidate[r270] = r194;
			s_candidate[r270 + 16] = r193;
		}
	}

	__syncthreads();

	uint r196 = r165 >> 28;
	uint r197 = tid + r196;
	uint r51 = r197 >> 4;
	uint r52 = r51 << 6;
	uint r198 = s_candidate[r52 + 32];
	uint r199 = s_candidate[r52];

	if (r199 > r198) {
		uint r203 = r197 & 2147483632;
		uint r204 = tid - r203;
		uint r205 = r204 << 1;
		uint r206 = r205 + r52;
		uint r53 = r206 + 1;
		uint r207 = r51 * 64 + r205;
		for (uint r271 = r207; r271 < r53; r271++) {
			uint r208 = s_candidate[r271];
			uint r209 = s_candidate[r271 + 32];
			s_candidate[r271] = r209;
			s_candidate[r271 + 32] = r208;
		}
	}

	__syncthreads();

	uint r211 = r165 >> 27;
	uint r212 = tid + r211;
	uint r57 = r212 >> 5;
	uint r58 = r57 << 7;
	uint r213 = s_candidate[r58 + 64];
	uint r214 = s_candidate[r58];


	if (r214 > r213) {
		uint r218 = r212 & 2147483616;
		uint r219 = tid - r218;
		uint r220 = r219 << 1;
		uint r221 = r220 + r58;
		uint r59 = r221 + 1;
		uint r222 = r57 * 128 + r220;
		for (uint r272 = r222; r272 < r59; r272++) {
			uint r223 = s_candidate[r272];
			uint r224 = s_candidate[r272 + 64];
			s_candidate[r272] = r224;
			s_candidate[r272 + 64] = r223;
		}
	}

	__syncthreads();

	uint r226 = r165 >> 26;
	uint r227 = tid + r226;
	uint r63 = r227 >> 6;
	uint r64 = r63 << 8;
	uint r228 = s_candidate[r64 + 128];
	uint r229 = s_candidate[r64];

	if (r229 > r228) {
		uint r233 = r227 & 2147483584;
		uint r234 = tid - r233;
		uint r235 = r234 << 1;
		uint r236 = r235 + r64;
		uint r65 = r236 + 1;
		uint r237 = r63 * 256 + r235;
		for (uint r273 = r237; r273 < r65; r273++) {
			uint r238 = s_candidate[r273];
			uint r239 = s_candidate[r273 + 128];
			s_candidate[r273] = r239;
			s_candidate[r273 + 128] = r238;
		}
	}

	__syncthreads();

	uint r241 = r165 >> 25;
	uint r242 = tid + r241;
	uint r69 = r242 >> 7;
	uint r70 = r69 << 9;
	uint r243 = s_candidate[r70 + 256];
	uint r244 = s_candidate[r70];

	if (r244 > r243) {
		uint r248 = r242 & 2147483520;
		uint r249 = tid - r248;
		uint r250 = r249 << 1;
		uint r251 = r250 + r70;
		uint r71 = r251 + 1;
		uint r252 = r69 * 512 + r250;

		for (uint r274 = r252; r274 < r71; r274++) {
			uint r253 = s_candidate[r274];
			uint r254 = s_candidate[r274 + 256];
			s_candidate[r274] = r254;
			s_candidate[r274 + 256] = r253;
		}
	}

	__syncthreads();

	if (tid == 0) 
	{
		s_sol_num = atomicAdd(&data->candidates.sol_nr[0], 1);
	}

	__syncthreads();

	if (laneid == 0)
	{

	}

	uint solc = s_sol_num;
	if (solc < 16)
	{
		uint4* p_cand = (uint4*) &data->candidates.vals[solc][tid_idx];
		uint4* p_s_cand = (uint4*)&s_candidate[tid_idx];
		*p_cand = *p_s_cand;
	}

}

mtm_cuda_context::mtm_cuda_context(int tpb, int blocks, int id)
: context(std::make_shared<mtm_context>(id))
{

}

void mtm_cuda_context::solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef)
{
	internalSolve(*context, tequihash_header, tequihash_header_len, nonce, nonce_len, cancelf, solutionf, hashdonef);
}

static constexpr auto COLLISION_BIT_LENGTH = (PARAM_N / (PARAM_K+1));
//static constexpr auto COLLISION_BYTE_LENGTH = ((COLLISION_BIT_LENGTH+7)/8);
//static constexpr auto FINAL_FULL_WIDTH = (2*COLLISION_BYTE_LENGTH+sizeof(uint32_t)*(1 << (PARAM_K)));

//static constexpr auto NDIGITS = (PARAM_K+1);
//static constexpr auto DIGITBITS = (PARAM_N/(NDIGITS));
static constexpr auto PROOFSIZE = (1u<<PARAM_K);
static constexpr auto COMPRESSED_PROOFSIZE = ((COLLISION_BIT_LENGTH+1)*PROOFSIZE*4/(8*sizeof(uint32_t)));


#include <mutex>
struct speed_test
{
	using time_point = std::chrono::high_resolution_clock::time_point;

	time_point m_start;
	int m_interval;

	speed_test(int interval)
		: m_start(std::chrono::high_resolution_clock::now())
		, m_interval(interval)
	{

	}

	std::vector<time_point> solutions;
	std::mutex sol_mutex;

	void AddSolution() {
		std::lock_guard<std::mutex> l(sol_mutex);
		solutions.push_back(std::chrono::high_resolution_clock::now());
	}

	double GetSolutionSpeed()
	{
		return Get(solutions, sol_mutex);
	}

	double Get(std::vector<time_point>& buffer, std::mutex& mutex)
	{
		time_point now = std::chrono::high_resolution_clock::now();
		time_point past = now - std::chrono::seconds(m_interval);
		double interval = (double)m_interval;
		if (past < m_start)
		{
			interval = ((double)std::chrono::duration_cast<std::chrono::milliseconds>(now - m_start).count()) / 1000;
			past = m_start;
		}
		size_t total = 0;

		mutex.lock();
		for (std::vector<time_point>::iterator it = buffer.begin(); it != buffer.end();)
		{
			if ((*it) < past)
			{
				it = buffer.erase(it);
			}
			else
			{
				++total;
				++it;
			}
		}
		mutex.unlock();

		return (double)total / (double)interval;
	}

};

speed_test speed2(300);
std::vector<uint> bin_counter(NR_ROWS * 512);
std::vector<uint> row_counter(NR_ROWS);



int bins[512] = { 0 };

template<int END = 256>
void PrintAverageBinCount(int round, mtm_context& ctx)
{
	std::fill(&bin_counter[0], &bin_counter[END], 0);
	checkCudaErrors(cudaMemcpy(&bin_counter[0], ctx.d_data->bin_counter, NR_ROWS * END * 4, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	std::fill(&bins[0], &bins[END], 0);

	for (int i = 0; i < NR_ROWS; i++) {
		for (int n = 0; n < END; n++) {
			bins[n] += bin_counter[i * END + n];
		}
	}

	int value = 0;

	for (int i = 0; i < END; i++) {
		value += bins[i];
	}

	int avg = value / END;

	//printf("Round %d: Avg Bin Count: %d\n", round, avg);
}

void PrintAverageRowCount(int round, mtm_context& ctx)
{
	std::fill(row_counter.begin(), row_counter.end(), 0);

	if (round % 2 == 0) {
		//row counter 1
		checkCudaErrors(cudaMemcpy(&row_counter[0], ctx.d_data->rowCounter1, NR_ROWS * 4, cudaMemcpyDeviceToHost));
	} else {
		//row couter 0
		checkCudaErrors(cudaMemcpy(&row_counter[0], ctx.d_data->rowCounter0, NR_ROWS * 4, cudaMemcpyDeviceToHost));
	}

	cudaDeviceSynchronize();

	int cnt = 0;
	for (int i = 0; i < NR_ROWS; i++) {
		cnt += row_counter[i];
	}

	int avg = cnt / NR_ROWS;

	printf("Round %d: Avg Row Count: %d\n", round - 1, avg);
}

/*
struct context_v1
{
	char* d_ht0;
	char* d_ht1;
	char* d_rowCounter0;
	char* d_rowCounter1;
	sols_t* d_sols;
	sols_t* h_sols;

	void init()
	{
		checkCudaErrors(cudaSetDevice(0));
		checkCudaErrors(cudaDeviceReset());

		checkCudaErrors(cudaMalloc((void**)&d_ht0, HT_SIZE));
		checkCudaErrors(cudaMalloc((void**)&d_ht1, HT_SIZE));

		checkCudaErrors(cudaMalloc((void**)&d_rowCounter0, NR_ROWS));
		checkCudaErrors(cudaMalloc((void**)&d_rowCounter1, NR_ROWS));

		checkCudaErrors(cudaMalloc((void**)&d_sols, sizeof(sols_t)));
		checkCudaErrors(cudaMallocHost((void**)&h_sols, sizeof(sols_t)));
	}

	void destroy()
	{
		checkCudaErrors(cudaFree(d_ht0));
		checkCudaErrors(cudaFree(d_ht1));
		checkCudaErrors(cudaFree(d_rowCounter0));
		checkCudaErrors(cudaFree(d_rowCounter1));
		checkCudaErrors(cudaFree(d_sols));
		checkCudaErrors(cudaFreeHost(h_sols));
	}
};

static void solve_v1(context_v1& ctx, const char* header, unsigned int header_len, const char* nonce, unsigned int nonce_len)
{
	unsigned char mcontext[140];
	memset(mcontext, 0, 140);
	memcpy(mcontext, header, header_len);
	memcpy(mcontext + header_len, nonce, nonce_len);

	blake2b_state_t initialCtx;
	zcash_blake2b_init(&initialCtx, ZCASH_HASH_LEN, PARAM_N, PARAM_K);
	zcash_blake2b_update(&initialCtx, (const uint8_t*)mcontext, 128, 0);

	checkCudaErrors(cudaMemcpyToSymbol(d_blake, &initialCtx, sizeof(blake2b_state_s), 0, cudaMemcpyHostToDevice));

	char* d_ht0 = ctx.d_ht0;
	char* d_ht1 = ctx.d_ht1;
	char* d_rowCounter0 = ctx.d_rowCounter0;
	char* d_rowCounter1 = ctx.d_rowCounter1;
	sols_t* d_sols = ctx.d_sols;
	sols_t* h_sols = ctx.h_sols;


	kernel_init_v1 << <NR_ROWS / ROWS_PER_UINT / 256, 256 >> > (d_rowCounter0);
	kernel_round0_v1 << < NR_INPUTS / 256, 256 >> > (d_ht0, d_rowCounter0);
}
*/

static void internalSolve(mtm_context& ctx, const char* header, unsigned int header_len, const char* nonce, unsigned int nonce_len,
	std::function<bool()> cancelf,
	std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
	std::function<void(void)> hashdonef)
{
	uint64_t blake_data[16];
	unsigned char mcontext[140];
	memset(mcontext, 0, 140);
	memcpy(mcontext, header, header_len);
	memcpy(mcontext + header_len, nonce, nonce_len);

	blake2b_state_t initialCtx;
	zcash_blake2b_init(&initialCtx, ZCASH_HASH_LEN, PARAM_N, PARAM_K);
	zcash_blake2b_update(&initialCtx, (const uint8_t*)mcontext, 128, 0);

	static uint64_t blake_iv[] =
	{
		0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
		0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
		0x510e527fade682d1, 0x9b05688c2b3e6c1f,
		0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
	};

	memcpy(&blake_data[0], initialCtx.h, 64);
	memcpy(&blake_data[8], blake_iv, 64);
	
	blake_data[12] ^= 144;
	blake_data[14] ^= ~0ULL;
	
	checkCudaErrors(cudaMemcpy(ctx.d_blake_data, blake_data, 128, cudaMemcpyHostToDevice));

	kernel_round0<<<NR_ROWS, 256>>>(ctx.d_data, ctx.d_blake_data);
	if (cancelf()) return;
	kernel_round1<<<NR_ROWS, NR_SLOTS >>>(ctx.d_data);
	if (cancelf()) return;
	kernel_round2<<<NR_ROWS, NR_SLOTS >>>(ctx.d_data);
	if (cancelf()) return;
	kernel_round3<<<NR_ROWS, NR_SLOTS>>>(ctx.d_data);
	if (cancelf()) return;
	kernel_round4<<<NR_ROWS, NR_SLOTS>>>(ctx.d_data);
	if (cancelf()) return;
	kernel_round5<<<NR_ROWS, NR_SLOTS >>>(ctx.d_data);
	if (cancelf()) return;
	kernel_round6<<<NR_ROWS, NR_SLOTS >>>(ctx.d_data);
	if (cancelf()) return;
	kernel_round7<<<NR_ROWS, NR_SLOTS >>>(ctx.d_data);
	if (cancelf()) return;
	kernel_round8<<<NR_ROWS, NR_SLOTS >>>(ctx.d_data);
	if (cancelf()) return;
	kernel_round9<<<NR_ROWS, NR_SLOTS >>>(ctx.d_data);
	if (cancelf()) return;
	kernel_candidates<<<512, 128>>>(ctx.d_data);
	if (cancelf()) return;

	checkCudaErrors(cudaMemcpy(ctx.h_candidates, &ctx.d_data->candidates, sizeof(candidate_t), cudaMemcpyDeviceToHost));
	
	ctx.h_candidates->sol_nr[0] = min(16, ctx.h_candidates->sol_nr[0]);

	uint8_t valid[16] = { 0 };
	for (unsigned sol_i = 0; sol_i < ctx.h_candidates->sol_nr[0]; sol_i++) {
		verify_sol(ctx.h_candidates, sol_i, valid);
	}

	int sols_found = 0;
	uint8_t proof[COMPRESSED_PROOFSIZE * 2];
	for (uint32_t i = 0; i < ctx.h_candidates->sol_nr[0]; i++) {
		if (valid[i]) {
			compress(proof, (uint32_t *)(ctx.h_candidates->vals[i]));
			solutionf(std::vector<uint32_t>(0), 1344, proof);
			sols_found++;
		}
	}
	hashdonef();
}

using stratum::primitives::CBlock;
using stratum::primitives::CEquihashInput;
using stratum::CDataStream;
using stratum::arith::uint256;

static std::vector<uint256*> benchmark_nonces;

static void generate_nounces(int hashes)
{
	std::srand(std::time(0));
	benchmark_nonces.push_back(new uint256());
	benchmark_nonces.back()->begin()[31] = 1;
	for (int i = 0; i < (hashes - 1); ++i)
	{
		benchmark_nonces.push_back(new uint256());
		for (unsigned int i = 0; i < 32; ++i)
			benchmark_nonces.back()->begin()[i] = std::rand() % 256;
	}
}

mtm_context g_ctx(0);

static bool benchmark_solve(mtm_context& ctx, const CBlock& block, const char* header, unsigned int header_len)
{
	if (benchmark_nonces.empty()) {
		return false;
	}
	
	uint256* nonce = benchmark_nonces.front();
	benchmark_nonces.erase(benchmark_nonces.begin());
	
	internalSolve(ctx, header, header_len, (const char*)nonce->begin(), nonce->size(), 
	[](){return false;}, 
	[](const std::vector<uint32_t>&, size_t, const unsigned char*) {
		speed2.AddSolution();
	},
	[](){

	});
	
	std::fill(&bins[0], &bins[256], 0);

	delete nonce;
	
	return true;
}

static int benchmark()
{
	try
	{
		CBlock pblock;
		CEquihashInput I{ pblock };
		CDataStream ss(stratum::SER_NETWORK, PROTOCOL_VERSION);
		ss << I;

		const char *tequihash_header = (char *)&ss[0];
		unsigned int tequihash_header_len = ss.size();

		while(benchmark_solve(g_ctx, pblock, tequihash_header, tequihash_header_len));
	}
	catch (const std::runtime_error &e)
	{
		exit(0);
		return 0;
	}
	
	return 0;
}


#include <thread>
#include <atomic>

#if 0
int main()
{
	

	g_ctx.init();

	//Step 1 - Generate nouces
	generate_nounces(3500);

	std::atomic<int> amdone(0);

	std::thread tellme([&amdone]() {
		for (; amdone.load() == 0;) {
			std::this_thread::sleep_for(std::chrono::seconds(2));
			std::cout << speed.GetSolutionSpeed() << " Sols/s" << std::endl;
		}
	});
	
	benchmark();
	
	printf("final %.2f sols/s\n", speed.GetSolutionSpeed());
	
	amdone.store(1);
	tellme.join();

	g_ctx.destroy();

	cudaDeviceReset();

	return 0;
}
#endif
