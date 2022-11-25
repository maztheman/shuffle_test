#pragma once
#if 0
#define BITS_PER_ROW 16

#define ROWS_PER_UINT 2
#define ROW_MASK 0xFFFF

#define NR_SLOTS 683 // Prime numbers are preferable.

#define SLOT_LEN 32

#define NR_ROWS_LOG 12
#define NR_ROWS (1 << NR_ROWS_LOG)
//#define HALF_SIZE (NR_ROWS / 2)

#define HT_SIZE	(NR_ROWS * NR_SLOTS * SLOT_LEN)

constexpr uint32_t c_HT_SIZE = HT_SIZE;

#define RC_SIZE (NR_ROWS * 4 / ROWS_PER_UINT)

#define COLL_DATA_SIZE_PER_TH		(NR_SLOTS * 5)

#define xi_offset_for_round(round)	(8 + ((round) / 2) * 4)

#define checkCudaErrors(call)								\
do {														\
	cudaError_t err = call;									\
	if (cudaSuccess != err) {								\
		char errorBuff[512];								\
        sprintf_s(errorBuff, sizeof(errorBuff) - 1,			\
			"CUDA error '%s' in func '%s' line %d",			\
			cudaGetErrorString(err), __FUNCTION__, __LINE__);	\
		printf("<error> %s\n", errorBuff); \
		exit(0); \
		}														\
} while (0)


__device__ blake2b_state_t d_blake;
__device__ uint32_t d_blake_test[32];

__device__ unsigned int rowCounter0[RC_SIZE / 4];
__device__ unsigned int rowCounter1[RC_SIZE / 4];

__device__ char d_ht0[HT_SIZE];
__device__ char d_ht1[HT_SIZE];
__device__ char d_ht2[HT_SIZE];
//__device__ char d_ht3[HT_SIZE];
//__device__ char d_ht4[HT_SIZE];
//__device__ char d_ht5[HT_SIZE];
//__device__ char d_ht6[HT_SIZE];
//__device__ char d_ht7[HT_SIZE];
//__device__ char d_ht8[HT_SIZE];

#define get_global_id() (blockIdx.x * blockDim.x + threadIdx.x)
#define get_local_id() (threadIdx.x)
#define get_local_size() (blockDim.x)
__device__
inline int lane_id() { return threadIdx.x % 32; }

#define declare_lane_id()\
unsigned int laneid;\
asm volatile("mov.u32 %0, %laneid;\n" : "=r"(laneid));

#define declare_warp_id()\
unsigned int warpid;\
asm volatile("mov.u32 %0, %warpid;\n" : "=r"(warpid));

#define declare_warp_size()\
unsigned int warp_size;\
asm volatile("mov.u32 %0, %nwarpid;\n" : "=r"(warp_size));

#define declare_cta_id()\
uint32_t ctaid;\
asm volatile("mov.u32 %0, %ctaid.x;\n" : "=r"(ctaid)); 

#define declare_cta_size()\
unsigned int cta_size; \
asm volatile("mov.u32 %0, %nctaid.x;\n" : "=r"(cta_size));

#define declare_grid_id()\
uint32_t gridid; \
asm volatile("mov.u32 %0, %gridid;\n" : "=r"(gridid));

/*
** Reset counters in hash table.
*/
__global__
void kernel_init_ht_0()
{
	if (get_global_id() < RC_SIZE / 4)
		rowCounter0[get_global_id()] = 0;
}

__global__
void kernel_init_ht_1()
{
	if (get_global_id() < RC_SIZE / 4)
		rowCounter1[get_global_id()] = 0;
}

__constant__ uint64_t blake_iv[] =
{
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
	0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f,
	0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};


#define PREFIX                 (PARAM_N / (PARAM_K + 1))
#define NR_INPUTS              (1 << PREFIX)
#define ZCASH_BLOCK_HEADER_LEN 140


#define rotate(a, bits) ((a) << (bits)) | ((a) >> (64 - (bits)))

#define mix(va, vb, vc, vd, x, y) \
    va = (va + vb + x); \
vd = rotate((vd ^ va), (uint64_t)64 - 32); \
vc = (vc + vd); \
vb = rotate((vb ^ vc), (uint64_t)64 - 24); \
va = (va + vb + y); \
vd = rotate((vd ^ va), (uint64_t)64 - 16); \
vc = (vc + vd); \
vb = rotate((vb ^ vc), (uint64_t)64 - 63);

__device__ __inline__ void sir_mix_a_lot(volatile uint64_t* v, uint64_t word1)
{
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
}

#define ADJUSTED_SLOT_LEN(round) (((round) <= 5) ? SLOT_LEN : SLOT_LEN - 16)

typedef uint32_t uint;


typedef union {

	struct __builtin_align__(16) {

		uint4 a;

		uint4 b;

	};

	struct __builtin_align__(16)

	{

		uint x0, x1, x2, x3, x4, x5, x6, x7;

	};

} uint8;

typedef union {
	struct {
		uint i;
		uint xi[6];
		uint padding;
	} slot;
	uint8 ui8;
	uint4 ui4[2];
	uint2 ui2[4];
	uint  ui[8];
} slot_t;


__device__ char *get_slot_ptr(char *ht, uint round, uint row, uint slot)
{
	return ht + (row * NR_SLOTS + slot) * ADJUSTED_SLOT_LEN(round);
}

__device__ char *get_xi_ptr(char *ht, uint round, uint row, uint slot)
{
	return get_slot_ptr(ht, round, row, slot) + xi_offset_for_round(round);
}

__device__ void get_row_counters_index(uint *rowIdx, uint *rowOffset, uint row)
{
	*rowIdx = row / ROWS_PER_UINT;
	*rowOffset = BITS_PER_ROW * (row % ROWS_PER_UINT);
}

__device__ uint get_row(uint round, uint xi0)
{
	uint           row;
#if NR_ROWS_LOG == 12
	if (!(round % 2))
		row = (xi0 & 0xfff);
	else
		row = ((xi0 & 0x0f0f00) >> 8) | ((xi0 & 0xf0000000) >> 24);
#elif NR_ROWS_LOG == 13
	if (!(round % 2))
		row = (xi0 & 0x1fff);
	else
		row = ((xi0 & 0x1f0f00) >> 8) | ((xi0 & 0xf0000000) >> 24);
#elif NR_ROWS_LOG == 14
	if (!(round % 2))
		row = (xi0 & 0x3fff);
	else
		row = ((xi0 & 0x3f0f00) >> 8) | ((xi0 & 0xf0000000) >> 24);
#elif NR_ROWS_LOG == 15
	if (!(round % 2))
		row = (xi0 & 0x7fff);
	else
		row = ((xi0 & 0x7f0f00) >> 8) | ((xi0 & 0xf0000000) >> 24);
#elif NR_ROWS_LOG == 16
	if (!(round % 2))
		row = (xi0 & 0xffff);
	else
		row = ((xi0 & 0xff0f00) >> 8) | ((xi0 & 0xf0000000) >> 24);
#else
#error "unsupported NR_ROWS_LOG"
#endif
	return row;
}

__device__ uint inc_row_counter(uint *rowCounters, uint row)
{
	uint rowIdx, rowOffset;
	get_row_counters_index(&rowIdx, &rowOffset, row);
	uint cnt = atomicAdd(rowCounters + rowIdx, 1 << rowOffset);
	cnt = (cnt >> rowOffset) & ROW_MASK;
	printf("%u\n", cnt);
	if (cnt >= NR_SLOTS) {
		// avoid overflows
		atomicSub(rowCounters + rowIdx, 1 << rowOffset);
	}
	return cnt;
}

__device__ uint ht_store(uint round, char *ht, uint i,
	uint xi0, uint xi1, uint xi2, uint xi3, uint xi4, uint xi5, uint xi6, uint *rowCounters)
{
	uint row = get_row(round, xi0);
	uint cnt = inc_row_counter(rowCounters, row);
	if (cnt >= NR_SLOTS)
		return 1;
	char *p = get_slot_ptr(ht, round, row, cnt);
	slot_t slot;
	slot.slot.i = i;
	slot.slot.xi[0] = ((xi1 << 24) | (xi0 >> 8));
	slot.slot.xi[1] = ((xi2 << 24) | (xi1 >> 8));
	slot.slot.xi[2] = ((xi3 << 24) | (xi2 >> 8));
	slot.slot.xi[3] = ((xi4 << 24) | (xi3 >> 8));
	slot.slot.xi[4] = ((xi5 << 24) | (xi4 >> 8));
	slot.slot.xi[5] = ((xi6 << 24) | (xi5 >> 8));
	if (round <= 5) {
		*(uint4 *)p = slot.ui4[0];
		*((uint4 *)p + sizeof(uint4)) = slot.ui4[1];
		//*(uint8 *)p = slot.ui8;
	}
	else {
		*(uint4 *)p = slot.ui4[0];
	}
	return 0;
}

__global__ void round0()
{
	uint64_t v[16];
	__shared__ uint64_t v_setup[16];

	if (threadIdx.x == 0) {
		v_setup[0] = d_blake.h[0];
		v_setup[1] = d_blake.h[0];
		v_setup[2] = d_blake.h[0];
		v_setup[3] = d_blake.h[0];
		v_setup[4] = d_blake.h[0];
		v_setup[5] = d_blake.h[0];
		v_setup[6] = d_blake.h[0];
		v_setup[7] = d_blake.h[0];
		v_setup[8] = blake_iv[0];
		v_setup[9] = blake_iv[1];
		v_setup[10] = blake_iv[2];
		v_setup[11] = blake_iv[3];
		v_setup[12] = blake_iv[4];
		v_setup[13] = blake_iv[5];
		v_setup[14] = blake_iv[6];
		v_setup[15] = blake_iv[7];
		v_setup[12] ^= ZCASH_BLOCK_HEADER_LEN + 4 /* length of "i" */;
		// last block
		v_setup[14] ^= (uint64_t)-1;
	}

	__syncthreads();

	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t inputs_per_thread = NR_ROWS / (gridDim.x * blockDim.x);
	uint32_t input = (tid * inputs_per_thread);
	uint32_t input_end = ((tid + 1) * inputs_per_thread);
	while (input < input_end)
	{
		uint64_t word1 = (uint64_t)input << 32;

		v[0] = v_setup[0];
		v[1] = v_setup[1];
		v[2] = v_setup[2];
		v[3] = v_setup[3];
		v[4] = v_setup[4];
		v[5] = v_setup[5];
		v[6] = v_setup[6];
		v[7] = v_setup[7];
		v[8] = v_setup[8];
		v[9] = v_setup[9];
		v[10] = v_setup[10];
		v[11] = v_setup[11];
		v[12] = v_setup[12];
		v[13] = v_setup[13];
		v[14] = v_setup[14];
		v[15] = v_setup[15];

		//sir_mix_a_lot(v, word1);

		uint64_t h[7];
		h[0] = v_setup[0] ^ v[0] ^ v[8];
		h[1] = v_setup[1] ^ v[1] ^ v[9];
		h[2] = v_setup[2] ^ v[2] ^ v[10];
		h[3] = v_setup[3] ^ v[3] ^ v[11];
		h[4] = v_setup[4] ^ v[4] ^ v[12];
		h[5] = v_setup[5] ^ v[5] ^ v[13];
		h[6] = (v_setup[6] ^ v[6] ^ v[14]) & 0xffff;

		printf("test\n");

		input++;
	}

}

__device__ __inline__ uint64_t shfl(uint64_t x, int lane)
{
	// Split the double number into 2 32b registers.
	int lo, hi;
	asm volatile("mov.b32 {%0, %1}, %2; ": "=r"(lo), "=r"(hi) : "l"(x));
	// Shuffle the two 32b registers.
	lo = __shfl(lo, lane);
	hi = __shfl(hi, lane);
	// Recreate the 64b number.
	asm volatile("mov.b64 %0, {%1, %2};": "=l"(x) : "r"(lo), "r"(hi));
	return x;
}

#endif
