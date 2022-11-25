#pragma once

#if 0
struct thread_data_t
{
	__device__ __forceinline__ void setup()
	{
		uint laneid = get_lane_id();
		bank = threadIdx.x - laneid;
		//first_row = blockIdx.x * blockDim.x + bank;
		//counter_idx = threadIdx.x / 32;
		//xi_offset = xi_offset_for_round(round);
	}

	__device__ __forceinline__ void setup_shared(uint16_t* thread_data, uint32_t* counters, coll_data_t* collisions)
	{
		//data = thread_data;
		//data_bank = &data[bank];
		warp_counter = &counters[threadIdx.x / 32];
		warp_collisions = &collisions[threadIdx.x / 32];

		if (get_lane_id() == 0) {
			*warp_counter = 0;
		}
	}

	__device__ __forceinline__ void lookup_counter(uint row, const uint* rowCountersSrc, uint* cnt)
	{
		uint rowIdx = row / ROWS_PER_UINT;
		uint rowOffset = BITS_PER_ROW * (row % ROWS_PER_UINT);
		*cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
		*cnt = min(*cnt, (uint)NR_SLOTS); // handle possible overflow in prev. round
	}

	/*__device__ __forceinline__ uchar get_other_key(int first_lane, int mask)
	{
	return data[bank + first_lane - 1] & mask;
	}*/

	__device__ __forceinline__ void process_collisions_12(int round, uint first_row, const char* ht_src, char* ht_dst, uint* rowCountersDst)
	{
		int counter = *warp_counter;

		for (int n = get_lane_id(); n < counter; n += 32) {
			uchar4* test = &warp_collisions->data[n];

			uint32_t row = test->x | first_row;

			uint2 c[3];
			ulong* xi = (ulong*)&c[0].x;

			const uint4* global_lws_i = (const uint4*)(ht_src + row * NR_SLOTS * SLOT_LEN + test->y * SLOT_LEN + xi_offset_for_round(round - 1));
			const uint4* global_lws_j = (const uint4*)(ht_src + row * NR_SLOTS * SLOT_LEN + test->z * SLOT_LEN + xi_offset_for_round(round - 1));


			if (round == 1 || round == 2) {
				uint2 a0, b0;
				uint4 a, b;

				// xor 24 bytes, 8 byte boundary
				asm volatile ("ld.global.v2.u32 {%0, %1}, [%2];\n" : "=r"(a0.x), "=r"(a0.y) : "l"(global_lws_i));
				asm volatile ("ld.global.v2.u32 {%0, %1}, [%2];\n" : "=r"(b0.x), "=r"(b0.y) : "l"(global_lws_j));
				asm volatile ("ld.global.v4.u32 {%0, %1, %2, %3}, [%4+8];\n" : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(global_lws_i));
				asm volatile ("ld.global.v4.u32 {%0, %1, %2, %3}, [%4+8];\n" : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(global_lws_j));

				c[0].x = a0.x ^ b0.x;
				c[0].y = a0.y ^ b0.y;
				c[1].x = a.x ^ b.x;
				c[1].y = a.y ^ b.y;
				c[2].x = a.z ^ b.z;
				c[2].y = a.w ^ b.w;

				if (round == 2) {
					// skip padding byte
					xi[0] = (xi[0] >> 8) | (xi[1] << (64 - 8));
					xi[1] = (xi[1] >> 8) | (xi[2] << (64 - 8));
					xi[2] = (xi[2] >> 8);
				}
			}
			else if (round == 3) {
				uint a0, b0;
				uint4 a, b;
				//20 bytes 4 bytes in fw then 16 bytes 
				//have to split it into 2 8 byte reads since its not on a 16 byte boundary, just 8
				asm volatile ("ld.global.u32 %0, [%1];\n" : "=r"(a0) : "l"(global_lws_i));
				asm volatile ("ld.global.v2.u32 {%0, %1}, [%4+4];\n"
					"ld.global.v2.u32 {%2, %3}, [%4+12];\n" : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(global_lws_i));

				asm volatile ("ld.global.u32 %0, [%1];\n" : "=r"(b0) : "l"(global_lws_j));
				asm volatile ("ld.global.v2.u32 {%0, %1}, [%4+4];\n"
					"ld.global.v2.u32 {%2, %3}, [%4+12];\n": "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(global_lws_j));

				c[0].x = a0 ^ b0;
				c[0].y = a.x ^ b.x;
				c[1].x = a.y ^ b.y;
				c[1].y = a.z ^ b.z;
				c[2].x = a.w ^ b.w;
				c[2].y = 0;
			}
			else if (round == 4) {
				uint a0, b0;
				uint2 a, b;
				uint a2, b2;

				//xor 16 bytes, 4 bytes loaded already so 12 left, aligned at 8 bytes
				//round 4 is slow ...

				asm volatile ("ld.global.u32 %0, [%1];\n" : "=r"(a0) : "l"(global_lws_i));
				asm volatile ("ld.global.v2.u32 {%0, %1}, [%3+4];\n\t"
					"ld.global.u32 %2, [%3+12];\n\t" : "=r"(a.x), "=r"(a.y), "=r"(a2) : "l"(global_lws_i));

				asm volatile ("ld.global.u32 %0, [%1];\n" : "=r"(b0) : "l"(global_lws_j));
				asm volatile ("ld.global.v2.u32 {%0, %1}, [%3+4];\n"
					"ld.global.u32 %2, [%3+12];\n" : "=r"(b.x), "=r"(b.y), "=r"(b2) : "l"(global_lws_j));

				c[0].x = a0 ^ b0;
				c[0].y = a.x ^ b.x;
				c[1].x = a.y ^ b.y;
				c[1].y = a2 ^ b2;
				c[2].x = 0;
				c[2].y = 0;

				// skip padding byte
				xi[0] = (xi[0] >> 8) | (xi[1] << (64 - 8));
				xi[1] = (xi[1] >> 8);

			}
			else if (round == 5) {
				uint4 a, b;
				//xor 16 bytes
				asm volatile ("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(global_lws_i));
				asm volatile ("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(global_lws_j));

				c[0].x = a.x ^ b.x;
				c[0].y = a.y ^ b.y;
				c[1].x = a.z ^ b.z;
				c[1].y = a.w ^ b.w;
				c[2].x = 0;
				c[2].y = 0;
			}
			else if (round == 6) {
				uint4 a, b;

				//xor 12 bytes, read 16 bytes ignoring last 4 bytes, its faster!
				asm volatile ("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(global_lws_i));
				asm volatile ("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(global_lws_j));

				c[0].x = a.x ^ b.x;
				c[0].y = a.y ^ b.y;
				c[1].x = a.z ^ b.z;
				c[1].y = 0;
				c[2].x = 0;
				c[2].y = 0;


				// skip padding byte
				xi[0] = (xi[0] >> 8) | (xi[1] << (64 - 8));
				xi[1] = (xi[1] >> 8);

			}
			else if (round == 7 || round == 8) {
				uint2 a, b;
				// xor 8 bytes, all bytes should be loaded possibly experiment with reading 16 bytes and throw away x, w

				asm volatile ("ld.global.u32 %0, [%2];\n\t"
					"ld.global.u32 %1, [%2+4];\n\t" : "=r"(a.x), "=r"(a.y) : "l"(global_lws_i));

				asm volatile ("ld.global.u32 %0, [%2];\n\t"
					"ld.global.u32 %1, [%2+4];\n\t" : "=r"(b.x), "=r"(b.y) : "l"(global_lws_j));

				c[0].x = a.x ^ b.x;
				c[0].y = a.y ^ b.y;
				c[1].x = 0;
				c[1].y = 0;
				c[2].x = 0;
				c[2].y = 0;

				if (round == 8) {
					// skip padding byte
					xi[0] = (xi[0] >> 8);
				}
			}

			if (!xi[0] && !xi[1])
				continue;

			ht_store_12(round, ht_dst, ENCODE_INPUTS(row, test->y, test->z), c, rowCountersDst);
			//ht_store(round, ht_dst, ENCODE_INPUTS(row, test->y, test->z), xi[0], xi[1], xi[2], 0, rowCountersDst);
		}

		if (get_lane_id() == 0) {
			*warp_counter = 0;
		}
	}

	//locals
	uint			bank;
	//shared
	//uint16_t*		data;
	//uint16_t*		data_bank;//subset of bank
	uint32_t*		warp_counter;
	coll_data_t*	warp_collisions;
};


__device__ __forceinline__ void lookup_counter(uint row, const uint* rowCountersSrc, uint* cnt)
{
	uint rowIdx = row / ROWS_PER_UINT;
	uint rowOffset = BITS_PER_ROW * (row % ROWS_PER_UINT);
	*cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
	*cnt = min(*cnt, (uint)NR_SLOTS); // handle possible overflow in prev. round
}

#define RND1_THD 128
#define RND1_BANK_COUNT (RND1_THD / 32)


typedef struct collision_table_s
{
	uint data[160];
} collision_table_t;

__global__
__launch_bounds__(512, 16)
void equihash_round_34(int round, const char* ht_src, char* ht_dst, const uint* rowCountersSrc, uint* rowCountersDst)
{
	__shared__ uint		cnt[256];//256 counters, each index is the value, count is the current index
	__shared__ uint4	slot[608];//first 16 bytes
	__shared__ uint2	slot_end[608];
	__shared__ uint16_t	collisions[3072];

	uint idx = blockIdx.x;
	uint tid = threadIdx.x;
	//this guy uses 1 << 12 rows, with 608 slots
	uint count = rowCountersSrc[idx];

	if (tid < 256) {
		cnt[tid] = 0;
	}

	__syncthreads();

	if (tid == 0) {
		//looks like set global counter to 0?? (instead of kernel_init! (dest counter)
		rowCountersDst[idx] = 0;
	}

	for (; tid < 608; tid += 512) {
		uint cntIdx = 0;
		uint4 data;
		uint2 data2;
		if (tid < count) {
			//im guessing data.x is masked off to 256 or something...
			//read 16 bytes
			data = *(const uint4*)(ht_src + NR_SLOTS * SLOT_LEN * idx + tid * SLOT_LEN);
			slot[tid] = data;
			data2 = *(const uint2*)(ht_src + NR_SLOTS * SLOT_LEN * idx + tid * SLOT_LEN + 16);
			slot_end[tid] = data2;
			uint cntIdx = atomicAdd(&cnt[data.x & 255], 1);
			cntIdx = min(cntIdx, 11);//something like only 12 collisions please, 0-11 index
									 //this is bascially binning
			collisions[(data.x & 255) * 12 + cntIdx] = tid;//nr_collision_slots is 12, so i can know how many slots are same value because they are being stored in this way
														   //this is where i can try to use thread teaming....!
														   //to avoid the atomic add.....!
														   //just need to figure out how the slot data is stored because it seems like first 4 bytes are and'd with 0xFF so traditionally the lowest byte is used where this is using whole 4 bytes ...HOW?
		}

		__syncthreads();//because the shared memory writes

		if (cntIdx > 0) {
			//12 is collision per element size
			uint16_t * addr = &collisions[(data.x & 255) * 12 + cntIdx];
			for (uint i = 0; i < cntIdx; i++, addr++) {
				uint16_t d = *addr;
				uint2 slot_low = slot_end[d];
				if (slot_low.y != data2.y) {
					uint4 c = slot[d];
					uint tmp = c.y ^ data.y;
					uint dstRow = tmp >> 20;
					uint rowCnt = atomicAdd(rowCountersDst + dstRow, 1);
					if (rowCnt < 608) {
						uint xi1 = slot_low.x ^ data2.x;
						uint xi2 = data.z ^ c.z;
						uint xi3 = data.w ^ c.w;
						uint xi4 = tmp & 0xFFFFF;
					}
				}
			}

		}
	}

done:
}

__device__ void equihash_round_12(int round, const char* ht_src, char* ht_dst, const uint* rowCountersSrc, uint* rowCountersDst)
{
	__shared__ coll_data_t	collisions[RND1_BANK_COUNT];//for 128 threads this is size of 4 * bla
														//__shared__ uint16_t		thread_data[RND1_THD];
	__shared__ uint32_t		counters[RND1_BANK_COUNT];


	int mask = ((!(round % 2)) ? 0x03 : 0x30);

	thread_data_t th_data;
	//locals
	th_data.setup();
	uint first_row = blockIdx.x * blockDim.x + th_data.bank;
	//shared
	th_data.setup_shared(0, counters, collisions);

	uint32_t row = first_row;//start at this row (chunk == 0 - 32, which means each thread is in its only 1k chunk at the beginning)

	const char *p_base = ht_src + row * NR_SLOTS * SLOT_LEN;

	for (int h = 0; h < 32; h++, row++, p_base += (NR_SLOTS * SLOT_LEN)) {
		uint cnt;
		th_data.lookup_counter(row, rowCountersSrc, &cnt);
		if (!cnt) {
			continue;
		}

		uint peers;
		{
			const uint16_t* global_fw = (const uint16_t*)(p_base + get_lane_id() * SLOT_LEN + xi_offset_for_round(round - 1));//last round
			uint16_t first_word;
			asm volatile ("ld.global.u16 %0, [%1];" : "=h"(first_word) : "l"(global_fw));
			first_word &= mask;
			//uchar first_word = th_data.data[threadIdx.x] & mask;
			//we must unassign all the lanes after cnt since its shit reads
			uint lane_mask = ((1 << cnt) - 1);
			uint unclaimed = lane_mask;//assuming laneid 0 is bit 0 ?
			bool is_peer;
			do {
				int first_lane = __ffs(unclaimed);
				uint16_t other_key = __shfl(first_word, first_lane - 1);//th_data.get_other_key(first_lane, mask);//   data[bank + first_lane - 1].x & 0x30;  //__shfl(first_word, first_lane - 1);
				is_peer = first_word == other_key;
				peers = __ballot(is_peer);
				unclaimed ^= (peers & lane_mask);//in cause peers match but are actually not counted
			} while (!is_peer && unclaimed);
			peers &= lane_mask;//only use the peers as high as count, yes there will be wasted threads of this isnt 0xFFFFFFFF, lets make sure its always pretty high...
							   //ok i have the collisions now what we need to do is only process the collisions
							   //find the first peer i collided with
							   //uint first = __ffs(peers) - 1;
			peers &= (0xFFFFFFFE << get_lane_id());
		}

		while (__any(peers)) {
			uint next = __ffs(peers);
			if (next) {
				uint32_t test = __ballot(1);
				int leader = __ffs(test) - 1;//-1 to get leader
				int idx = *th_data.warp_counter;
				if (leader == get_lane_id()) {
					*th_data.warp_counter += __popc(test);
				}
				idx += __popc(test << (32 - get_lane_id()));
				th_data.warp_collisions->data[idx] = make_uchar4(h, get_lane_id(), next - 1, 0);
				peers ^= (1 << (next - 1));
			}

			if (*th_data.warp_counter > 31) {
				th_data.process_collisions_12(round, first_row, ht_src, ht_dst, rowCountersDst);
			}

		}

	}

	//finish up any left over collisions	
	th_data.process_collisions_12(round, first_row, ht_src, ht_dst, rowCountersDst);
}



__global__
__launch_bounds__(RND1_THD, 16)
void kernel_round1(const char* ht_src, char* ht_dst, const char* rowCounterSrc, char* rowCounterDst)
{
	equihash_round_12(1, ht_src, ht_dst, (const uint*)rowCounterSrc, (uint*)rowCounterDst);
}

__global__
__launch_bounds__(RND1_THD, 16)
void kernel_round2(const char* ht_src, char* ht_dst, const char* rowCounterSrc, char* rowCounterDst)
{
	equihash_round_12(2, ht_src, ht_dst, (const uint*)rowCounterSrc, (uint*)rowCounterDst);
}

__global__
__launch_bounds__(RND1_THD, 16)
void kernel_round3(const char* ht_src, char* ht_dst, const char* rowCounterSrc, char* rowCounterDst)
{
	equihash_round_12(3, ht_src, ht_dst, (const uint*)rowCounterSrc, (uint*)rowCounterDst);
}

__global__
__launch_bounds__(RND1_THD, 16)
void kernel_round4(const char* ht_src, char* ht_dst, const char* rowCounterSrc, char* rowCounterDst)
{
	equihash_round_12(4, ht_src, ht_dst, (const uint*)rowCounterSrc, (uint*)rowCounterDst);
}

__global__
__launch_bounds__(RND1_THD, 16)
void kernel_round5(const char* ht_src, char* ht_dst, const char* rowCounterSrc, char* rowCounterDst)
{
	equihash_round_12(5, ht_src, ht_dst, (const uint*)rowCounterSrc, (uint*)rowCounterDst);
}

__global__
__launch_bounds__(RND1_THD, 16)
void kernel_round6(const char* ht_src, char* ht_dst, const char* rowCounterSrc, char* rowCounterDst)
{
	equihash_round_12(6, ht_src, ht_dst, (const uint*)rowCounterSrc, (uint*)rowCounterDst);
}

__global__
__launch_bounds__(RND1_THD, 16)
void kernel_round7(const char* ht_src, char* ht_dst, const char* rowCounterSrc, char* rowCounterDst)
{
	equihash_round_12(7, ht_src, ht_dst, (const uint*)rowCounterSrc, (uint*)rowCounterDst);
}

__global__
__launch_bounds__(RND1_THD, 16)
void kernel_round8(const char* ht_src, char* ht_dst, const char* rowCounterSrc, char* rowCounterDst, sols_t* sols)
{
	uint tid = get_global_id();
	equihash_round_12(8, ht_src, ht_dst, (const uint*)rowCounterSrc, (uint*)rowCounterDst);
	if (!tid) {
		sols->nr = sols->likely_invalids = 0;
	}
}



__device__
uint expand_ref(const char *ht, uint xi_offset, uint row, uint slot)
{
	return *(const uint *)(ht + row * NR_SLOTS * SLOT_LEN + slot * SLOT_LEN + xi_offset - 4);
}

__device__
uint expand_refs(uint *ins, uint nr_inputs, const char **htabs, uint round)
{
	const char *ht = htabs[round % 2];
	uint    i = nr_inputs - 1;
	uint    j = nr_inputs * 2 - 1;
	uint    xi_offset = xi_offset_for_round(round);
	int     dup_to_watch = -1;
	do
	{
		ins[j] = expand_ref(ht, xi_offset,
			DECODE_ROW(ins[i]), DECODE_SLOT1(ins[i]));
		ins[j - 1] = expand_ref(ht, xi_offset,
			DECODE_ROW(ins[i]), DECODE_SLOT0(ins[i]));
		if (!round)
		{
			if (dup_to_watch == -1)
				dup_to_watch = ins[j];
			else if (ins[j] == dup_to_watch || ins[j - 1] == dup_to_watch)
				return 0;
		}
		if (!i)
			break;
		i--;
		j -= 2;
	} while (1);
	return 1;
}

/*
** Verify if a potential solution is in fact valid.
*/
__device__
void potential_sol(const char **htabs, sols_t *sols, uint ref0, uint ref1)
{
	uint  nr_values;
	uint  values_tmp[(1 << PARAM_K)];
	uint  sol_i;
	uint  i;
	nr_values = 0;
	values_tmp[nr_values++] = ref0;
	values_tmp[nr_values++] = ref1;
	uint round = PARAM_K - 1;
	do
	{
		round--;
		if (!expand_refs(values_tmp, nr_values, htabs, round))
			return;
		nr_values *= 2;
	} while (round > 0);
	// solution appears valid, copy it to sols
	sol_i = atomicAdd(&sols->nr, 1);
	if (sol_i >= MAX_SOLS)
		return;
	for (i = 0; i < (1 << PARAM_K); i++)
		sols->values[sol_i][i] = values_tmp[i];
	sols->valid[sol_i] = 1;
}


struct sol_coll_t
{
	uint2 data[32];
};

#define SOL_THD 128
#define SOL_BANK_COUNT (SOL_THD / 32)


__inline__ __device__
uint warpAllReduceSum(uint val) {
	for (int mask = 16; mask > 0; mask >>= 1)
		val += __shfl_xor(val, mask);
	return val;
}

template<typename G>
__device__ __inline__ uint get_peers(G key, uint lane_mask) {
	uint peers = 0;
	bool is_peer;
	uint unclaimed = lane_mask;
	do {
		// fetch key of first unclaimed lane and compare with this key
		is_peer = (key == __shfl(key, __ffs(unclaimed) - 1));
		// determine which lanes had a match
		peers = __ballot(is_peer) & lane_mask;//ignore certain lanes because of "count"
											  // remove lanes with matching keys from the pool
		unclaimed ^= peers;
		// quit if we had a match
	} while (!is_peer);
	return peers;
}

// warp-aggregated atomic increment
__device__
int atomicAggInc(int *ctr) {
	int mask = __ballot(1);
	// select the leader
	int leader = __ffs(mask) – 1;
	// leader does the update
	int res;
	if (get_lane_id() == leader)
		res = atomicAdd(ctr, __popc(mask));
	// broadcast result
	res = __shfl(res, leader);
	// each thread computes its own value
	return res + __popc(mask & ((1 << get_lane_id()) – 1));
} // atomicAggInc

__global__
__launch_bounds__(SOL_THD, 16)
void kernel_sols_12(const char *ht0, const char *ht1, sols_t *sols, const uint *rowCountersSrc, const uint *rowCountersDst)
{
	__shared__ uint2	thread_data[SOL_THD];
	//__shared__ uint thread_data[RND1_THD];
	__shared__ sol_coll_t	collisions[SOL_BANK_COUNT];//for 128 threads this is size of 4 * bla
	__shared__ uint32_t	counters[SOL_BANK_COUNT];

	//thread_data_t th_data;
	//th_data.setup();

	//th_data.setup_shared(thread_data, counters, collisions);
	uint bank = threadIdx.x - get_lane_id();
	uint32_t* warp_counter = &counters[bank / 32];
	sol_coll_t* warp_collisions = &collisions[bank / 32];

	uint first_row = blockIdx.x * blockDim.x + bank;
	uint32_t row = first_row;//start at this row (chunk == 0 - 32, which means each thread is in its only 1k chunk at the beginning)

	if (get_lane_id() == 0)
		*warp_counter = 0;

	const char *htabs[2] = { ht0, ht1 };
	uint    cnt;
#if NR_ROWS_LOG >= 12 && NR_ROWS_LOG <= 20
	// in the final hash table, we are looking for a match on both the bits
	// part of the previous PREFIX colliding bits, and the last PREFIX bits.
	uint    mask = 0xffffff;
#else
#error "unsupported NR_ROWS_LOG"
#endif

	const char *p_base = ht0 + row * NR_SLOTS * SLOT_LEN;

	for (int h = 0; h < 32; h++, row++, p_base += (NR_SLOTS * SLOT_LEN)) {
		//each 32 threads process 32 rows
		uint rowIdx = row / ROWS_PER_UINT;
		uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
		cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
		cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in last round
		if (!cnt) {
			continue;
		}
		uint2 slot_data;

		const uint4* global_fw = (const uint4*)(p_base + get_lane_id() * SLOT_LEN + xi_offset_for_round(PARAM_K - 1) - 8);//last round
		asm volatile ("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(slot_data.x), "=r"(thread_data[threadIdx.x].x), "=r"(thread_data[threadIdx.x].y), "=r"(slot_data.y) : "l"(global_fw));
		//throw away .x, .z

		uint peers;
		{
			uint first_word = thread_data[threadIdx.x].y & mask;
			//we must unassign all the lanes after cnt since its shit reads
			uint lane_mask = ((1 << cnt) - 1);
			uint unclaimed = lane_mask;//assuming laneid 0 is bit 0 ?
			bool is_peer;
			do {
				int first_lane = __ffs(unclaimed);
				uint other_key = __shfl(first_word, first_lane - 1);//thread_data[bank + first_lane - 1].y & mask;// th_data.get_other_key(first_lane, mask);//   data[bank + first_lane - 1].x & 0x30;  //__shfl(first_word, first_lane - 1);
				is_peer = first_word == other_key;
				peers = __ballot(is_peer);
				unclaimed ^= (peers & lane_mask);//in cause peers match but are actually not counted
			} while (!is_peer && unclaimed);
			peers &= lane_mask;//only use the peers as high as count, yes there will be wasted threads of this isnt 0xFFFFFFFF, lets make sure its always pretty high...
							   //ok i have the collisions now what we need to do is only process the collisions
							   //find the first peer i collided with
							   //uint first = __ffs(peers) - 1;
			peers &= (0xFFFFFFFE << get_lane_id());
		}

		while (__any(peers)) {
			uint next = __ffs(peers);
			if (next) {
				int idx = atomicAggInc(warp_counter)
				warp_collisions->data[idx] = make_uint2(thread_data[bank + get_lane_id()].x, thread_data[bank + next - 1].x);
				break;//1 collision
			}
		}
	}

	int counter = *warp_counter;

	for (int n = get_lane_id(); n < counter; n += 32) {
		uint2* test = &warp_collisions->data[n];
		potential_sol(htabs, sols, test->y, test->x);
	}
}


//32 bytes
//


__device__ uint ht_store_12(uint round, char *ht, uint i, uint2* xi, uint *rowCounters)
{
	uint    row;
	char       *p;
	uint                cnt;

	if (!(round % 2))
		row = (xi[0].x & 0xffff) | ((xi[0].x & 0xc00000) >> 6);
	else
		row = ((xi[0].x & 0xc0000) >> 2) |
		((xi[0].x & 0xf00) << 4) | ((xi[0].x & 0xf00000) >> 12) |
		((xi[0].x & 0xf) << 4) | ((xi[0].x & 0xf000) >> 12);

	xi[0].x = __byte_perm(xi[0].x, xi[0].y, 0x5432);
	xi[0].y = __byte_perm(xi[0].y, xi[1].x, 0x5432);

	xi[1].x = __byte_perm(xi[1].x, xi[1].y, 0x5432);
	xi[1].y = __byte_perm(xi[1].y, xi[2].x, 0x5432);

	xi[2].x = __byte_perm(xi[2].x, xi[2].y, 0x5432);

	if (round == 0) {
		xi[2].y = __byte_perm(xi[2].y, xi[3].x, 0x5432);
	}
	else {
		xi[2].y = __byte_perm(xi[2].y, 0, 0x5432);
	}

	p = ht + row * NR_SLOTS * SLOT_LEN;

	uint rowIdx = row / ROWS_PER_UINT;
	uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
	uint xcnt = atomicAdd(rowCounters + rowIdx, 1u << rowOffset);
	xcnt = (xcnt >> rowOffset) & ROW_MASK;
	cnt = xcnt;
	if (cnt >= NR_SLOTS) {
		atomicSub(rowCounters + rowIdx, 1u << rowOffset);
		return 1;
	}
	p += cnt * SLOT_LEN + xi_offset_for_round(round);

	if (round == 0 || round == 1)
	{
		// store 24 bytes, offset 8
		asm volatile ("st.global.u32 [%0+-4], %1;\n" : : "l"(p), "r"(i));
		asm volatile ("st.global.v2.u32 [%0], {%1, %2};\n" :: "l"(p), "r"(xi[0].x), "r"(xi[0].y));
		asm volatile ("st.global.v4.u32 [%0+8], {%1, %2, %3, %4};\n" :: "l"(p), "r"(xi[1].x), "r"(xi[1].y), "r"(xi[2].x), "r"(xi[2].y));
	}
	else if (round == 2) {
		//store 20 bytes, offset 12
		//12-4=8, store 8 bytes i and xi[0].x, now offset is 16
		// store 20 bytes
		asm volatile ("st.global.v2.u32 [%0+-4], {%1, %2};\n" :: "l"(p), "r"(i), "r"(xi[0].x));
		asm volatile ("st.global.v4.u32 [%0+4], {%1, %2, %3, %4};\n" :: "l"(p), "r"(xi[0].y), "r"(xi[1].x), "r"(xi[1].y), "r"(xi[2].x));
	}
	else if (round == 3) {
		// store 16 bytes, offset 12
		//12-4=8, store 8 bytes now offset is 16, leaving 12 bytes to store
		asm volatile ("st.global.v2.u32 [%0+-4], {%1, %2};\n" :: "l"(p), "r"(i), "r"(xi[0].x));
		asm volatile ("st.global.v2.u32 [%0+4], {%1, %2};\n" :: "l"(p), "r"(xi[0].y), "r"(xi[1].x));
		asm volatile ("st.global.u32 [%0+12], %1;\n" :: "l"(p), "r"(xi[1].y));
	}
	else if (round == 4) {
		// store 16 bytes
		asm volatile ("st.global.u32 [%0+-4], %1;\n" : : "l"(p), "r"(i));
		asm volatile ("st.global.v4.u32 [%0], {%1, %2, %3, %4};\n" :: "l"(p), "r"(xi[0].x), "r"(xi[0].y), "r"(xi[1].x), "r"(xi[1].y));
	}
	else if (round == 5) {
		// store 12 bytes
		asm volatile ("st.global.u32 [%0+-4], %1;\n" : : "l"(p), "r"(i));
		asm volatile ("st.global.v2.u32 [%0], {%1, %2};\n" :: "l"(p), "r"(xi[0].x), "r"(xi[0].y));
		asm volatile ("st.global.u32 [%0+8], %1;\n" : : "l"(p), "r"(xi[1].x));
	}
	else if (round == 6 || round == 7) {
		// store 8 bytes
		asm volatile ("st.global.v2.u32 [%0+-4], {%1, %2};\n" :: "l"(p), "r"(i), "r"(xi[0].x));
		asm volatile ("st.global.u32 [%0+4], %1;\n" : : "l"(p), "r"(xi[0].y));
	}
	else if (round == 8) {
		asm volatile ("st.global.u32 [%0+-4], %1;\n" : : "l"(p), "r"(i));
		asm volatile ("st.global.u32 [%0], %1;\n" : : "l"(p), "r"(xi[0].x));
	}

	return 0;
}

#if 0
__device__ uint ht_store(uint round, char *ht, uint i, ulong xi0, ulong xi1, ulong xi2, ulong xi3, uint *rowCounters)
{
	uint    row;
	char       *p;
	uint                cnt;
#if NR_ROWS_LOG == 16
	if (!(round % 2))
		row = (xi0 & 0xffff);
	else
		// if we have in hex: "ab cd ef..." (little endian xi0) then this
		// formula computes the row as 0xdebc. it skips the 'a' nibble as it
		// is part of the PREFIX. The Xi will be stored starting with "ef...";
		// 'e' will be considered padding and 'f' is part of the current PREFIX
		row = ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 18
	if (!(round % 2))
		row = (xi0 & 0xffff) | ((xi0 & 0xc00000) >> 6);
	else
		row = ((xi0 & 0xc0000) >> 2) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 19
	if (!(round % 2))
		row = (xi0 & 0xffff) | ((xi0 & 0xe00000) >> 5);
	else
		row = ((xi0 & 0xe0000) >> 1) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 20
	if (!(round % 2))
		row = (xi0 & 0xffff) | ((xi0 & 0xf00000) >> 4);
	else
		row = ((xi0 & 0xf0000) >> 0) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#else
#error "unsupported NR_ROWS_LOG"
#endif
	xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
	xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
	xi2 = (xi2 >> 16) | (xi3 << (64 - 16));
	p = ht + row * NR_SLOTS * SLOT_LEN;

	uint rowIdx = row / ROWS_PER_UINT;
	uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
	uint xcnt = atomicAdd(rowCounters + rowIdx, 1u << rowOffset);
	xcnt = (xcnt >> rowOffset) & ROW_MASK;
	cnt = xcnt;
	if (cnt >= NR_SLOTS) {
		//atomicSub(rowCounters + rowIdx, 1u << rowOffset);
		return 1;
	}
	p += cnt * SLOT_LEN + xi_offset_for_round(round);
	// store "i" (always 4 bytes before Xi)
	*(uint *)(p - 4) = i;
	if (round == 0 || round == 1)
	{
		// store 24 bytes
		*(ulong *)(p + 0) = xi0;
		*(ulong *)(p + 8) = xi1;
		*(ulong *)(p + 16) = xi2;
	}
	else if (round == 2)
	{
		// store 20 bytes
		*(uint *)(p + 0) = xi0;
		*(ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
		*(ulong *)(p + 12) = (xi1 >> 32) | (xi2 << 32);
	}
	else if (round == 3)
	{
		// store 16 bytes
		*(uint *)(p + 0) = xi0;
		*(ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
		*(uint *)(p + 12) = (xi1 >> 32);
	}
	else if (round == 4)
	{
		// store 16 bytes
		*(ulong *)(p + 0) = xi0;
		*(ulong *)(p + 8) = xi1;
	}
	else if (round == 5)
	{
		// store 12 bytes
		*(ulong *)(p + 0) = xi0;
		*(uint *)(p + 8) = xi1;
	}
	else if (round == 6 || round == 7)
	{
		// store 8 bytes
		*(uint *)(p + 0) = xi0;
		*(uint *)(p + 4) = (xi0 >> 32);
	}
	else if (round == 8)
	{
		// store 4 bytes
		*(uint *)(p + 0) = xi0;
	}
	return 0;
}

#endif


#define get_global_id() (blockIdx.x * blockDim.x + threadIdx.x)
#define get_global_size() (gridDim.x * blockDim.x)

#define xi_offset_for_round(round)	(8 + ((round) / 2) * 4)


#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 20) | ((slot1 & 0x3ff) << 10) | (slot0 & 0x3ff))
#define DECODE_ROW(REF)   (REF >> 20)
#define DECODE_SLOT1(REF) ((REF >> 10) & 0x3ff)
#define DECODE_SLOT0(REF) (REF & 0x3ff)


#define declare_lane_id()\
unsigned int laneid;\
asm volatile("mov.u32 %0, %laneid;\n" : "=r"(laneid));

#define declare_warp_id()\
unsigned int warpid;\
asm volatile("mov.u32 %0, %warpid;\n" : "=r"(warpid));

#define declare_warp_size()\
unsigned int warp_size;\
asm volatile("mov.u32 %0, %nwarpid;\n" : "=r"(warp_size));

__device__ blake2b_state_t d_blake;
__device__ char d_processedRows[NR_ROWS];

__constant__ uint64_t blake_iv[] =
{
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
	0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f,
	0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};


__global__ void kernel_init(char* rowCounter)
{
	((uint*)rowCounter)[get_global_id()] = 0;
}

__global__
__launch_bounds__(256, 16)
void kernel_round0(char* d_ht0, char* rowCounter0)
{
	ulong v[16];

	asm("{\n\t"
		".reg .v4 .u32 v1,v2, v3, v4;\n\t"
		"ld.global.v4.u32 v1, [%8];\n\t"
		"ld.global.v4.u32 v2, [%8+16];\n\t"
		"ld.global.v4.u32 v3, [%8+32];\n\t"
		"ld.global.v4.u32 v4, [%8+48];\n\t"
		"mov.b64 %0, {v1.x, v1.y};\n\t"
		"mov.b64 %1, {v1.z, v1.w};\n\t"
		"mov.b64 %2, {v2.x, v2.y};\n\t"
		"mov.b64 %3, {v2.z, v2.w};\n\t"
		"mov.b64 %4, {v3.x, v3.y};\n\t"
		"mov.b64 %5, {v3.z, v3.w};\n\t"
		"mov.b64 %6, {v4.x, v4.y};\n\t"
		"mov.b64 %7, {v4.z, v4.w};\n\t"
		"}\n"
		:
	"=l"(v[0]), "=l"(v[1]), "=l"(v[2]), "=l"(v[3]),
		"=l"(v[4]), "=l"(v[5]), "=l"(v[6]), "=l"(v[7])
		: "l"(d_blake.h)
		);

	asm(
		"{\n\t"
		".reg .v4 .u32 v1,v2, v3, v4;\n\t"
		"ld.const.v4.u32 v1, [%8];\n\t"
		"ld.const.v4.u32 v2, [%8+16];\n\t"
		"ld.const.v4.u32 v3, [%8+32];\n\t"
		"ld.const.v4.u32 v4, [%8+48];\n\t"
		"mov.b64 %0, {v1.x, v1.y};\n\t"
		"mov.b64 %1, {v1.z, v1.w};\n\t"
		"mov.b64 %2, {v2.x, v2.y};\n\t"
		"mov.b64 %3, {v2.z, v2.w};\n\t"
		"mov.b64 %4, {v3.x, v3.y};\n\t"
		"mov.b64 %5, {v3.z, v3.w};\n\t"
		"mov.b64 %6, {v4.x, v4.y};\n\t"
		"mov.b64 %7, {v4.z, v4.w};\n\t"
		"}\n"
		:
	"=l"(v[8]), "=l"(v[9]), "=l"(v[10]), "=l"(v[11]),
		"=l"(v[12]), "=l"(v[13]), "=l"(v[14]), "=l"(v[15])
		: "l"(blake_iv)
		);

	v[12] ^= ZCASH_BLOCK_HEADER_LEN + 4 /* length of "i" */;
	// last block
	v[14] ^= ~0ULL;

	uint input = blockIdx.x * blockDim.x + threadIdx.x;

	ulong vt[7];
	vt[0] = v[0];
	vt[1] = v[1];
	vt[2] = v[2];
	vt[3] = v[3];
	vt[4] = v[4];
	vt[5] = v[5];
	vt[6] = v[6];

	// shift "i" to occupy the high 32 bits of the second ulong word in the
	// message block
	ulong word1 = (ulong)input << 32;

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
	v[0] = vt[0] ^ v[0] ^ v[8];
	v[1] = vt[1] ^ v[1] ^ v[9];
	v[2] = vt[2] ^ v[2] ^ v[10];
	v[3] = vt[3] ^ v[3] ^ v[11];
	v[4] = vt[4] ^ v[4] ^ v[12];
	v[5] = vt[5] ^ v[5] ^ v[13];
	v[6] = (vt[6] ^ v[6] ^ v[14]) & 0xffff;
	// store the two Xi values in the hash table
	uint2* xi = (uint2*)&v[0];

	ht_store_12(0, d_ht0, input * 2, (uint2*)&v[0], (uint*)rowCounter0);

	xi[3].x = __byte_perm(xi[3].x, xi[3].y, 0x4321);
	xi[3].y = __byte_perm(xi[3].y, xi[4].x, 0x4321);

	xi[4].x = __byte_perm(xi[4].x, xi[4].y, 0x4321);
	xi[4].y = __byte_perm(xi[4].y, xi[5].x, 0x4321);

	xi[5].x = __byte_perm(xi[5].x, xi[5].y, 0x4321);
	xi[5].y = __byte_perm(xi[5].y, xi[6].x, 0x4321);

	xi[6].x = __byte_perm(xi[6].x, xi[6].y, 0x4321);
	xi[6].y = __byte_perm(xi[6].y, 0, 0x4321);

	ht_store_12(0, d_ht0, input * 2, (uint2*)&v[3], (uint*)rowCounter0);
}

__device__ __forceinline__ uint get_lane_id() {
	uint retval;
	asm volatile("mov.u32 %0, %laneid;\n" : "=r"(retval));
	return retval;
}

struct coll_data_t
{
	uchar4 data[96];
};


__device__ uint xor_and_store(uint round, uint row, uint slot_a, uint slot_b, const char* ht_src, char* ht_dst, uint* rowCountersDst)
{
	uint2 c[3];
	ulong* xi = (ulong*)&c[0].x;

	const uint4* global_lws_i = (const uint4*)(ht_src + row * NR_SLOTS * SLOT_LEN + slot_a * SLOT_LEN + xi_offset_for_round(round - 1));
	const uint4* global_lws_j = (const uint4*)(ht_src + row * NR_SLOTS * SLOT_LEN + slot_b * SLOT_LEN + xi_offset_for_round(round - 1));


	if (round == 1 || round == 2) {
		uint2 a0, b0;
		uint4 a, b;

		// xor 24 bytes, 8 byte boundary
		asm volatile ("ld.global.v2.u32 {%0, %1}, [%2];\n" : "=r"(a0.x), "=r"(a0.y) : "l"(global_lws_i));
		asm volatile ("ld.global.v2.u32 {%0, %1}, [%2];\n" : "=r"(b0.x), "=r"(b0.y) : "l"(global_lws_j));
		asm volatile ("ld.global.v4.u32 {%0, %1, %2, %3}, [%4+8];\n" : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(global_lws_i));
		asm volatile ("ld.global.v4.u32 {%0, %1, %2, %3}, [%4+8];\n" : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(global_lws_j));

		c[0].x = a0.x ^ b0.x;
		c[0].y = a0.y ^ b0.y;
		c[1].x = a.x ^ b.x;
		c[1].y = a.y ^ b.y;
		c[2].x = a.z ^ b.z;
		c[2].y = a.w ^ b.w;

		if (round == 2) {
			// skip padding byte
			xi[0] = (xi[0] >> 8) | (xi[1] << (64 - 8));
			xi[1] = (xi[1] >> 8) | (xi[2] << (64 - 8));
			xi[2] = (xi[2] >> 8);
		}
	}
	else if (round == 3) {
		uint a0, b0;
		uint4 a, b;
		//20 bytes 4 bytes in fw then 16 bytes 
		//have to split it into 2 8 byte reads since its not on a 16 byte boundary, just 8
		asm volatile ("ld.global.u32 %0, [%1];\n" : "=r"(a0) : "l"(global_lws_i));
		asm volatile ("ld.global.v2.u32 {%0, %1}, [%4+4];\n"
			"ld.global.v2.u32 {%2, %3}, [%4+12];\n" : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(global_lws_i));

		asm volatile ("ld.global.u32 %0, [%1];\n" : "=r"(b0) : "l"(global_lws_j));
		asm volatile ("ld.global.v2.u32 {%0, %1}, [%4+4];\n"
			"ld.global.v2.u32 {%2, %3}, [%4+12];\n": "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(global_lws_j));

		c[0].x = a0 ^ b0;
		c[0].y = a.x ^ b.x;
		c[1].x = a.y ^ b.y;
		c[1].y = a.z ^ b.z;
		c[2].x = a.w ^ b.w;
		c[2].y = 0;
	}
	else if (round == 4) {
		uint a0, b0;
		uint2 a, b;
		uint a2, b2;

		//xor 16 bytes, 4 bytes loaded already so 12 left, aligned at 8 bytes
		//round 4 is slow ...

		asm volatile ("ld.global.u32 %0, [%1];\n" : "=r"(a0) : "l"(global_lws_i));
		asm volatile ("ld.global.v2.u32 {%0, %1}, [%3+4];\n\t"
			"ld.global.u32 %2, [%3+12];\n\t" : "=r"(a.x), "=r"(a.y), "=r"(a2) : "l"(global_lws_i));

		asm volatile ("ld.global.u32 %0, [%1];\n" : "=r"(b0) : "l"(global_lws_j));
		asm volatile ("ld.global.v2.u32 {%0, %1}, [%3+4];\n"
			"ld.global.u32 %2, [%3+12];\n" : "=r"(b.x), "=r"(b.y), "=r"(b2) : "l"(global_lws_j));

		c[0].x = a0 ^ b0;
		c[0].y = a.x ^ b.x;
		c[1].x = a.y ^ b.y;
		c[1].y = a2 ^ b2;
		c[2].x = 0;
		c[2].y = 0;

		// skip padding byte
		xi[0] = (xi[0] >> 8) | (xi[1] << (64 - 8));
		xi[1] = (xi[1] >> 8);

	}
	else if (round == 5) {
		uint4 a, b;
		//xor 16 bytes
		asm volatile ("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(global_lws_i));
		asm volatile ("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(global_lws_j));

		c[0].x = a.x ^ b.x;
		c[0].y = a.y ^ b.y;
		c[1].x = a.z ^ b.z;
		c[1].y = a.w ^ b.w;
		c[2].x = 0;
		c[2].y = 0;
	}
	else if (round == 6) {
		uint4 a, b;

		//xor 12 bytes, read 16 bytes ignoring last 4 bytes, its faster!
		asm volatile ("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(global_lws_i));
		asm volatile ("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(global_lws_j));

		c[0].x = a.x ^ b.x;
		c[0].y = a.y ^ b.y;
		c[1].x = a.z ^ b.z;
		c[1].y = 0;
		c[2].x = 0;
		c[2].y = 0;


		// skip padding byte
		xi[0] = (xi[0] >> 8) | (xi[1] << (64 - 8));
		xi[1] = (xi[1] >> 8);

	}
	else if (round == 7 || round == 8) {
		uint2 a, b;
		// xor 8 bytes, all bytes should be loaded possibly experiment with reading 16 bytes and throw away x, w

		asm volatile ("ld.global.u32 %0, [%2];\n\t"
			"ld.global.u32 %1, [%2+4];\n\t" : "=r"(a.x), "=r"(a.y) : "l"(global_lws_i));

		asm volatile ("ld.global.u32 %0, [%2];\n\t"
			"ld.global.u32 %1, [%2+4];\n\t" : "=r"(b.x), "=r"(b.y) : "l"(global_lws_j));

		c[0].x = a.x ^ b.x;
		c[0].y = a.y ^ b.y;
		c[1].x = 0;
		c[1].y = 0;
		c[2].x = 0;
		c[2].y = 0;

		if (round == 8) {
			// skip padding byte
			xi[0] = (xi[0] >> 8);
		}
	}

	if (!xi[0] && !xi[1])
		return;

	ht_store_12(round, ht_dst, ENCODE_INPUTS(row, slot_a, slot_b), c, rowCountersDst);
}

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


#if 0	

inline void solve_v1(context_v1& ctx, const char* header, unsigned int header_len, const char* nonce, unsigned int nonce_len)
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


	kernel_init_v1 << <NR_ROWS / ROWS_PER_UINT / 256, 256 >> >(d_rowCounter0);
	kernel_round0_v1 << < NR_INPUTS / 256, 256 >> >(d_ht0, d_rowCounter0);

	kernel_init << <NR_ROWS / ROWS_PER_UINT / 256, 256 >> >(d_rowCounter1);
	kernel_round1 << <NR_ROWS / RND1_THD, RND1_THD >> >(d_ht0, d_ht1, d_rowCounter0, d_rowCounter1);
	kernel_init << <NR_ROWS / ROWS_PER_UINT / 256, 256 >> >(d_rowCounter0);
	kernel_round2 << <NR_ROWS / RND1_THD, RND1_THD >> >(d_ht1, d_ht0, d_rowCounter1, d_rowCounter0);
	kernel_init << <NR_ROWS / ROWS_PER_UINT / 256, 256 >> >(d_rowCounter1);
	kernel_round3 << <NR_ROWS / RND1_THD, RND1_THD >> >(d_ht0, d_ht1, d_rowCounter0, d_rowCounter1);
	kernel_init << <NR_ROWS / ROWS_PER_UINT / 256, 256 >> >(d_rowCounter0);
	kernel_round4 << <NR_ROWS / RND1_THD, RND1_THD >> >(d_ht1, d_ht0, d_rowCounter1, d_rowCounter0);
	kernel_init << <NR_ROWS / ROWS_PER_UINT / 256, 256 >> >(d_rowCounter1);
	kernel_round5 << <NR_ROWS / RND1_THD, RND1_THD >> >(d_ht0, d_ht1, d_rowCounter0, d_rowCounter1);
	kernel_init << <NR_ROWS / ROWS_PER_UINT / 256, 256 >> >(d_rowCounter0);
	kernel_round6 << <NR_ROWS / RND1_THD, RND1_THD >> >(d_ht1, d_ht0, d_rowCounter1, d_rowCounter0);
	kernel_init << <NR_ROWS / ROWS_PER_UINT / 256, 256 >> >(d_rowCounter1);
	kernel_round7 << <NR_ROWS / RND1_THD, RND1_THD >> >(d_ht0, d_ht1, d_rowCounter0, d_rowCounter1);
	kernel_init << <NR_ROWS / ROWS_PER_UINT / 256, 256 >> >(d_rowCounter0);
	kernel_round8 << <NR_ROWS / RND1_THD, RND1_THD >> >(d_ht1, d_ht0, d_rowCounter1, d_rowCounter0, d_sols);
	kernel_sols_12 << <NR_ROWS / SOL_THD, SOL_THD >> >(d_ht0, d_ht1, d_sols, (uint*)d_rowCounter0, (uint*)d_rowCounter1);

	checkCudaErrors(cudaMemcpy(h_sols, d_sols, sizeof(sols_t), cudaMemcpyDeviceToHost));

	if (h_sols->nr > MAX_SOLS)
		h_sols->nr = MAX_SOLS;

	for (unsigned sol_i = 0; sol_i < h_sols->nr; sol_i++) {
		verify_sol(h_sols, sol_i);
	}

	int sols_found = 0;
	uint8_t proof[COMPRESSED_PROOFSIZE * 2];
	for (uint32_t i = 0; i < h_sols->nr; i++) {
		if (h_sols->valid[i]) {
			compress(proof, (uint32_t *)(h_sols->values[i]), 1 << PARAM_K);
			speed.AddSolution();
			sols_found++;
		}
	}

	//if (sols_found)
	//printf("%d sols found.\n", sols_found);
	return;
}
#endif



#endif
