#pragma once
/*
__global__
__launch_bounds__(608, 16)
void kernel_round1(data_t* data)
{
__shared__ uint		s_bincount[256];//256 counters, each index is the value, count is the current index
__shared__ uint4	s_slot1[608];//first 16 bytes
__shared__ uint2	s_slot2[608];//last 8 bytes, 24 total bytes
__shared__ uint16_t	s_collisions[3072];

uint idx = blockIdx.x;
uint tid = threadIdx.x;
//this guy uses 1 << 12 rows, with 608 slots
uint count = data->rowCounter0[idx];

if (tid < 256) {
s_bincount[tid] = 0;//reset bin count
}

if (tid == 0) {
data->rowCounter0[idx] = 0;//reset counter after reading
}

__syncthreads();

for (; tid < 608; tid += blockDim.x) {
uint bin_idx = ~0UL;
uint4 slot1;
uint2 slot2;
uint bin = 0;
uint bin_total;
if (tid < count) {
//im guessing data.x is masked off to 256 or something...
//read 16 bytes
slot1 = data->round0.rows[idx].slots[tid].x;
s_slot1[tid] = slot1;
slot2 = *(uint2*)(&data->round0.rows[idx].slots[tid].y);
s_slot2[tid] = slot2;
bin = slot1.x >> 20;
if (s_bincount[bin] < 12) {
bin_idx = atomicAdd(&s_bincount[bin], 1);
bin_idx = min(bin_idx, 11);//something like only 12 collisions please, 0-11 index
s_collisions[bin * 12 + bin_idx] = tid;//nr_collision_slots is 12, so i can know how many slots are same value because they are being stored in this way
}
}

//at this point the binning is done. Now all elagible threads will loop and produce the collisions
__syncthreads();//because the shared memory writes

if (bin_idx != ~0UL) {
bin_total = min(11, s_bincount[bin]);
//12 is collision per element size
uint16_t * addr = &s_collisions[bin * 12];
for (uint i = 0; i < bin_total; i++, addr++) {
uint16_t j = *addr;
uint2 other_slot2;
other_slot2 = s_slot2[j];
if (other_slot2.y != slot2.y) {//is it me?
uint4 other_slot1;
other_slot1 = s_slot1[j];

uint dstRow;
uint tmp;

asm volatile(
"{\n\t"
".reg .u32 tt;\n\t"
"xor.b32 %0, %2, %3;\t\n"
"prmt.b32 tt, %0, 0, 17185;\n\t"
"and.b32 %1, tt, 4095;\t\n"
"}\n"
: "=r"(tmp), "=r"(dstRow) : "r"(other_slot1.x), "r"(slot1.x));

uint rowCnt = atomicAdd(data->rowCounter1 + dstRow, 1);
if (rowCnt < 608) {
slot32_t to_slot;
to_slot.x.x = tmp & 255;
to_slot.x.y = other_slot1.y ^ slot1.y;
to_slot.x.z = other_slot1.z ^ slot1.z;
to_slot.x.w = other_slot1.w ^ slot1.w;
data->round1.rows[dstRow].slots[rowCnt].x = to_slot.x;
to_slot.y.x = other_slot2.x ^ slot2.x;
to_slot.y.y = other_slot2.y ^ slot2.y;
//ENCODE_INPUTS(idx, j, tid);//      (((idx << 10) | d) << 10) | tid;//this is basically ENCODE_ROW
to_slot.y.z = ((j | (idx << 10)) << 10) | tid;
to_slot.y.w = 0;
data->round1.rows[dstRow].slots[rowCnt].y = to_slot.y;
}
}
}

}
}
}


__global__
__launch_bounds__(608, 16)
void kernel_round2(data_t* data)
{
__shared__ uint		s_bincount[256];//256 counters, each index is the value, count is the current index
__shared__ uint4	s_slot1[608];//first 16 bytes
__shared__ uint2	s_slot2[608];//last 8 bytes, 24 total bytes
__shared__ uint16_t	s_collisions[3072];

uint idx = blockIdx.x;
uint tid = threadIdx.x;
uint count = data->rowCounter1[idx];

if (tid < 256) {
s_bincount[tid] = 0;//reset bin count
}

if (tid == 0) {
data->rowCounter1[idx] = 0;//reset counter after we read it
}

__syncthreads();

count = min(count, 608);

for (; tid < 608; tid += blockDim.x) {
uint bin_idx = ~0UL;
uint4 slot1;
uint2 slot2;
uint bin = 0;
if (tid < count) {
//im guessing data.x is masked off to 256 or something...
//read 16 bytes
slot1 = data->round1.rows[idx].slots[tid].x;
s_slot1[tid] = slot1;
slot2 = *(uint2*)(&data->round1.rows[idx].slots[tid].y);
s_slot2[tid] = slot2;
bin = slot1.x;
if (s_bincount[bin] < 12) {
bin_idx = atomicAdd(&s_bincount[bin], 1);
bin_idx = min(bin_idx, 11);//something like only 12 collisions please, 0-11 index
s_collisions[bin * 12 + bin_idx] = tid;//nr_collision_slots is 12, so i can know how many slots are same value because they are being stored in this way
}
}

__syncthreads();//because the shared memory writes

if (bin_idx != ~0UL) {
uint bin_total = min(11, s_bincount[bin]);
//12 is collision per element size
uint16_t * addr = &s_collisions[bin * 12];
for (uint i = 0; i < bin_total; i++, addr++) {
uint16_t j = *addr;
uint2 other_slot2;
other_slot2 = s_slot2[j];

if (other_slot2.y != slot2.y) {
uint4 other_slot1;
other_slot1 = s_slot1[j];

uint dstRow;
uint tmp;

asm volatile(
"{\n\t"
".reg .u32 tt;\n\t"
"xor.b32 %0, %2, %3;\t\n"
"shr.u32 %1, %0, 20;\n\t"
"}\n"
: "=r"(tmp), "=r"(dstRow) : "r"(other_slot1.y), "r"(slot1.y));

uint rowCnt = atomicAdd(data->rowCounter0 + dstRow, 1);
if (rowCnt < 608) {
slot32_t to_slot;
to_slot.x.x = tmp & 0xFFFFF;//from slot1.y xor, dest row was calculated from shr 20, but now we are taking lower 20 bits..?
to_slot.x.y = other_slot1.z ^ slot1.z;
to_slot.x.z = other_slot1.w ^ slot1.w;
to_slot.x.w = other_slot2.x ^ slot2.x;
data->round2.rows[dstRow].slots[rowCnt].x = to_slot.x;
to_slot.y.x = other_slot2.y ^ slot2.y;
//ENCODE_INPUTS(idx, j, tid);//  (((idx << 10) | d) << 10) | tid;
to_slot.y.y = ((j | (idx << 10)) << 10) | tid;
to_slot.y.z = 0;
to_slot.y.w = 0;
data->round2.rows[dstRow].slots[rowCnt].y = to_slot.y;
}
}
}
}
}
}


__global__
__launch_bounds__(608, 16)
void kernel_round3(data_t* data)
{
__shared__ uint		s_bincount[256];//256 counters, each index is the value, count is the current index
__shared__ uint4	s_slot1[608];//first 16 bytes
__shared__ uint		s_slot2[608];//last 4 bytes, 20 total bytes
__shared__ uint16_t	s_collisions[3072];

uint idx = blockIdx.x;
uint tid = threadIdx.x;
uint count = data->rowCounter0[idx];

if (tid < 256) {
s_bincount[tid] = 0;//reset bin count
}

__syncthreads();

if (tid == 0) {
data->rowCounter0[idx] = 0;//reset counter after we read it
}

count = min(count, 608);


for (; tid < 608; tid += blockDim.x) {
uint bin_idx = ~0UL;
uint4 slot1;
uint slot2;
uint bin = 0;
if (tid < count) {
slot1 = data->round2.rows[idx].slots[tid].x;
s_slot1[tid] = slot1;
slot2 = *(uint*)(&data->round2.rows[idx].slots[tid].y);
s_slot2[tid] = slot2;
bin = slot1.x >> 12;
if (s_bincount[bin] < 12) {
bin_idx = atomicAdd(&s_bincount[bin], 1);
bin_idx = min(bin_idx, 11);//something like only 12 collisions please, 0-11 index
s_collisions[bin * 12 + bin_idx] = tid;//nr_collision_slots is 12, so i can know how many slots are same value because they are being stored in this way
}
}

//binning is done for first 512 slots
__syncthreads();//because the shared memory writes

if (bin_idx != ~0UL) {
//12 is collision per element size
uint16_t * addr = &s_collisions[bin * 12];
for (uint i = 0; i < bin_idx; i++, addr++) {
uint j = *addr;
uint other_slot2 = s_slot2[j];
if (other_slot2 != slot2) {
uint4 other_slot1 = s_slot1[j];
uint dstRow = (slot1.x ^ other_slot1.x) & 4095;
uint rowCnt = atomicAdd(data->rowCounter1 + dstRow, 1);
if (rowCnt < 608) {
slot32_t to_slot;
to_slot.x.x = other_slot1.y ^ slot1.y;
to_slot.x.y = other_slot1.z ^ slot1.z;
to_slot.x.z = other_slot1.w ^ slot1.w;
to_slot.x.w = other_slot2 ^ slot2;
//ENCODE_INPUTS(idx, j, tid);
to_slot.y.x = ((j | (idx << 10)) << 10) | tid;
to_slot.y.y = 0;
to_slot.y.z = 0;
to_slot.y.w = 0;
data->round3.rows[dstRow].slots[rowCnt].x = to_slot.x;
data->round3.rows[dstRow].slots[rowCnt].y = to_slot.y;
}
}
}
}
}
}


__global__
__launch_bounds__(608, 16)
void kernel_round4(data_t* data)
{
__shared__ uint		s_bincount[256];//256 counters, each index is the value, count is the current index
__shared__ uint4	s_slot1[608];//first 16 bytes
__shared__ uint16_t	s_collisions[3072];

uint idx = blockIdx.x;
uint tid = threadIdx.x;
uint count = data->rowCounter1[idx];

if (tid < 256) {
s_bincount[tid] = 0;//reset bin count
}

__syncthreads();

if (tid == 0) {
data->rowCounter1[idx] = 0;//reset counter after we read it
}

count = min(count, 608);

for (; tid < 608; tid += blockDim.x) {
uint bin_idx = ~0UL;
uint4 slot1;
uint bin = 0;
if (tid < count) {
slot1 = data->round3.rows[idx].slots[tid].x;
s_slot1[tid] = slot1;
bin = slot1.x >> 24;//have to shift 24 because all 32 bits were here, basically collisions are the matchin top 8 bits same as 0xFF000000;
bin_idx = atomicAdd(&s_bincount[bin], 1);
bin_idx = min(bin_idx, 11);//something like only 12 collisions please, 0-11 index
s_collisions[bin * 12 + bin_idx] = tid;//nr_collision_slots is 12, so i can know how many slots are same value because they are being stored in this way
}

//binning is done for first 512 slots
__syncthreads();//because the shared memory writes

if (bin_idx != ~0UL) {
uint16_t * addr = &s_collisions[bin * 12];
for (uint i = 0; i < bin_idx; i++, addr++) {
uint j = *addr;
uint4 other_slot1 = s_slot1[j];
uint tmp;
uint dstRow;
if (other_slot1.w != slot1.w) {
asm volatile(
"xor.b32 %0, %2, %3;\n"
"bfe.u32 %1, %0, 12, 12;\n"
: "=r"(tmp), "=r"(dstRow)
: "r"(other_slot1.x),"r"(slot1.x)
);
uint rowCnt = atomicAdd(data->rowCounter0 + dstRow, 1);
if (rowCnt < 608) {
slot32_t to_slot;
to_slot.x.x = tmp & 4095;
to_slot.x.y = other_slot1.y ^ slot1.y;
to_slot.x.z = other_slot1.z ^ slot1.z;
to_slot.x.w = other_slot1.w ^ slot1.w;
to_slot.y.x = ((j | (idx << 10)) << 10) | tid;
to_slot.y.y = 0;
to_slot.y.z = 0;
to_slot.y.w = 0;
data->round4.rows[dstRow].slots[rowCnt].x = to_slot.x;
data->round4.rows[dstRow].slots[rowCnt].y = to_slot.y;
}
}
}
}
}
}


__global__
__launch_bounds__(608, 16)
void kernel_round5(data_t* data)
{
__shared__ uint		s_bincount[256];//256 counters, each index is the value, count is the current index
__shared__ uint4	s_slot1[608];//first 16 bytes
__shared__ uint16_t	s_collisions[3072];

uint idx = blockIdx.x;
uint tid = threadIdx.x;
uint count = data->rowCounter0[idx];

if (tid < 256) {
s_bincount[tid] = 0;//reset bin count
}

__syncthreads();

if (tid == 0) {
data->rowCounter0[idx] = 0;//reset counter after we read it
}
if (tid > 607) {
return;
}
count = min(count, 608);

for (; tid < 608; tid += blockDim.x) {
uint bin_idx = 0;
uint4 slot1;
uint bin = 0;
if (tid < count) {
slot1 = data->round4.rows[idx].slots[tid].x;
s_slot1[tid] = slot1;
bin = slot1.x >> 4;//have to shift 4 because 12 bits were here, basically collisions are the matchin top 8 bits
uint cnt = atomicAdd(&s_bincount[bin], 1);
bin_idx = min(cnt, 11);//something like only 12 collisions please, 0-11 index
s_collisions[bin * 12 + bin_idx] = tid;//nr_collision_slots is 12, so i can know how many slots are same value because they are being stored in this way
}

//binning is done for first 512 slots
__syncthreads();//because the shared memory writes

if (bin_idx >= 1) {
uint col_idx = bin * 12;
for (uint i = 0; i < bin_idx; i++, col_idx++) {
uint col = s_collisions[col_idx];
uint4 other_slot1 = s_slot1[col];
uint tmp;
if (other_slot1.w != slot1.w) {
uint r28 = other_slot1.y ^ slot1.y;
uint r53 = (other_slot1.x ^ slot1.x) & 15;
uint dst_row;
asm volatile("prmt.b32 %0, %1, %2, 13063;" : "=r"(dst_row) : "r"(r53), "r"(r28));
uint row_count = atomicAdd(&data->rowCounter1[dst_row], 1);
if (row_count < 608) {
uint4 to_slot;
to_slot.x = tmp & 0xFFFFFF;
to_slot.y = other_slot1.z ^ slot1.z;
to_slot.z = other_slot1.w ^ slot1.w;
to_slot.w = ((col | (idx << 10)) << 10) | tid;
data->round5.rows[dst_row].slots[row_count] = to_slot;
}
}
}
}
}
}

__global__
__launch_bounds__(608, 16)
void kernel_round6(data_t* data)
{
__shared__ uint		s_bincount[256];//256 counters, each index is the value, count is the current index
__shared__ uint4	s_slot1[608];//first 16 bytes
__shared__ uint16_t	s_collisions[3072];

uint idx = blockIdx.x;
uint tid = threadIdx.x;
uint count = data->rowCounter1[idx];

if (tid < 256) {
s_bincount[tid] = 0;//reset bin count
}

__syncthreads();

if (tid == 0) {
data->rowCounter1[idx] = 0;//reset counter after we read it
}

if (tid > 607) {
return;
}

count = min(count, 608);

for (; tid < 608; tid += blockDim.x) {
uint bin_idx = ~0UL;
uint4 slot1;
uint bin = 0;
if (tid < count) {
slot1 = data->round5.rows[idx].slots[tid];
s_slot1[tid] = slot1;
bin = slot1.x >> 16;//have to shift 4 because 12 bits were here, basically collisions are the matchin top 8 bits
bin_idx = atomicAdd(&s_bincount[bin], 1);
bin_idx = min(bin_idx, 11);//something like only 12 collisions please, 0-11 index
s_collisions[bin * 12 + bin_idx] = tid;//nr_collision_slots is 12, so i can know how many slots are same value because they are being stored in this way
}

//binning is done for first 512 slots
__syncthreads();//because the shared memory writes

if (bin_idx != ~0UL) {
uint16_t * addr = &s_collisions[bin * 12];
for (uint i = 0; i < bin_idx; i++, addr++) {
uint j = *addr;
uint4 other_slot1 = s_slot1[j];
uint tmp;
uint dstRow;
if (other_slot1.z != slot1.z) {
asm volatile(
"{\n\t"
".reg .u32 a,b;\n\t"
"xor.b32 %0, %2, %3;\n\t"
"bfe.u32 %1, %0, 4, 12;\n\t"
"}\n"
: "=r"(tmp), "=r"(dstRow)
: "r"(other_slot1.x), "r"(slot1.x)
);
uint rowCnt = atomicAdd(data->rowCounter0 + dstRow, 1);
if (rowCnt < 608) {
uint4 to_slot;
to_slot.x = ((tmp & 15) << 4) | ((other_slot1.y ^ slot1.y) >> 28);
to_slot.y = (other_slot1.y ^ slot1.y) & 0xFFFFFFF;
to_slot.z = other_slot1.z ^ slot1.z;
to_slot.w = ((j | (idx << 10)) << 10) | tid;
data->round6.rows[dstRow].slots[rowCnt] = to_slot;
}
}
}
}
}
}

__global__
__launch_bounds__(608, 16)
void kernel_round7(data_t* data)
{
__shared__ uint		s_bincount[256];//256 counters, each index is the value, count is the current index
__shared__ uint3	s_slot1[608];//first 16 bytes
__shared__ uint16_t	s_collisions[3072];

uint idx = blockIdx.x;
uint tid = threadIdx.x;
uint count = data->rowCounter0[idx];

if (tid < 256) {
s_bincount[tid] = 0;//reset bin count
}

__syncthreads();

if (tid == 0) {
data->rowCounter0[idx] = 0;//reset counter after we read it
}

count = min(count, 608);

for (; tid < 608; tid += blockDim.x) {
uint bin_idx = ~0UL;
uint4 slot1;
uint bin = 0;
if (tid < count) {
slot1 = data->round6.rows[idx].slots[tid];
s_slot1[tid] = *(uint3*)&slot1;
bin = slot1.x;
bin_idx = atomicAdd(&s_bincount[bin], 1);
bin_idx = min(bin_idx, 11);//something like only 12 collisions please, 0-11 index
s_collisions[bin * 12 + bin_idx] = tid;//nr_collision_slots is 12, so i can know how many slots are same value because they are being stored in this way
}

//binning is done for first 512 slots
__syncthreads();//because the shared memory writes

if (bin_idx != ~0UL) {
uint16_t * addr = &s_collisions[bin * 12];
for (uint i = 0; i < bin_idx; i++, addr++) {
uint j = *addr;
uint3 other_slot1 = s_slot1[j];
uint tmp;
uint dstRow;
if (other_slot1.z != slot1.z) {
asm volatile(
"xor.b32 %0, %2, %3;\n\t"
"shr.u32 %1, %0, 16;\n\t"
: "=r"(tmp), "=r"(dstRow)
: "r"(other_slot1.y), "r"(slot1.y)
);
uint rowCnt = atomicAdd(data->rowCounter1 + dstRow, 1);
if (rowCnt < 608) {
uint4 to_slot;
to_slot.x = tmp & 65535;
to_slot.y = other_slot1.z ^ slot1.z;
to_slot.z = ((j | (idx << 10)) << 10) | tid;// ENCODE_INPUTS(idx, j, tid);
to_slot.w = 0;
data->round7.rows[dstRow].slots[rowCnt] = to_slot;
}
}
}
}
}
}

__global__
__launch_bounds__(608, 16)
void kernel_round8(data_t* data)
{
__shared__ uint		s_bincount[256];//256 counters, each index is the value, count is the current index
__shared__ uint2	s_slot1[608];//first 16 bytes
__shared__ uint16_t	s_collisions[3072];

uint idx = blockIdx.x;
uint tid = threadIdx.x;
uint count = data->rowCounter1[idx];

if (tid < 256) {
s_bincount[tid] = 0;//reset bin count
}

__syncthreads();

if (tid == 0) {
data->rowCounter1[idx] = 0;//reset counter after we read it
}

count = min(count, 608);

for (; tid < 608; tid += blockDim.x) {
uint bin_idx = ~0UL;
uint2 slot1;
uint bin = 0;
if (tid < count) {
slot1 = *(uint2*)&data->round7.rows[idx].slots[tid];
s_slot1[tid] = slot1;
bin = slot1.x >> 8;
bin_idx = atomicAdd(&s_bincount[bin], 1);
bin_idx = min(bin_idx, 11);//something like only 12 collisions please, 0-11 index
s_collisions[bin * 12 + bin_idx] = tid;//nr_collision_slots is 12, so i can know how many slots are same value because they are being stored in this way
}

//binning is done for first 512 slots
__syncthreads();//because the shared memory writes

if (bin_idx != ~0UL) {
uint16_t * addr = &s_collisions[bin * 12];
for (uint i = 0; i < bin_idx; i++, addr++) {
uint j = *addr;
uint2 other_slot1 = s_slot1[j];
if (other_slot1.y != slot1.y) {
uint tmp = other_slot1.y ^ slot1.y;
uint dstRow = (((other_slot1.x ^ slot1.x) & 255) << 4) | (tmp >> 28);
uint rowCnt = atomicAdd(data->rowCounter0 + dstRow, 1);
if (rowCnt < 608) {
uint2 to_slot;
to_slot.x = tmp & 0xFFFFFFF;
to_slot.y = ((j | (idx << 10)) << 10) | tid;
data->round8.rows[dstRow].slots[rowCnt] = to_slot;
}
}
}
}
}
}



asm volatile ("{\n\t"
".reg .v4 .u32 v1,v2,v3,v4,v5,v6,v7,v8;\n\t"
"ld.param.v4.u32 v1, [%1];\n\t"
"ld.param.v4.u32 v2, [%1+16];\n\t"
"ld.param.v4.u32 v3, [%1+32];\n\t"
"ld.param.v4.u32 v4, [%1+48];\n\t"
"ld.param.v4.u32 v5, [%1+64];\n\t"
"ld.param.v4.u32 v6, [%1+80];\n\t"
"ld.param.v4.u32 v7, [%1+96];\n\t"
"ld.param.v4.u32 v8, [%1+112];\n\t"
"st.shared.v4.u32 [%0], v1;\n\t"
"st.shared.v4.u32 [%0+16], v2;\n\t"
"st.shared.v4.u32 [%0+32], v3;\n\t"
"st.shared.v4.u32 [%0+48], v4;\n\t"
"st.shared.v4.u32 [%0+64], v5;\n\t"
"st.shared.v4.u32 [%0+80], v6;\n\t"
"st.shared.v4.u32 [%0+96], v7;\n\t"
"st.shared.v4.u32 [%0+112], v8;\n\t"
"}\n"
: : "l"(sv),  "l"(blake_data)
);
*/