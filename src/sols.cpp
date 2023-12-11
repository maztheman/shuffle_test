#include "sols.h"
#include "param.h"
#include <memory>
#include <cassert>
#include <cstring>

static constexpr auto N = 1 << PARAM_K;

void compress(uint8_t *out, uint32_t *inputs)
{
	
	uint32_t byte_pos = 0;
	int32_t bits_left = PREFIX + 1;
	uint8_t x = 0;
	uint8_t x_bits_used = 0;
	uint8_t *pOut = out;
	while (byte_pos < N)
	{
		if (bits_left >= 8 - x_bits_used)
		{
			x |= inputs[byte_pos] >> (bits_left - 8 + x_bits_used);
			bits_left -= 8 - x_bits_used;
			x_bits_used = 8;
		}
		else if (bits_left > 0)
		{
			uint32_t mask = ~(-1 << (8 - x_bits_used));
			mask = ((~mask) >> bits_left) & mask;
			x |= (inputs[byte_pos] << (8 - x_bits_used - bits_left)) & mask;
			x_bits_used += bits_left;
			bits_left = 0;
		}
		else if (bits_left <= 0)
		{
			assert(!bits_left);
			byte_pos++;
			bits_left = PREFIX + 1;
		}
		if (x_bits_used == 8)
		{
			*pOut++ = x;
			x = x_bits_used = 0;
		}
	}
}

/*
** Sort a pair of binary blobs (a, b) which are consecutive in memory and
** occupy a total of 2*len 32-bit words.
**
** a            points to the pair
** len          number of 32-bit words in each pair
*/
static void sort_pair(uint32_t *a, uint32_t len)
{
	uint32_t    *b = a + len;
	uint32_t     tmp, need_sorting = 0;
	for (uint32_t i = 0; i < len; i++)
		if (need_sorting || a[i] > b[i])
		{
			need_sorting = 1;
			tmp = a[i];
			a[i] = b[i];
			b[i] = tmp;
		}
		else if (a[i] < b[i])
			return;
}

uint32_t verify_sol(sols_t *sols, unsigned sol_i)
{
	uint32_t  *inputs = sols->values[sol_i];
	uint32_t  seen_len = (1 << (PREFIX + 1)) / 8;
	uint8_t seen[(1 << (PREFIX + 1)) / 8];
	uint32_t  i;
	uint8_t tmp;
	// look for duplicate inputs
	memset(seen, 0, seen_len);
	for (i = 0; i < (1 << PARAM_K); i++)
	{
		tmp = seen[inputs[i] / 8];
		seen[inputs[i] / 8] |= 1 << (inputs[i] & 7);
		if (tmp == seen[inputs[i] / 8])
		{
			// at least one input value is a duplicate
			sols->valid[sol_i] = 0;
			return 0;
		}
	}
	// the valid flag is already set by the GPU, but set it again because
	// I plan to change the GPU code to not set it
	sols->valid[sol_i] = 1;
	// sort the pairs in place
	for (uint32_t level = 0; level < PARAM_K; level++)
		for (i = 0; i < N; i += (2 << level))
			sort_pair(&inputs[i], 1 << level);
	return 1;
}

uint32_t verify_sol(candidate_t *sols, unsigned sol_i, uint8_t (&valid)[16])
{
	uint32_t  *inputs = sols->vals[sol_i];
	uint32_t  seen_len = (1 << (PREFIX + 1)) / 8;
	uint8_t seen[(1 << (PREFIX + 1)) / 8];
	uint32_t  i;
	uint8_t tmp;
	// look for duplicate inputs
	memset(seen, 0, seen_len);
	for (i = 0; i < N; i++)
	{
		tmp = seen[inputs[i] / 8];
		seen[inputs[i] / 8] |= 1 << (inputs[i] & 7);
		if (tmp == seen[inputs[i] / 8]) {
			// at least one input value is a duplicate
			valid[sol_i] = 0;
			return 0;
		}
	}
	// the valid flag is already set by the GPU, but set it again because
	// I plan to change the GPU code to not set it
	valid[sol_i] = 1;
	// sort the pairs in place
	for (uint32_t level = 0; level < PARAM_K; level++)
		for (i = 0; i < N; i += (2 << level))
			sort_pair(&inputs[i], 1 << level);
	return 1;
}
