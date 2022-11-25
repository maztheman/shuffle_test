#pragma once

#include <functional>
#include <vector>

struct st_context;

struct st_cuda_context
{
	int threadsperblock;
	int totalblocks;
	int device_id;
	st_context* eq;

	st_cuda_context(int tpb, int blocks, int id);
	~st_cuda_context();

	void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef);
};
