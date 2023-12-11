#pragma once

#include "param.h"

#include <functional>
#include <vector>
#include <cstdint>
#include <memory>

struct mtm_context;

struct mtm_cuda_context
{
    mtm_cuda_context(int tpb, int blocks, int id);

    void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef);

    std::shared_ptr<mtm_context> context;
};