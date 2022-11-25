#ifndef STRATUM_DETAIL_BENCHMARK_HPP
#define STRATUM_DETAIL_BENCHMARK_HPP

#include <atomic>
#include <cstdint>
#include <cassert>

#include <boost/log/trivial.hpp>

#include <stratum/primitives/block.hpp>
#include <stratum/arith/uint256.hpp>

extern std::mutex benchmark_work;
extern std::vector<stratum::arith::uint256*> benchmark_nonces;
extern std::atomic_int benchmark_solutions;

namespace stratum {
namespace benchmark_detail {
    
template <typename Solver>
bool benchmark_solve_equihash(const primitives::CBlock& pblock, const char *tequihash_header, unsigned int tequihash_header_len, Solver& extra);


template <typename Solver>
inline int benchmark_thread(int tid, Solver& extra)
{
	BOOST_LOG_TRIVIAL(debug) << "Thread #" << tid << " started (" << extra.getname() << ")";

	try
	{
		primitives::CBlock pblock;
		stratum::primitives::CEquihashInput I{ pblock };
		CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
		ss << I;

		const char *tequihash_header = (char *)&ss[0];
		unsigned int tequihash_header_len = ss.size();

		while (benchmark_detail::benchmark_solve_equihash<Solver>(pblock, tequihash_header, tequihash_header_len, extra)) {}
	}
	catch (const std::runtime_error &e)
	{
		BOOST_LOG_TRIVIAL(error) << e.what();
		exit(0);
		return 0;
	}

	BOOST_LOG_TRIVIAL(debug) << "Thread #" << tid << " ended (" << extra.getname() << ")";

	return 0;
}

inline void CompressArray(const unsigned char* in, size_t in_len, unsigned char* out, size_t out_len, size_t bit_len, size_t byte_pad)
{
	assert(bit_len >= 8);
	assert(8 * sizeof(uint32_t) >= 7 + bit_len);

	size_t in_width{ (bit_len + 7) / 8 + byte_pad };
	assert(out_len == bit_len*in_len / (8 * in_width));

	uint32_t bit_len_mask{ ((uint32_t)1 << bit_len) - 1 };

	// The acc_bits least-significant bits of acc_value represent a bit sequence
	// in big-endian order.
	size_t acc_bits = 0;
	uint32_t acc_value = 0;

	size_t j = 0;
	for (size_t i = 0; i < out_len; i++) {
		// When we have fewer than 8 bits left in the accumulator, read the next
		// input element.
		if (acc_bits < 8) {
			acc_value = acc_value << bit_len;
			for (size_t x = byte_pad; x < in_width; x++) {
				acc_value = acc_value | (
					(
					// Apply bit_len_mask across byte boundaries
					in[j + x] & ((bit_len_mask >> (8 * (in_width - x - 1))) & 0xFF)
					) << (8 * (in_width - x - 1))); // Big-endian
			}
			j += in_width;
			acc_bits += bit_len;
		}

		acc_bits -= 8;
		out[i] = (acc_value >> acc_bits) & 0xFF;
	}
}

using eh_index = uint32_t;

inline void EhIndexToArray(const eh_index i, unsigned char* array)
{
	BOOST_STATIC_ASSERT(sizeof(eh_index) == 4);
	eh_index bei = htobe32(i);
	memcpy(array, &bei, sizeof(eh_index));
}


inline std::vector<unsigned char> GetMinimalFromIndices(std::vector<eh_index> indices, size_t cBitLen)
{
	assert(((cBitLen + 1) + 7) / 8 <= sizeof(eh_index));
	size_t lenIndices{ indices.size()*sizeof(eh_index) };
	size_t minLen{ (cBitLen + 1)*lenIndices / (8 * sizeof(eh_index)) };
	size_t bytePad{ sizeof(eh_index) - ((cBitLen + 1) + 7) / 8 };
	std::vector<unsigned char> array(lenIndices);
	for (int i = 0; i < indices.size(); i++) {
		EhIndexToArray(indices[i], array.data() + (i*sizeof(eh_index)));
	}
	std::vector<unsigned char> ret(minLen);
	CompressArray(array.data(), lenIndices, ret.data(), minLen, cBitLen + 1, bytePad);
	return ret;
}

template <typename Solver>
inline bool benchmark_solve_equihash(const primitives::CBlock& pblock, const char *tequihash_header, unsigned int tequihash_header_len, Solver& extra)
{


	benchmark_work.lock();
	if (benchmark_nonces.empty())
	{
		benchmark_work.unlock();
		return false;
	}
	arith::uint256* nonce = benchmark_nonces.front();
	benchmark_nonces.erase(benchmark_nonces.begin());
	benchmark_work.unlock();

	BOOST_LOG_TRIVIAL(debug) << "Testing, nonce = " << nonce->ToString();

	std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionFound =
		[&pblock, &nonce]
	(const std::vector<uint32_t>& index_vector, size_t cbitlen, const unsigned char* compressed_sol)
	{
		primitives::CBlockHeader hdr = pblock.GetBlockHeader();
		hdr.nNonce = *nonce;

		if (compressed_sol)
		{
			hdr.nSolution = std::vector<unsigned char>(1344);
			for (size_t i = 0; i < cbitlen; ++i)
				hdr.nSolution[i] = compressed_sol[i];
		}
		else
			hdr.nSolution = GetMinimalFromIndices(index_vector, cbitlen);

		BOOST_LOG_TRIVIAL(debug) << "Solution found, header = " << hdr.GetHash().ToString();

		++benchmark_solutions;
	};

	Solver::solve(tequihash_header,
		tequihash_header_len,
		(const char*)nonce->begin(),
		nonce->size(),
		[]() { return false; },
		solutionFound,
		[]() {},
		extra);

	delete nonce;

	return true;
}



    
} // namespace benchmark_detail
} // namespace stratum



#endif