// Copyright (c) 2014 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_CRYPTO_RIPEMD160_H
#define BITCOIN_CRYPTO_RIPEMD160_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stratum/crypto/common.hpp>
#include <stratum/crypto/detail/ripemd160.hpp>

namespace stratum {
namespace crypto {	


/** A hasher class for RIPEMD-160. */
class CRIPEMD160
{
private:
    uint32_t s[5];
    unsigned char buf[64];
    size_t bytes;

public:
    static const size_t OUTPUT_SIZE = 20;

    CRIPEMD160() {
		ripemd160_detail::Initialize(s);
	}
    CRIPEMD160& Write(const unsigned char* data, size_t len) {
		const unsigned char* end = data + len;
		size_t bufsize = bytes % 64;
		if (bufsize && bufsize + len >= 64) {
			// Fill the buffer, and process it.
			memcpy(buf + bufsize, data, 64 - bufsize);
			bytes += 64 - bufsize;
			data += 64 - bufsize;
			ripemd160_detail::Transform(s, buf);
			bufsize = 0;
		}
		while (end >= data + 64) {
			// Process full chunks directly from the source.
			ripemd160_detail::Transform(s, data);
			bytes += 64;
			data += 64;
		}
		if (end > data) {
			// Fill the buffer with what remains.
			memcpy(buf + bufsize, data, end - data);
			bytes += end - data;
		}
		return *this;		
	}
    void Finalize(unsigned char hash[OUTPUT_SIZE]) {
		static const unsigned char pad[64] = {0x80};
		unsigned char sizedesc[8];
		WriteLE64(sizedesc, bytes << 3);
		Write(pad, 1 + ((119 - (bytes % 64)) % 64));
		Write(sizedesc, 8);
		WriteLE32(hash, s[0]);
		WriteLE32(hash + 4, s[1]);
		WriteLE32(hash + 8, s[2]);
		WriteLE32(hash + 12, s[3]);
		WriteLE32(hash + 16, s[4]);		
	}
    CRIPEMD160& Reset() {
	    bytes = 0;
		ripemd160_detail::Initialize(s);
		return *this;
	}
};

} // namespace crypto
} // namespace stratum

#endif // BITCOIN_CRYPTO_RIPEMD160_H
