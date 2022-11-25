// Copyright (c) 2014 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_CRYPTO_SHA256_H
#define BITCOIN_CRYPTO_SHA256_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <stratum/crypto/common.hpp>
#include <stratum/crypto/detail/sha256.hpp>


namespace stratum {
namespace crypto {	

/** A hasher class for SHA-256. */
class CSHA256
{
public:
    static const size_t OUTPUT_SIZE = 32;

    CSHA256() {
		sha256_detail::Initialize(s);
	}
    CSHA256& Write(const unsigned char* data, size_t len) {
		const unsigned char* end = data + len;
		size_t bufsize = bytes % 64;
		if (bufsize && bufsize + len >= 64) {
			// Fill the buffer, and process it.
			memcpy(buf + bufsize, data, 64 - bufsize);
			bytes += 64 - bufsize;
			data += 64 - bufsize;
			sha256_detail::Transform(s, buf);
			bufsize = 0;
		}
		while (end >= data + 64) {
			// Process full chunks directly from the source.
			sha256_detail::Transform(s, data);
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
		WriteBE64(sizedesc, bytes << 3);
		Write(pad, 1 + ((119 - (bytes % 64)) % 64));
		Write(sizedesc, 8);
		FinalizeNoPadding(hash, false);		
	}
    void FinalizeNoPadding(unsigned char hash[OUTPUT_SIZE]) {
    	FinalizeNoPadding(hash, true);
    };
    CSHA256& Reset() {
		bytes = 0;
		sha256_detail::Initialize(s);
		return *this;
	}

private:
    uint32_t s[8];
    unsigned char buf[64];
    size_t bytes;
    void FinalizeNoPadding(unsigned char hash[OUTPUT_SIZE], bool enforce_compression) {
		if (enforce_compression && bytes != 64) {
			throw std::length_error("SHA256Compress should be invoked with a 512-bit block");
		}

		WriteBE32(hash, s[0]);
		WriteBE32(hash + 4, s[1]);
		WriteBE32(hash + 8, s[2]);
		WriteBE32(hash + 12, s[3]);
		WriteBE32(hash + 16, s[4]);
		WriteBE32(hash + 20, s[5]);
		WriteBE32(hash + 24, s[6]);
		WriteBE32(hash + 28, s[7]);
		
	}
};

} // namespace crypto
} // namespace stratum

#endif // BITCOIN_CRYPTO_SHA256_H
