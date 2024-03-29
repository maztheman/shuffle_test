// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2014 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_UINT256_H
#define BITCOIN_UINT256_H

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <vector>

#include <stratum/arith/detail/uint256.hpp>
#include <stratum/arith/utilstrencodings.hpp>

#ifdef _MSC_VER
# define _ALIGN(x) __declspec(align(x))
#else
# define _ALIGN(x) __attribute__ ((aligned(x)))
#endif

namespace stratum {
	namespace arith {

		/** Template base class for fixed-sized opaque blobs. */
		template<unsigned int BITS>
		class base_blob
		{
		protected:
			enum { WIDTH = BITS / 8 };
			uint8_t _ALIGN(4) data[WIDTH];
		public:
			base_blob()
			{
				memset(data, 0, sizeof(data));
			}

			explicit base_blob(const std::vector<unsigned char>& vch) {
				assert(vch.size() == sizeof(data));
				memcpy(data, &vch[0], sizeof(data));
			}

			bool IsNull() const
			{
				for (int i = 0; i < WIDTH; i++)
					if (data[i] != 0)
						return false;
				return true;
			}

			void SetNull()
			{
				memset(data, 0, sizeof(data));
			}

			friend inline bool operator==(const base_blob& a, const base_blob& b) { return memcmp(a.data, b.data, sizeof(a.data)) == 0; }
			friend inline bool operator!=(const base_blob& a, const base_blob& b) { return memcmp(a.data, b.data, sizeof(a.data)) != 0; }
			friend inline bool operator<(const base_blob& a, const base_blob& b) 
			{ 
				return memcmp(a.data, b.data, sizeof(a.data)) < 0; 
			}

			std::string GetHex() const {
				char psz[sizeof(data) * 2 + 1];
				for (unsigned int i = 0; i < sizeof(data); i++)
					sprintf(psz + i * 2, "%02x", data[sizeof(data) - i - 1]);
				return std::string(psz, psz + sizeof(data) * 2);
			}

			void SetHex(const char* psz) {
				memset(data, 0, sizeof(data));

				// skip leading spaces
				while (isspace(*psz))
					psz++;

				// skip 0x
				if (psz[0] == '0' && tolower(psz[1]) == 'x')
					psz += 2;

				// hex string to uint
				const char* pbegin = psz;
				while (HexDigit(*psz) != -1)
					psz++;
				psz--;
				unsigned char* p1 = (unsigned char*)data;
				unsigned char* pend = p1 + WIDTH;
				while (psz >= pbegin && p1 < pend) {
					*p1 = HexDigit(*psz--);
					if (psz >= pbegin) {
						*p1 |= ((unsigned char)HexDigit(*psz--) << 4);
						p1++;
					}
				}
			}
			void SetHex(const std::string& str) {
				SetHex(str.c_str());
			}
			std::string ToString() const {
				return (GetHex());
			}

			unsigned char* begin()
			{
				return &data[0];
			}

			unsigned char* end()
			{
				return &data[WIDTH];
			}

			const unsigned char* begin() const
			{
				return &data[0];
			}

			const unsigned char* end() const
			{
				return &data[WIDTH];
			}

			unsigned int size() const
			{
				return sizeof(data);
			}

			unsigned int GetSerializeSize(int nType, int nVersion) const
			{
				return sizeof(data);
			}

			template<typename Stream>
			void Serialize(Stream& s, int nType, int nVersion) const
			{
				s.write((char*)data, sizeof(data));
			}

			template<typename Stream>
			void Unserialize(Stream& s, int nType, int nVersion)
			{
				s.read((char*)data, sizeof(data));
			}
		};

		/** 160-bit opaque blob.
		 * @note This type is called uint160 for historical reasons only. It is an opaque
		 * blob of 160 bits and has no integer operations.
		 */
		class uint160 : public base_blob<160> {
		public:
			uint160() {}
			uint160(const base_blob<160>& b) : base_blob<160>(b) {}
			explicit uint160(const std::vector<unsigned char>& vch) : base_blob<160>(vch) {}
		};


		// Explicit instantiations for base_blob<160>
		template base_blob<160>::base_blob(const std::vector<unsigned char>&);
		template std::string base_blob<160>::GetHex() const;
		template std::string base_blob<160>::ToString() const;
		template void base_blob<160>::SetHex(const char*);
		template void base_blob<160>::SetHex(const std::string&);

		// Explicit instantiations for base_blob<256>
		template base_blob<256>::base_blob(const std::vector<unsigned char>&);
		template std::string base_blob<256>::GetHex() const;
		template std::string base_blob<256>::ToString() const;
		template void base_blob<256>::SetHex(const char*);
		template void base_blob<256>::SetHex(const std::string&);

		/** 256-bit opaque blob.
		 * @note This type is called uint256 for historical reasons only. It is an
		 * opaque blob of 256 bits and has no integer operations. Use arith_uint256 if
		 * those are required.
		 */
		class uint256 : public base_blob<256> {
		public:
			uint256() {}
			uint256(const base_blob<256>& b) : base_blob<256>(b) {}
			explicit uint256(const std::vector<unsigned char>& vch) : base_blob<256>(vch) {}

			/** A cheap hash function that just returns 64 bits from the result, it can be
			 * used when the contents are considered uniformly random. It is not appropriate
			 * when the value can easily be influenced from outside as e.g. a network adversary could
			 * provide values to trigger worst-case behavior.
			 * @note The result of this function is not stable between little and big endian.
			 */
			uint64_t GetCheapHash() const
			{
				uint64_t result;
				memcpy((void*)&result, (void*)data, 8);
				return result;
			}

			/** A more secure, salted hash function.
			 * @note This hash is not stable between little and big endian.
			 */
			inline uint64_t GetHash(const uint256& salt) const {
				uint32_t a, b, c;
				const uint32_t *pn = (const uint32_t*)data;
				const uint32_t *salt_pn = (const uint32_t*)salt.data;
				a = b = c = 0xdeadbeef + WIDTH;

				a += pn[0] ^ salt_pn[0];
				b += pn[1] ^ salt_pn[1];
				c += pn[2] ^ salt_pn[2];
				arith::arith_detail::HashMix(a, b, c);
				a += pn[3] ^ salt_pn[3];
				b += pn[4] ^ salt_pn[4];
				c += pn[5] ^ salt_pn[5];
				arith::arith_detail::HashMix(a, b, c);
				a += pn[6] ^ salt_pn[6];
				b += pn[7] ^ salt_pn[7];
				arith::arith_detail::HashFinal(a, b, c);

				return ((((uint64_t)b) << 32) | c);
			}
		};

		/* uint256 from const char *.
		 * This is a separate function because the constructor uint256(const char*) can result
		 * in dangerously catching uint256(0).
		 */
		inline uint256 uint256S(const char *str)
		{
			uint256 rv;
			rv.SetHex(str);
			return rv;
		}
		/* uint256 from std::string.
		 * This is a separate function because the constructor uint256(const std::string &str) can result
		 * in dangerously catching uint256(0) via std::string(const char*).
		 */
		inline uint256 uint256S(const std::string& str)
		{
			uint256 rv;
			rv.SetHex(str);
			return rv;
		}

	}
}
#endif // BITCOIN_UINT256_H
