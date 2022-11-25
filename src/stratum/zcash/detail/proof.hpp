#ifndef STRATUM_ZCASH_DETAIL_PROOF_HPP
#define STRATUM_ZCASH_DETAIL_PROOF_HPP

namespace stratum {
namespace zcash {
namespace zcash_detail {

typedef alt_bn128_pp curve_pp;
typedef alt_bn128_pp::G1_type curve_G1;
typedef alt_bn128_pp::G2_type curve_G2;
typedef alt_bn128_pp::GT_type curve_GT;
typedef alt_bn128_pp::Fp_type curve_Fr;
typedef alt_bn128_pp::Fq_type curve_Fq;
typedef alt_bn128_pp::Fqe_type curve_Fq2;


// FE2IP as defined in the protocol spec and IEEE Std 1363a-2004.
inline bigint<8> fq2_to_bigint(const curve_Fq2 &e)
{
    auto modq = curve_Fq::field_char();
    auto c0 = e.c0.as_bigint();
    auto c1 = e.c1.as_bigint();

    bigint<8> temp = c1 * modq;
    temp += c0;
    return temp;
}

// Writes a bigint in big endian
template<mp_size_t LIMBS>
inline void write_bigint(base_blob<8 * LIMBS * sizeof(mp_limb_t)> &blob, const bigint<LIMBS> &val)
{
    auto ptr = blob.begin();
    for (ssize_t i = LIMBS-1; i >= 0; i--, ptr += 8) {
        WriteBE64(ptr, val.data[i]);
    }
}

// Reads a bigint from big endian
template<mp_size_t LIMBS>
inline bigint<LIMBS> read_bigint(const base_blob<8 * LIMBS * sizeof(mp_limb_t)> &blob)
{
    bigint<LIMBS> ret;

    auto ptr = blob.begin();

    for (ssize_t i = LIMBS-1; i >= 0; i--, ptr += 8) {
        ret.data[i] = ReadBE64(ptr);
    }

    return ret;
}
	
	
} // namespace zcash_detail
} // namespace zcash
} // namespace stratum



#endif
