#ifndef STRATUM_UTIL_STR_ENCODINGS_HPP
#define STRATUM_UTIL_STR_ENCODINGS_HPP

#include <stratum/arith/detail/utilstrencodings.hpp>

#define BEGIN(a)            ((char*)&(a))
#define END(a)              ((char*)&((&(a))[1]))

namespace stratum {
namespace arith {


inline signed char HexDigit(char c)
{
	return arith::arith_detail::p_util_hexdigit[(unsigned char)c];
}

template<typename T>
inline std::string HexStr(const T itbegin, const T itend, bool fSpaces=false)
{
    std::string rv;
    static const char hexmap[16] = { '0', '1', '2', '3', '4', '5', '6', '7',
                                     '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };
    rv.reserve((itend-itbegin)*3);
    for(T it = itbegin; it < itend; ++it)
    {
        unsigned char val = (unsigned char)(*it);
        if(fSpaces && it != itbegin)
            rv.push_back(' ');
        rv.push_back(hexmap[val>>4]);
        rv.push_back(hexmap[val&15]);
    }

    return rv;
}

template<typename T>
inline std::string HexStr(const T& vch, bool fSpaces=false)
{
    return HexStr(vch.begin(), vch.end(), fSpaces);
}
	
}
}



#endif