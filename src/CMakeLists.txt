find_package(Boost 1.80.0 EXACT REQUIRED)

include(CheckIncludeFile)
include(TestBigEndian)
include(CheckFunctionExists)
include(CheckSymbolExists)

include(CheckCXXSymbolExists)

set(CMAKE_REQUIRED_INCLUDES "/usr/include;/usr/local/include")

check_include_file("endian.h" HAVE_ENDIAN_H)
check_include_file("byteswap.h" HAVE_BYTESWAP_H)

if(HAVE_BYTESWAP_H)
    check_cxx_symbol_exists("bswap_16" "byteswap.h" HAVE_DECL_BSWAP_16)
    check_cxx_symbol_exists("bswap_32" "byteswap.h" HAVE_DECL_BSWAP_32)
    check_cxx_symbol_exists("bswap_64" "byteswap.h" HAVE_DECL_BSWAP_64)
endif(HAVE_BYTESWAP_H)

test_big_endian(WORDS_BIGENDIAN)
if(WORDS_BIGENDIAN)
    set(BYTEORDER 4321)
else(WORDS_BIGENDIAN)
    set(BYTEORDER 1234)
endif(WORDS_BIGENDIAN)

check_cxx_symbol_exists(htobe16 "endian.h" HAVE_DECL_HTOBE16)
    
if(HAVE_ENDIAN_H)
    #set(CMAKE_REQUIRED_DEFINITIONS -DBYTEORDER=${BYTEORDER})
    #check_symbol_exists(htobe16 "endian.h" HAVE_DECL_HTOBE16)
    check_cxx_symbol_exists("htole16" "endian.h" HAVE_DECL_HTOLE16)
    check_cxx_symbol_exists(be16toh "endian.h" HAVE_DECL_BE16TOH) 
    check_cxx_symbol_exists("le16toh" "endian.h" HAVE_DECL_LE16TOH)
    check_cxx_symbol_exists("htobe32" "endian.h" HAVE_DECL_HTOBE32)
    check_cxx_symbol_exists("htole32" "endian.h" HAVE_DECL_HTOLE32)
    check_cxx_symbol_exists("be32toh" "endian.h" HAVE_DECL_BE32TOH)
    check_cxx_symbol_exists("le32toh" "endian.h" HAVE_DECL_LE32TOH)
    check_cxx_symbol_exists("htobe64" "endian.h" HAVE_DECL_HTOBE64)
    check_cxx_symbol_exists("htole64" "endian.h" HAVE_DECL_HTOLE64)
    check_cxx_symbol_exists("be64toh" "endian.h" HAVE_DECL_BE64TOH)
    check_cxx_symbol_exists("le64toh" "endian.h" HAVE_DECL_LE64TOH)
endif(HAVE_ENDIAN_H)

configure_file(stratum/config/bitcoin-config.hpp.cmake stratum/config/bitcoin-config.hpp)

add_executable(shuffle_test)

set(SRCS 
    shuffle_test.cu
    sols.cpp
)

target_sources(shuffle_test PRIVATE ${SRCS})
target_compile_definitions(shuffle_test PRIVATE -DHAVE_CONFIG_HPP)
target_include_directories(shuffle_test PRIVATE "." ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(shuffle_test PRIVATE Boost::Boost)
set_target_properties(shuffle_test PROPERTIES CUDA_ARCHITECTURES "61;72")
