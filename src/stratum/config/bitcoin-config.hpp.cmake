#pragma once

#cmakedefine HAVE_ENDIAN_H

/* whether byteorder is bigendian */
#cmakedefine WORDS_BIGENDIAN

#cmakedefine HAVE_BYTESWAP_H

/* whether bswap_16 exists */
#cmakedefine HAVE_DECL_BSWAP_16 1

/* whether bswap_32 exists */
#cmakedefine HAVE_DECL_BSWAP_32 1

/* whether bswap_64 exists */
#cmakedefine HAVE_DECL_BSWAP_64 1

#cmakedefine HAVE_DECL_HTOBE16 1
#cmakedefine HAVE_DECL_HTOLE16 1
#cmakedefine HAVE_DECL_BE16TOH 1
#cmakedefine HAVE_DECL_LE16TOH 1
#cmakedefine HAVE_DECL_HTOBE32 1
#cmakedefine HAVE_DECL_HTOLE32 1
#cmakedefine HAVE_DECL_BE32TOH 1
#cmakedefine HAVE_DECL_LE32TOH 1
#cmakedefine HAVE_DECL_HTOBE64 1
#cmakedefine HAVE_DECL_HTOLE64 1
#cmakedefine HAVE_DECL_BE64TOH 1
#cmakedefine HAVE_DECL_LE64TOH 1