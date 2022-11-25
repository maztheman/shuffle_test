// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2013 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_PRIMITIVES_BLOCK_H
#define BITCOIN_PRIMITIVES_BLOCK_H

#include "transaction.hpp"

#include <stratum/serialize.hpp>
#include <stratum/arith/uint256.hpp>
#include <stratum/hash.hpp>
#include <stratum/tinyformat.hpp>
#include <stratum/arith/utilstrencodings.hpp>
#include <stratum/crypto/common.hpp>

namespace stratum {
namespace primitives {	

/** Nodes collect new transactions into a block, hash them into a hash tree,
 * and scan through nonce values to make the block's hash satisfy proof-of-work
 * requirements.  When they solve the proof-of-work, they broadcast the block
 * to everyone and the block is added to the block chain.  The first transaction
 * in the block is a special one that creates a new coin owned by the creator
 * of the block.
 */
class CBlockHeader
{
public:
    // header
    static const size_t HEADER_SIZE=4+32+32+32+4+4+32; // excluding Equihash solution
    static const int32_t CURRENT_VERSION=4;
    int32_t nVersion;
    uint256 hashPrevBlock;
    uint256 hashMerkleRoot;
    uint256 hashReserved;
    uint32_t nTime;
    uint32_t nBits;
    uint256 nNonce;
    std::vector<unsigned char> nSolution;

    CBlockHeader()
    {
        SetNull();
    }

    ADD_SERIALIZE_METHODS

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(this->nVersion);
        nVersion = this->nVersion;
        READWRITE(hashPrevBlock);
        READWRITE(hashMerkleRoot);
        READWRITE(hashReserved);
        READWRITE(nTime);
        READWRITE(nBits);
        READWRITE(nNonce);
        READWRITE(nSolution);
    }

    void SetNull()
    {
        nVersion = CBlockHeader::CURRENT_VERSION;
        hashPrevBlock.SetNull();
        hashMerkleRoot.SetNull();
        hashReserved.SetNull();
        nTime = 0;
        nBits = 0;
        nNonce = uint256();
        nSolution.clear();
    }

    bool IsNull() const
    {
        return (nBits == 0);
    }

    uint256 GetHash() const {
		return SerializeHash(*this);
	}

    int64_t GetBlockTime() const
    {
        return (int64_t)nTime;
    }
};


class CBlock : public CBlockHeader
{
public:
    // network and disk
    std::vector<CTransaction> vtx;

    // memory only
    mutable std::vector<uint256> vMerkleTree;

    CBlock()
    {
        SetNull();
    }

    CBlock(const CBlockHeader &header)
    {
        SetNull();
        *((CBlockHeader*)this) = header;
    }

    ADD_SERIALIZE_METHODS

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(*(CBlockHeader*)this);
        READWRITE(vtx);
    }

    void SetNull()
    {
        CBlockHeader::SetNull();
        vtx.clear();
        vMerkleTree.clear();
    }

    CBlockHeader GetBlockHeader() const
    {
        CBlockHeader block;
        block.nVersion       = nVersion;
        block.hashPrevBlock  = hashPrevBlock;
        block.hashMerkleRoot = hashMerkleRoot;
        block.hashReserved   = hashReserved;
        block.nTime          = nTime;
        block.nBits          = nBits;
        block.nNonce         = nNonce;
        block.nSolution      = nSolution;
        return block;
    }

    // Build the in-memory merkle tree for this block and return the merkle root.
    // If non-NULL, *mutated is set to whether mutation was detected in the merkle
    // tree (a duplication of transactions in the block leading to an identical
    // merkle root).
    uint256 BuildMerkleTree(bool* fMutated = NULL) const {
		/* WARNING! If you're reading this because you're learning about crypto
		   and/or designing a new system that will use merkle trees, keep in mind
		   that the following merkle tree algorithm has a serious flaw related to
		   duplicate txids, resulting in a vulnerability (CVE-2012-2459).

		   The reason is that if the number of hashes in the list at a given time
		   is odd, the last one is duplicated before computing the next level (which
		   is unusual in Merkle trees). This results in certain sequences of
		   transactions leading to the same merkle root. For example, these two
		   trees:

						A               A
					  /  \            /   \
					B     C         B       C
				   / \    |        / \     / \
				  D   E   F       D   E   F   F
				 / \ / \ / \     / \ / \ / \ / \
				 1 2 3 4 5 6     1 2 3 4 5 6 5 6

		   for transaction lists [1,2,3,4,5,6] and [1,2,3,4,5,6,5,6] (where 5 and
		   6 are repeated) result in the same root hash A (because the hash of both
		   of (F) and (F,F) is C).

		   The vulnerability results from being able to send a block with such a
		   transaction list, with the same merkle root, and the same block hash as
		   the original without duplication, resulting in failed validation. If the
		   receiving node proceeds to mark that block as permanently invalid
		   however, it will fail to accept further unmodified (and thus potentially
		   valid) versions of the same block. We defend against this by detecting
		   the case where we would hash two identical hashes at the end of the list
		   together, and treating that identically to the block having an invalid
		   merkle root. Assuming no double-SHA256 collisions, this will detect all
		   known ways of changing the transactions without affecting the merkle
		   root.
		*/
		vMerkleTree.clear();
		vMerkleTree.reserve(vtx.size() * 2 + 16); // Safe upper bound for the number of total nodes.
		for (std::vector<CTransaction>::const_iterator it(vtx.begin()); it != vtx.end(); ++it)
			vMerkleTree.push_back(it->GetHash());
		int j = 0;
		bool mutated = false;
		for (int nSize = vtx.size(); nSize > 1; nSize = (nSize + 1) / 2)
		{
			for (int i = 0; i < nSize; i += 2)
			{
				int i2 = std::min(i+1, nSize-1);
				if (i2 == i + 1 && i2 + 1 == nSize && vMerkleTree[j+i] == vMerkleTree[j+i2]) {
					// Two identical hashes at the end of the list at a particular level.
					mutated = true;
				}
				vMerkleTree.push_back(Hash(BEGIN(vMerkleTree[j+i]),  END(vMerkleTree[j+i]),
										   BEGIN(vMerkleTree[j+i2]), END(vMerkleTree[j+i2])));
			}
			j += nSize;
		}
		if (fMutated) {
			*fMutated = mutated;
		}
		return (vMerkleTree.empty() ? uint256() : vMerkleTree.back());		
	}

    std::vector<uint256> GetMerkleBranch(int nIndex) const {
		if (vMerkleTree.empty())
			BuildMerkleTree();
		std::vector<uint256> vMerkleBranch;
		int j = 0;
		for (int nSize = vtx.size(); nSize > 1; nSize = (nSize + 1) / 2)
		{
			int i = std::min(nIndex^1, nSize-1);
			vMerkleBranch.push_back(vMerkleTree[j+i]);
			nIndex >>= 1;
			j += nSize;
		}
		return vMerkleBranch;
	}
    static uint256 CheckMerkleBranch(uint256 hash, const std::vector<uint256>& vMerkleBranch, int nIndex)
	{
		if (nIndex == -1)
			return uint256();
		for (std::vector<uint256>::const_iterator it(vMerkleBranch.begin()); it != vMerkleBranch.end(); ++it)
		{
			if (nIndex & 1)
				hash = Hash(BEGIN(*it), END(*it), BEGIN(hash), END(hash));
			else
				hash = Hash(BEGIN(hash), END(hash), BEGIN(*it), END(*it));
			nIndex >>= 1;
		}
		return hash;
	}
    std::string ToString() const {
		std::stringstream s;
		s << strprintf("CBlock(hash=%s, ver=%d, hashPrevBlock=%s, hashMerkleRoot=%s, hashReserved=%s, nTime=%u, nBits=%08x, nNonce=%s, vtx=%u)\n",
			GetHash().ToString(),
			nVersion,
			hashPrevBlock.ToString(),
			hashMerkleRoot.ToString(),
			hashReserved.ToString(),
			nTime, nBits, nNonce.ToString(),
			vtx.size());
		/*for (unsigned int i = 0; i < vtx.size(); i++)
		{
			s << "  " << vtx[i].ToString() << "\n";
		}*/
		s << "  vMerkleTree: ";
		for (unsigned int i = 0; i < vMerkleTree.size(); i++)
			s << " " << vMerkleTree[i].ToString();
		s << "\n";
		return s.str();
	}
};


/**
 * Custom serializer for CBlockHeader that omits the nonce and solution, for use
 * as input to Equihash.
 */
class CEquihashInput : private CBlockHeader
{
public:
    CEquihashInput(const CBlockHeader &header)
    {
        CBlockHeader::SetNull();
        *((CBlockHeader*)this) = header;
    }

    ADD_SERIALIZE_METHODS

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(this->nVersion);
        nVersion = this->nVersion;
        READWRITE(hashPrevBlock);
        READWRITE(hashMerkleRoot);
        READWRITE(hashReserved);
        READWRITE(nTime);
        READWRITE(nBits);
    }
};


/** Describes a place in the block chain to another node such that if the
 * other node doesn't have the same branch, it can find a recent common trunk.
 * The further back it is, the further before the fork it may be.
 */
struct CBlockLocator
{
    std::vector<uint256> vHave;

    CBlockLocator() {}

    CBlockLocator(const std::vector<uint256>& vHaveIn)
    {
        vHave = vHaveIn;
    }

    ADD_SERIALIZE_METHODS

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        if (!(nType & SER_GETHASH))
            READWRITE(nVersion);
        READWRITE(vHave);
    }

    void SetNull()
    {
        vHave.clear();
    }

    bool IsNull() const
    {
        return vHave.empty();
    }
};

} // namespace primitives
} // namespace stratum


#endif // BITCOIN_PRIMITIVES_BLOCK_H
