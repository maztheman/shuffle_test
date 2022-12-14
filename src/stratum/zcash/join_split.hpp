#ifndef _ZCJOINSPLIT_H_
#define _ZCJOINSPLIT_H_

#include <stratum/zcash/zcash.hpp>
#include <stratum/zcash/proof.hpp>
#include <stratum/zcash/address.hpp>
#include <stratum/zcash/note.hpp>
#include <stratum/zcash/incremental_merkle_tree.hpp>
#include <stratum/zcash/note_encryption.hpp>

#include <stratum/arith/uint256.hpp>
#include <stratum/arith/uint252.hpp>

#include <boost/array.hpp>

namespace stratum {
namespace zcash {	

class JSInput {
public:
    ZCIncrementalWitness witness;
    Note note;
    SpendingKey key;

    JSInput();
    JSInput(ZCIncrementalWitness witness,
            Note note,
            SpendingKey key) : witness(witness), note(note), key(key) { }

    uint256 nullifier() const {
        return note.nullifier(key);
    }
};

class JSOutput {
public:
    PaymentAddress addr;
    uint64_t value;
    boost::array<unsigned char, ZC_MEMO_SIZE> memo;

    JSOutput() { memo = {{ 0xF6 }}; };  // 0xF6 is invalid UTF8 as per spec, rest of array is 0x00
    JSOutput(PaymentAddress addr, uint64_t value) : addr(addr), value(value) { JSOutput(); }

    Note note(const uint252& phi, const uint256& r, size_t i, const uint256& h_sig) const;
};

template<size_t NumInputs, size_t NumOutputs>
class JoinSplit {
public:
    virtual ~JoinSplit() {}

    static JoinSplit<NumInputs, NumOutputs>* Generate();
    static JoinSplit<NumInputs, NumOutputs>* Unopened();
    static uint256 h_sig(const uint256& randomSeed,
                         const boost::array<uint256, NumInputs>& nullifiers,
                         const uint256& pubKeyHash
                        );

    // TODO: #789
    virtual void setProvingKeyPath(std::string) = 0;
    virtual void loadProvingKey() = 0;

    virtual void saveProvingKey(std::string path) = 0;
    virtual void loadVerifyingKey(std::string path) = 0;
    virtual void saveVerifyingKey(std::string path) = 0;

    virtual ZCProof prove(
        const boost::array<JSInput, NumInputs>& inputs,
        const boost::array<JSOutput, NumOutputs>& outputs,
        boost::array<Note, NumOutputs>& out_notes,
        boost::array<ZCNoteEncryption::Ciphertext, NumOutputs>& out_ciphertexts,
        uint256& out_ephemeralKey,
        const uint256& pubKeyHash,
        uint256& out_randomSeed,
        boost::array<uint256, NumInputs>& out_hmacs,
        boost::array<uint256, NumInputs>& out_nullifiers,
        boost::array<uint256, NumOutputs>& out_commitments,
        uint64_t vpub_old,
        uint64_t vpub_new,
        const uint256& rt,
        bool computeProof = true
    ) = 0;

    virtual bool verify(
        const ZCProof& proof,
        const uint256& pubKeyHash,
        const uint256& randomSeed,
        const boost::array<uint256, NumInputs>& hmacs,
        const boost::array<uint256, NumInputs>& nullifiers,
        const boost::array<uint256, NumOutputs>& commitments,
        uint64_t vpub_old,
        uint64_t vpub_new,
        const uint256& rt
    ) = 0;

protected:
    JoinSplit() {}
};

} // namespace zcash
} // namespace stratum

typedef stratum::zcash::JoinSplit<ZC_NUM_JS_INPUTS, ZC_NUM_JS_OUTPUTS> ZCJoinSplit;

#endif // _ZCJOINSPLIT_H_
