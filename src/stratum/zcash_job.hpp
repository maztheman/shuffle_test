#ifndef STRATUM_ZCASH_JOB_HPP
#define STRATUM_ZCASH_JOB_HPP

// Copyright (c) 2016 Jack Grigg <jack@z.cash>
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <stratum/arith/arith_uint256.hpp>
#include <stratum/primitives/block.hpp>
#include <stratum/arith/uint256.hpp>
#include <stratum/json/json_spirit_value.hpp>
#include <stratum/solver_stub.hpp>
#include <stratum/version.hpp>
#include <stratum/arith/utilstrencodings.hpp>
#include <stratum/streams.hpp>

#include <boost/thread/exceptions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/static_assert.hpp>

#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include <mutex>


//#define BOOST_LOG_CUSTOM(sev, pos) BOOST_LOG_TRIVIAL(sev) << "miner#" << pos << " | "

namespace stratum {

struct EquihashSolution
{
    arith::uint256 nonce;
	std::string time;
	size_t nonce1size;
    std::vector<unsigned char> solution;

    EquihashSolution(arith::uint256 n, std::vector<unsigned char> s, std::string t, size_t n1s)
		: nonce{ n }, nonce1size{ n1s } { solution = s; time = t; }

    std::string toString() const { return nonce.GetHex(); }
};

struct ZcashJob
{
    std::string job;
    primitives::CBlockHeader header;
    std::string time;
    size_t nonce1Size;
    arith::arith_uint256 nonce2Space;
    arith::arith_uint256 nonce2Inc;
    arith::arith_uint256 serverTarget;
    bool clean;

    ZcashJob* clone() const {
        ZcashJob* ret = new ZcashJob;
        ret->job = job;
        ret->header = header;
        ret->time = time;
        ret->nonce1Size = nonce1Size;
        ret->nonce2Space = nonce2Space;
        ret->nonce2Inc = nonce2Inc;
        ret->serverTarget = serverTarget;
        ret->clean = clean;
        return ret;
    }
    
    bool equals(const ZcashJob& a) const { return job == a.job; }

    // Access Stratum flags
    std::string jobId() const { return job; }
    bool cleanJobs() const { return clean; }

    void setTarget(std::string target) {
        if (target.size() > 0) {
            serverTarget = stratum::arith::UintToArith256(stratum::arith::uint256S(target));
        } else {
            BOOST_LOG_TRIVIAL(debug) << "miner | New job but no server target, assuming powLimit";
            serverTarget = stratum::arith::UintToArith256(stratum::arith::uint256S("0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f"));
        }
    }

    /**
     * Returns a comma-separated string of Stratum submission values
     * corresponding to the given solution.
     */
    std::string getSubmission(const EquihashSolution* solution) {
        CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
        ss << solution->nonce;
        ss << solution->solution;
        std::string strHex = stratum::arith::HexStr(ss.begin(), ss.end());

        std::stringstream stream;
        stream << "\"" << job;
        stream << "\",\"" << time;
        stream << "\",\"" << strHex.substr(nonce1Size, 64-nonce1Size);
        stream << "\",\"" << strHex.substr(64);
        stream << "\"";
        return stream.str();    
    }
};

inline bool operator==(const ZcashJob& a, const ZcashJob& b)
{
    return a.equals(b);
}

} // namespace stratum
    
#endif