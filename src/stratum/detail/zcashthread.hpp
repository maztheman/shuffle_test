#pragma once

#include "benchmark.hpp"

namespace stratum
{
    template<typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
    class ZcashMiner;
}

namespace stratum::miner_detail
{
    using NewJob_t = boost::signals2::signal<void (const stratum::ZcashJob*)>;
    using namespace stratum::primitives;
    using namespace stratum::arith;

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver, typename Solver>
void static ZcashMinerThread(ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>* miner, int size, int pos, Solver& extra)
{
	BOOST_LOG_CUSTOM(info, pos) << "Starting thread #" << pos << " (" << extra.getname() << ") " << extra.getdevinfo();

    std::shared_ptr<std::mutex> m_zmt(new std::mutex);
    CBlockHeader header;
    arith_uint256 space;
    size_t offset;
    arith_uint256 inc;
    arith_uint256 target;
	std::string jobId;
	std::string nTime;
    std::atomic_bool workReady {false};
    std::atomic_bool cancelSolver {false};
	std::atomic_bool pauseMining {false};

    miner->NewJob.connect(NewJob_t::slot_type(
		[&m_zmt, &header, &space, &offset, &inc, &target, &workReady, &cancelSolver, pos, &pauseMining, &jobId, &nTime]
        (const ZcashJob* job) mutable {
            std::lock_guard<std::mutex> lock{*m_zmt.get()};
            if (job) {
				BOOST_LOG_CUSTOM(debug, pos) << "Loading new job #" << job->jobId();
				jobId = job->jobId();
				nTime = job->time;
                header = job->header;
                space = job->nonce2Space;
                offset = job->nonce1Size * 4; // Hex length to bit length
                inc = job->nonce2Inc;
                target = job->serverTarget;
				pauseMining.store(false);
                workReady.store(true);
                /*if (job->clean) {
                    cancelSolver.store(true);
                }*/
            } else {
                workReady.store(false);
                cancelSolver.store(true);
				pauseMining.store(true);
            }
        }
    ).track_foreign(m_zmt)); // So the signal disconnects when the mining thread exits

    try {

		Solver::start(extra);

        while (true) {
            // Wait for work
            bool expected;
            do {
                expected = true;
				if (!miner->minerThreadActive[pos])
					throw boost::thread_interrupted();
                //boost::this_thread::interruption_point();
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            } while (!workReady.compare_exchange_weak(expected, false));
            // TODO change atomically with workReady
            cancelSolver.store(false);

            // Calculate nonce limits
            arith_uint256 nonce;
            arith_uint256 nonceEnd;
			CBlockHeader actualHeader;
			std::string actualJobId;
			std::string actualTime;
			arith_uint256 actualTarget;
			size_t actualNonce1size;
            {
                std::lock_guard<std::mutex> lock{*m_zmt.get()};
                arith_uint256 baseNonce = UintToArith256(header.nNonce);
				arith_uint256 add(pos);
				nonce = baseNonce | (add << (8 * 19));
				nonceEnd = baseNonce | ((add + 1) << (8 * 19));
				//nonce = baseNonce + ((space/size)*pos << offset);
				//nonceEnd = baseNonce + ((space/size)*(pos+1) << offset);

				// save job id and time
				actualHeader = header;
				actualJobId = jobId;
				actualTime = nTime;
				actualNonce1size = offset / 4;
				actualTarget = target;
            }

			// I = the block header minus nonce and solution.
			CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
			{
				//std::lock_guard<std::mutex> lock{ *m_zmt.get() };
				CEquihashInput I{ actualHeader };
				ss << I;
			}

			char *tequihash_header = (char *)&ss[0];
			unsigned int tequihash_header_len = ss.size();

            // Start working
            while (true) {
				BOOST_LOG_CUSTOM(debug, pos) << "Running Equihash solver with nNonce = " << nonce.ToString();

				auto bNonce = ArithToUint256(nonce);

				std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionFound =
					[&actualHeader, &bNonce, &actualTarget, &miner, pos, &actualJobId, &actualTime, &actualNonce1size]
				(const std::vector<uint32_t>& index_vector, size_t cbitlen, const unsigned char* compressed_sol) 
				{
					actualHeader.nNonce = bNonce;
					if (compressed_sol)
					{
						actualHeader.nSolution = std::vector<unsigned char>(1344);
						for (size_t i = 0; i < cbitlen; ++i)
							actualHeader.nSolution[i] = compressed_sol[i];
					}
					else
						actualHeader.nSolution = stratum::benchmark_detail::GetMinimalFromIndices(index_vector, cbitlen);

					speed().AddSolution();

					BOOST_LOG_CUSTOM(debug, pos) << "Checking solution against target...";

					uint256 headerhash = actualHeader.GetHash();
					if (UintToArith256(headerhash) > actualTarget) {
						BOOST_LOG_CUSTOM(debug, pos) << "Too large: " << headerhash.ToString();
//						return;
					}

					// Found a solution
					BOOST_LOG_CUSTOM(debug, pos) << "Found solution with header hash: " << headerhash.ToString();
					EquihashSolution solution{ bNonce, actualHeader.nSolution, actualTime, actualNonce1size };
					miner->submitSolution(solution, actualJobId);
				};

				std::function<bool()> cancelFun = [&cancelSolver]() {
					return cancelSolver.load();
				};

				std::function<void(void)> hashDone = []() {
					speed().AddHash();
				};

				Solver::solve(tequihash_header,
					tequihash_header_len,
					(const char*)bNonce.begin(),
					bNonce.size(),
					cancelFun,
					solutionFound,
					hashDone,
					extra);
				
                // Check for stop
				if (!miner->minerThreadActive[pos])
					throw boost::thread_interrupted();
                //boost::this_thread::interruption_point();

				// Update nonce
				nonce += inc;

                if (nonce == nonceEnd) {
                    break;
                }

                // Check for new work
                if (workReady.load()) {
					BOOST_LOG_CUSTOM(debug, pos) << "New work received, dropping current work";
                    break;
                }

				if (pauseMining.load())
				{
					BOOST_LOG_CUSTOM(debug, pos) << "Mining paused";
					break;
				}
            }
        }
    }
    catch (const boost::thread_interrupted&)
    {
        //throw;
    }
    catch (const std::runtime_error &e)
    {
		BOOST_LOG_CUSTOM(error, pos) << e.what();
    }

	try
	{
		Solver::stop(extra);
	}
	catch (const std::runtime_error &e)
	{
		BOOST_LOG_CUSTOM(error, pos) << e.what();
	}

	BOOST_LOG_CUSTOM(info, pos) << "Thread #" << pos << " ended (" << extra.getname() << ")";
}
}
