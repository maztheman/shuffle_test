#pragma once

#define BOOST_LOG_CUSTOM(sev, pos) BOOST_LOG_TRIVIAL(sev) << "miner#" << pos << " | "

#include <cstdint>

#include <stratum/arith/arith_uint256.hpp>
#include <stratum/arith/uint256.hpp>
#include <stratum/zcash_job.hpp>
#include <boost/signals2.hpp>
#include <stratum/speed.hpp>
#include <stratum/detail/benchmark.hpp>
#include <stratum/detail/zcashthread.hpp>
#include <stratum/utils.hpp>

extern std::mutex benchmark_work;
extern std::vector<stratum::arith::uint256*> benchmark_nonces;
extern std::atomic_int benchmark_solutions;


namespace stratum {

typedef uint32_t eh_index;
typedef boost::signals2::signal<void (const stratum::ZcashJob*)> NewJob_t;

struct NopSolver
{

};

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
class ZcashMiner
{
    int nThreads;
	std::thread* minerThreads;
    arith::uint256 nonce1;
    size_t nonce1Size;
    arith::arith_uint256 nonce2Space;
    arith::arith_uint256 nonce2Inc;
    std::function<bool(const EquihashSolution&, const std::string&)> solutionFoundCallback;
	bool m_isActive;

	std::vector<CPUSolver*> cpu_contexts;
	std::vector<CUDASolver*> cuda_contexts;
	std::vector<OPENCLSolver*> opencl_contexts;

public:
	stratum::NewJob_t NewJob;
	bool* minerThreadActive;

	ZcashMiner(int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
		int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t) 
    : minerThreads{nullptr} {
        extern int use_avx2;

        m_isActive = false;
        nThreads = 0;

        for (int i = 0; i < cuda_count; ++i) {
            CUDASolver* context = new CUDASolver(0, cuda_en[i]);
            if (cuda_b[i] > 0)
                context->blocks = cuda_b[i];
            if (cuda_t[i] > 0)
                context->threadsperblock = cuda_t[i];

            cuda_contexts.push_back(context);
        }
        nThreads += cuda_contexts.size();

        if constexpr (!std::is_same_v<OPENCLSolver, NopSolver>)
        {
            for (int i = 0; i < opencl_count; ++i) {
                if (opencl_t[i] < 1) opencl_t[i] = 1;

                // add multiple threads if wanted
                for (int k = 0; k < opencl_t[i]; ++k) {
                    OPENCLSolver* context = new OPENCLSolver(opencl_platf, opencl_en[i]);
                    // todo: save local&global work size
                    opencl_contexts.push_back(context);
                }
            }
            nThreads += opencl_contexts.size();
        }

        if (cpu_threads < 0) {
            cpu_threads = std::thread::hardware_concurrency();
            if (cpu_threads < 1) {
                cpu_threads = 1;
            } else if (cuda_contexts.size() + opencl_contexts.size() > 0) { 
                --cpu_threads; // decrease number of threads if there are GPU workers
            }
        }

        if constexpr (!std::is_same_v<CPUSolver, NopSolver>)
        {
            for (int i = 0; i < cpu_threads; ++i) {
                CPUSolver* context = new CPUSolver();
                context->use_opt = use_avx2;
                cpu_contexts.push_back(context);
            }
            nThreads += cpu_contexts.size();
        }   
    }
    
	~ZcashMiner() {
        stop();
        for (auto it = cpu_contexts.begin(); it != cpu_contexts.end(); ++it)
            delete (*it);
        for (auto it = cuda_contexts.begin(); it != cuda_contexts.end(); ++it)
            delete (*it);
        cpu_contexts.clear();
        cuda_contexts.clear();        
    }

    std::string userAgent() {
        return "equihashminer/" STANDALONE_MINER_VERSION;
    }
    void start() {
        if (minerThreads) {
            stop();
        }

        m_isActive = true;

        minerThreads = new std::thread[nThreads];
        minerThreadActive = new bool[nThreads];

        // start cpu threads
        int i = 0;

        if constexpr(!std::is_same_v<CPUSolver, NopSolver>)
        {
            for ( ; i < cpu_contexts.size(); ++i) {
                minerThreadActive[i] = true;
                minerThreads[i] = std::thread(boost::bind(&miner_detail::ZcashMinerThread<CPUSolver, CUDASolver, OPENCLSolver, CPUSolver>, this, nThreads, i, *cpu_contexts.at(i)));
    #ifdef WIN32
    /*
            HANDLE hThread = minerThreads[i].native_handle();
            if (!SetThreadPriority(hThread, THREAD_PRIORITY_LOWEST))
            {
                BOOST_LOG_CUSTOM(warning, i) << "Failed to set low priority";
            }
            else
            {
                BOOST_LOG_CUSTOM(debug, i) << "Priority set to " << GetThreadPriority(hThread);
            }
    */
    #else
            // todo: linux set low priority
    #endif        
            }
        }
        
        if constexpr(!std::is_same_v<CUDASolver, NopSolver>)
        {        
            // start CUDA threads
            for (; i < (cpu_contexts.size() + cuda_contexts.size()); ++i) {
                minerThreadActive[i] = true;
                minerThreads[i] = std::thread(boost::bind(&miner_detail::ZcashMinerThread<CPUSolver, CUDASolver, OPENCLSolver, CUDASolver>, this, nThreads, i, *cuda_contexts.at(i - cpu_contexts.size())));
            }
        }

        if constexpr(!std::is_same_v<OPENCLSolver, NopSolver>)
        { 
            // start OPENCL threads
            for (; i < (cpu_contexts.size() + cuda_contexts.size() + opencl_contexts.size()); ++i) {
                minerThreadActive[i] = true;
                minerThreads[i] = std::thread(boost::bind(&miner_detail::ZcashMinerThread<CPUSolver, CUDASolver, OPENCLSolver, OPENCLSolver>, this, nThreads, i, *opencl_contexts.at(i - cpu_contexts.size() - cuda_contexts.size())));
            }
        }
        
        speed().Reset();
    }
    
    void stop() {
        m_isActive = false;
        if (minerThreads) {
            for (int i = 0; i < nThreads; i++) {
                minerThreadActive[i] = false;
            }
            for (int i = 0; i < nThreads; i++) {
                minerThreads[i].join();
            }
            
            delete [] minerThreads;
            minerThreads = nullptr;
            delete [] minerThreadActive;
        }
    }
	bool isMining() { return m_isActive; }
	void setServerNonce(const std::string& n1str) {
        BOOST_LOG_TRIVIAL(info) << "miner | Extranonce is " << n1str;
        std::vector<unsigned char> nonceData(ParseHex(n1str));
        while (nonceData.size() < 32) {
            nonceData.push_back(0);
        }
        CDataStream ss(nonceData, SER_NETWORK, PROTOCOL_VERSION);
        ss >> nonce1;

        nonce1Size = n1str.size();
        size_t nonce1Bits = nonce1Size * 4; // Hex length to bit length
        size_t nonce2Bits = 256 - nonce1Bits;

        nonce2Space = 1;
        nonce2Space <<= nonce2Bits;
        nonce2Space -= 1;

        nonce2Inc = 1;
        nonce2Inc <<= nonce1Bits;        
    }

    ZcashJob* parseJob(const json::Array& params) {
        if (params.size() < 2) {
            throw std::logic_error("Invalid job params");
        }

        ZcashJob* ret = new ZcashJob;
        ret->job = params[0].get_str();

        int32_t version;
        sscanf(params[1].get_str().c_str(), "%x", &version);
        // TODO: On a LE host shouldn't this be le32toh?
        ret->header.nVersion = be32toh(version);

        if (ret->header.nVersion == 4) {
            if (params.size() < 8) {
                throw std::logic_error("Invalid job params");
            }

            std::stringstream ssHeader;
            ssHeader << params[1].get_str()
                     << params[2].get_str()
                     << params[3].get_str()
                     << params[4].get_str()
                     << params[5].get_str()
                     << params[6].get_str()
                        // Empty nonce
                     << "0000000000000000000000000000000000000000000000000000000000000000"
                     << "00"; // Empty solution
            auto strHexHeader = ssHeader.str();
            std::vector<unsigned char> headerData(ParseHex(strHexHeader));
            CDataStream ss(headerData, SER_NETWORK, PROTOCOL_VERSION);
            try {
                ss >> ret->header;
            } catch (const std::ios_base::failure&) {
                throw std::logic_error("ZcashMiner::parseJob(): Invalid block header parameters");
            }

            ret->time = params[5].get_str();
            ret->clean = params[7].get_bool();
        } else {
            throw std::logic_error("ZcashMiner::parseJob(): Invalid or unsupported block header version");
        }

        ret->header.nNonce = nonce1;
        ret->nonce1Size = nonce1Size;
        ret->nonce2Space = nonce2Space;
        ret->nonce2Inc = nonce2Inc;

        return ret;
    }
    void setJob(ZcashJob* job) {
        NewJob(job);
    }
	void onSolutionFound(const std::function<bool(const EquihashSolution&, const std::string&)> callback) {
        solutionFoundCallback = callback;
    }
	void submitSolution(const EquihashSolution& solution, const std::string& jobid) {
        solutionFoundCallback(solution, jobid);
        speed().AddShare();
    }
    void acceptedSolution(bool) {
        speed().AddShareOK();
    }
    void rejectedSolution(bool) {
    }
    void failedSolution() {
    }

    static void doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
		int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t) {
		extern int use_avx2;
		
		using namespace arith;
        // generate array of various nonces
        std::srand(std::time(0));
        benchmark_nonces.push_back(new uint256());
        benchmark_nonces.back()->begin()[31] = 1;
        for (int i = 0; i < (hashes - 1); ++i) {
            benchmark_nonces.push_back(new uint256());
            for (unsigned int i = 0; i < 32; ++i) {
                benchmark_nonces.back()->begin()[i] = std::rand() % 256;
            }
        }
        benchmark_solutions = 0;

        size_t total_hashes = benchmark_nonces.size();

        std::vector<CPUSolver*> cpu_contexts;
        std::vector<CUDASolver*> cuda_contexts;
        std::vector<OPENCLSolver*> opencl_contexts;

        for (int i = 0; i < cuda_count; ++i) {
            CUDASolver* context = new CUDASolver(0, cuda_en[i]);
            if (cuda_b[i] > 0) {
                context->blocks = cuda_b[i];
            }
            if (cuda_t[i] > 0) {
                context->threadsperblock = cuda_t[i];
            }

            BOOST_LOG_TRIVIAL(info) << "Benchmarking CUDA worker (" << context->getname() << ") " << context->getdevinfo();

            CUDASolver::start(*context); // init CUDA before to get more accurate benchmark

            cuda_contexts.push_back(context);
        }

        if constexpr (!std::is_same_v<OPENCLSolver, NopSolver>)
        {
            for (int i = 0; i < opencl_count; ++i) {
                if (opencl_t[i] < 1) {
                    opencl_t[i] = 1;
                }

                for (int k = 0; k < opencl_t[i]; ++k) {
                    OPENCLSolver* context = new OPENCLSolver(opencl_platf, opencl_en[i]);
                    // todo: save local&global work size
                    BOOST_LOG_TRIVIAL(info) << "Benchmarking OPENCL worker (" << context->getname() << ") " << context->getdevinfo();
                    OPENCLSolver::start(*context); // init OPENCL before to get more accurate benchmark
                    opencl_contexts.push_back(context);
                }
            }
        }

        if (cpu_threads < 0) {
            cpu_threads = std::thread::hardware_concurrency();
            if (cpu_threads < 1) {
                cpu_threads = 1;
            } else if (cuda_contexts.size() + opencl_contexts.size() > 0) {
                --cpu_threads; // decrease number of threads if there are GPU workers
            }
        }

        if constexpr (!std::is_same_v<CPUSolver, NopSolver>)
        {
            for (int i = 0; i < cpu_threads; ++i) {
                CPUSolver* context = new CPUSolver;
                context->use_opt = use_avx2;
                BOOST_LOG_TRIVIAL(info) << "Benchmarking CPU worker (" << context->getname() << ") " << context->getdevinfo();
                CPUSolver::start(*context);
                cpu_contexts.push_back(context);
            }
        }

        int nThreads = cpu_contexts.size() + cuda_contexts.size() + opencl_contexts.size();

        std::thread* bthreads = new std::thread[nThreads];

        BOOST_LOG_TRIVIAL(info) << "Benchmark starting... this may take several minutes, please wait...";

        auto start = std::chrono::high_resolution_clock::now();

        int i = 0;
        if constexpr (!std::is_same_v<CPUSolver, NopSolver>)
        {
            for ( ; i < cpu_contexts.size(); ++i) {
                bthreads[i] = std::thread(boost::bind(&benchmark_detail::benchmark_thread<CPUSolver>, i, *cpu_contexts.at(i)));
            }
        }

        if constexpr (!std::is_same_v<CUDASolver, NopSolver>)
        {
            for (; i < (cuda_contexts.size() + cpu_contexts.size()); ++i) {
                bthreads[i] = std::thread(boost::bind(&benchmark_detail::benchmark_thread<CUDASolver>, i, *cuda_contexts.at(i - cpu_contexts.size())));
            }
        }

        if constexpr (!std::is_same_v<OPENCLSolver, NopSolver>)
        {
            for (; i < (opencl_contexts.size() + cuda_contexts.size() + cpu_contexts.size()); ++i) {
                bthreads[i] = std::thread(boost::bind(&benchmark_detail::benchmark_thread<OPENCLSolver>, i, *opencl_contexts.at(i - cpu_contexts.size() - cuda_contexts.size())));
            }
        }

        for (int i = 0; i < nThreads; ++i) {
            bthreads[i].join();
        }

        auto end = std::chrono::high_resolution_clock::now();

        uint64_t msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        size_t hashes_done = total_hashes - benchmark_nonces.size();

        if constexpr (!std::is_same_v<CPUSolver, NopSolver>)
        {
            for (auto it = cpu_contexts.begin(); it != cpu_contexts.end(); ++it) {
                CPUSolver::stop(**it);
                delete (*it);
            }
        }
        
        if constexpr (!std::is_same_v<CUDASolver, NopSolver>)
        {
            for (auto it = cuda_contexts.begin(); it != cuda_contexts.end(); ++it) {
                CUDASolver::stop(**it);
                delete (*it);
            }
        }
        
        if constexpr (!std::is_same_v<OPENCLSolver, NopSolver>)
        {
            for (auto it = opencl_contexts.begin(); it != opencl_contexts.end(); ++it) {
                OPENCLSolver::stop(**it);
                delete (*it);
            }
        }
        cpu_contexts.clear();
        cuda_contexts.clear();
        opencl_contexts.clear();

        BOOST_LOG_TRIVIAL(info) << "Benchmark done!";
        BOOST_LOG_TRIVIAL(info) << "Total time : " << msec << " ms";
        BOOST_LOG_TRIVIAL(info) << "Total iterations: " << hashes_done;
        BOOST_LOG_TRIVIAL(info) << "Total solutions found: " << benchmark_solutions;
        BOOST_LOG_TRIVIAL(info) << "Speed: " << ((double)hashes_done * 1000 / (double)msec) << " I/s";
        BOOST_LOG_TRIVIAL(info) << "Speed: " << ((double)benchmark_solutions * 1000 / (double)msec) << " Sols/s";        
    }
};

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver, typename Solver>
inline void static ZcashMinerThread(ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>* miner, int size, int pos, Solver& extra)
{
	BOOST_LOG_CUSTOM(info, pos) << "Starting thread #" << pos << " (" << extra.getname() << ") " << extra.getdevinfo();

	std::shared_ptr<std::mutex> m_zmt(new std::mutex);
	primitives::CBlockHeader header;
	arith::arith_uint256 space;
	size_t offset;
	arith::arith_uint256 inc;
	arith::arith_uint256 target;
	std::string jobId;
	std::string nTime;
	std::atomic_bool workReady{ false };
	std::atomic_bool cancelSolver{ false };
	std::atomic_bool pauseMining{ false };

	miner->NewJob.connect(NewJob_t::slot_type(
		[&m_zmt, &header, &space, &offset, &inc, &target, &workReady, &cancelSolver, pos, &pauseMining, &jobId, &nTime]
	(const ZcashJob* job) mutable {
		std::lock_guard<std::mutex> lock{ *m_zmt.get() };
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
		}
		else {
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
			arith::arith_uint256 nonce;
			arith::arith_uint256 nonceEnd;
			primitives::CBlockHeader actualHeader;
			std::string actualJobId;
			std::string actualTime;
			arith::arith_uint256 actualTarget;
			size_t actualNonce1size;
			{
				std::lock_guard<std::mutex> lock{ *m_zmt.get() };
				arith::arith_uint256 baseNonce = UintToArith256(header.nNonce);
				arith::arith_uint256 add(pos);
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
				primitives::CEquihashInput I{ actualHeader };
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
						actualHeader.nSolution =  benchmark_detail::GetMinimalFromIndices(index_vector, cbitlen);

					speed().AddSolution();

					BOOST_LOG_CUSTOM(debug, pos) << "Checking solution against target...";

					arith::uint256 headerhash = actualHeader.GetHash();
					if (UintToArith256(headerhash) > actualTarget) {
						BOOST_LOG_CUSTOM(debug, pos) << "Too large: " << headerhash.ToString();
						return;
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


} // namespace stratum
