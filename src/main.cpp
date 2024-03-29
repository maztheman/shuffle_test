#include <stratum/zcash_miner.hpp>
#include <stratum/client.hpp>
#include "mtm_solver.h"
#include "api.h"

#include "cuda_dj.hpp"

#include <boost/log/core/core.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

namespace logging = boost::log;
namespace sinks = boost::log::sinks;
namespace src = boost::log::sources;


using MyMiner = stratum::ZcashMiner<stratum::NopSolver, cuda_mtm_solver, stratum::NopSolver>;
using MyStratumClient = stratum::StratumClient<MyMiner, stratum::ZcashJob, stratum::EquihashSolution>;

static int cuda_enabled[8] = { 0 };
static int cuda_blocks[8] = { 0 };
static int cuda_tpb[8] = { 0 };

static int opencl_enabled[8] = { 0 };
static int opencl_threads[8] = {0};

int use_avx2 = 0;

std::vector<stratum::arith::uint256*> benchmark_nonces;
std::mutex benchmark_work;
std::atomic_int benchmark_solutions;

static MyStratumClient* scSig;
extern "C" void stratum_sigint_handler(int signum) 
{ 
	if (scSig) scSig->disconnect(); 
	exit(0);
}

#if 1
template <typename MinerType, typename StratumType>
void start_mining(int api_port, int cpu_threads, int cuda_device_count, int opencl_device_count, int opencl_platform,
	const std::string& host, const std::string& port, const std::string& user, const std::string& password,
	StratumType* handler)
{
	std::shared_ptr<boost::asio::io_service> io_service(new boost::asio::io_service);

	API* api = nullptr;
	if (api_port > 0)
	{
		api = new API(io_service);
		if (!api->start(api_port))
		{
			delete api;
			api = nullptr;
		}
	}

	MinerType miner(cpu_threads, cuda_device_count, cuda_enabled, cuda_blocks, cuda_tpb, opencl_device_count, opencl_platform, opencl_enabled, opencl_threads);
	StratumType sc{

		io_service, &miner, host, port, user, password, 0, 0
	};

	miner.onSolutionFound([&](const stratum::EquihashSolution& solution, const std::string& jobid) {
		return sc.submit(&solution, jobid);
	});

	handler = &sc;
	signal(SIGINT, stratum_sigint_handler);

	int c = 0;
	while (sc.isRunning()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		if (++c % 1000 == 0)
		{
			double allshares = speed().GetShareSpeed() * 60;
			double accepted = speed().GetShareOKSpeed() * 60;
			BOOST_LOG_TRIVIAL(info) << CL_YLW "Speed [" << INTERVAL_SECONDS << " sec]: " <<
				speed().GetHashSpeed() << " I/s, " <<
				speed().GetSolutionSpeed() << " Sols/s" <<
				//accepted << " AS/min, " << 
				//(allshares - accepted) << " RS/min" 
				CL_N;
		}
		if (api) while (api->poll()) {}
	}

	if (api) delete api;
}

#endif

int main()
{
    int num_hashes = 2000;
    int cuda_device_count = 1;
	int cuda_bc = 0;
	int cuda_tbpc = 0;

    cuda_enabled[0] = 0;
    int log_level = 2;
		boost::log::add_console_log(
        std::clog,
		boost::log::keywords::auto_flush = true,
		boost::log::keywords::filter = boost::log::trivial::severity >= log_level,
		boost::log::keywords::format = (
		boost::log::expressions::stream
		<< "[" << boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%H:%M:%S")
		<< "][" << boost::log::expressions::attr<boost::log::attributes::current_thread_id::value_type>("ThreadID")
		<< "] "  << boost::log::expressions::smessage));

    MyMiner::doBenchmark(num_hashes, 24, cuda_device_count, cuda_enabled, cuda_blocks, cuda_tpb, 0, 0, nullptr, nullptr);


	//start_mining<MyMiner>(8080, 24, cuda_device_count, 0, 0, "equihash.f2pool.com", "5100", "maztheman.one", "", scSig);
	start_mining<MyMiner>(8080, 24, cuda_device_count, 0, 0, "zencash.f2pool.com", "3377", "maztheman.one", "", scSig);
//    start_mining<MyMiner>(8080, 24, cuda_device_count, 0, 0, "zec.f2pool.com", "3357", "maztheman.one", "", scSig);
}
