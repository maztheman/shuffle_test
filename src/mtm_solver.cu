#include "mtm_solver.h"
#include "shuffle_test.h"

#include <fmt/format.h>

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#define checkCudaErrors(call)								\
do {														\
	cudaError_t err = call;									\
	if (cudaSuccess != err) {								\
		char errorBuff[512];								\
        sprintf(errorBuff, 			\
			"CUDA error '%s' in func '%s' line %d",			\
			cudaGetErrorString(err), __FUNCTION__, __LINE__);	\
		fprintf(stderr, "<error> %s\n", errorBuff); \
		}														\
} while (0)

cuda_mtm_solver::cuda_mtm_solver(int platf_id, int dev_id)
{
    device_id = dev_id;
	getinfo(platf_id, dev_id, m_gpu_name, m_sm_count, m_version);

    //completely ignore threads per block as we have the best. (tbb = number of slots)
    //we iterate over row number so more SMs means more rows on the fly we can calculate...yay
}

std::string cuda_mtm_solver::getdevinfo()
{
    return fmt::format("{} (#{}) BLOCKS={}, THREADS={}", m_gpu_name, device_id, NR_ROWS, NR_SLOTS);
}

int cuda_mtm_solver::getcount()
{
    int device_count;
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	return device_count;
}

void cuda_mtm_solver::getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version)
{
    cudaDeviceProp device_props;

	checkCudaErrors(cudaGetDeviceProperties(&device_props, d_id));

	gpu_name = device_props.name;
	sm_count = device_props.multiProcessorCount;
	version = fmt::format("{}.{}", device_props.major, device_props.minor);
}

void cuda_mtm_solver::start(cuda_mtm_solver& device_context)
{
    device_context.context = std::make_shared<mtm_cuda_context>(NR_SLOTS, NR_ROWS, device_context.device_id);
}

void cuda_mtm_solver::stop(cuda_mtm_solver& device_context)
{
    device_context.context.reset();
}

void cuda_mtm_solver::solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef,
		cuda_mtm_solver& device_context)
{
    device_context.context->solve(tequihash_header, tequihash_header_len, nonce, nonce_len, cancelf, solutionf, hashdonef);
}