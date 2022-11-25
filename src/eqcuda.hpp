#pragma once



typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
typedef unsigned char uchar;

struct packer_default;
struct packer_cantor;

#define MAXREALSOLS 9

struct scontainerreal
{
	u32 sols[MAXREALSOLS][512];
	u32 nsols;
};

template <u32 RB, u32 SM>
struct equi;

struct eq_cuda_context_interface
{
	virtual ~eq_cuda_context_interface();

	virtual void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef);
};


struct innercontext;

template <u32 RB, u32 SM, u32 SSM, u32 THREADS, typename PACKER>
struct eq_cuda_context : public eq_cuda_context_interface
{
	int threadsperblock;
	int totalblocks;
	int device_id;
	equi<RB, SM>* device_eq;
	scontainerreal* solutions;
	innercontext* innerctx;

	eq_cuda_context(int id);
	~eq_cuda_context();

	void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef);
};

#define CONFIG_MODE_1	9, 1248, 12, 640, packer_cantor

#define CONFIG_MODE_2	8, 640, 12, 512, packer_default