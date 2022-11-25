#ifndef STRATUM_SPEED_HPP
#define STRATUM_SPEED_HPP

#include <iostream>
#include <chrono>
#include <vector>
#include <mutex>

#define INTERVAL_SECONDS 300 // 5 minutes

namespace stratum {

class Speed
{
	int m_interval;

	using time_point = std::chrono::high_resolution_clock::time_point;

	time_point m_start;

	std::vector<time_point> m_buffer_hashes;
	std::vector<time_point> m_buffer_solutions;
	std::vector<time_point> m_buffer_shares;
	std::vector<time_point> m_buffer_shares_ok;

	std::mutex m_mutex_hashes;
	std::mutex m_mutex_solutions;
	std::mutex m_mutex_shares;
	std::mutex m_mutex_shares_ok;

	void Add(std::vector<time_point>& buffer, std::mutex& mutex) {
		mutex.lock();
		buffer.push_back(std::chrono::high_resolution_clock::now());
		mutex.unlock();
	}
	double Get(std::vector<time_point>& buffer, std::mutex& mutex) {
		time_point now = std::chrono::high_resolution_clock::now();
		time_point past = now - std::chrono::seconds(m_interval);
		double interval = (double)m_interval;
		if (past < m_start)
		{
			interval = ((double)std::chrono::duration_cast<std::chrono::milliseconds>(now - m_start).count()) / 1000;
			past = m_start;
		}
		size_t total = 0;

		mutex.lock();
		for (std::vector<time_point>::iterator it = buffer.begin(); it != buffer.end();)
		{
			if ((*it) < past)
			{
				it = buffer.erase(it);
			}
			else
			{
				++total;
				++it;
			}
		}
		mutex.unlock();

		return (double)total / (double)interval;		
	}

public:
	Speed(int interval)
	: m_interval(interval), m_start(std::chrono::high_resolution_clock::now()) {
		
	}

	virtual ~Speed() {}

	void AddHash() {
		Add(m_buffer_hashes, m_mutex_hashes);
	}
	void AddSolution() {
		Add(m_buffer_solutions, m_mutex_solutions);
	}
	void AddShare() {
		Add(m_buffer_shares, m_mutex_shares);
	}
	void AddShareOK() {
		Add(m_buffer_shares_ok, m_mutex_shares_ok);
	}
	double GetHashSpeed() {
		return Get(m_buffer_hashes, m_mutex_hashes);
	}
	double GetSolutionSpeed() {
		return Get(m_buffer_solutions, m_mutex_solutions);
	}
	double GetShareSpeed() {
		return Get(m_buffer_shares, m_mutex_shares);
	}
	double GetShareOKSpeed() {
		return Get(m_buffer_shares_ok, m_mutex_shares_ok);
	}

	void Reset() {
		m_mutex_hashes.lock();
		m_buffer_hashes.clear();
		m_mutex_hashes.unlock();

		m_mutex_solutions.lock();
		m_buffer_solutions.clear();
		m_mutex_solutions.unlock();

		m_mutex_shares.lock();
		m_buffer_shares.clear();
		m_mutex_shares.unlock();

		m_mutex_shares_ok.lock();
		m_buffer_shares_ok.clear();
		m_mutex_shares_ok.unlock();

		m_start = std::chrono::high_resolution_clock::now();		
	}
};

}


inline stratum::Speed& speed()
{
	static stratum::Speed instance(INTERVAL_SECONDS);
	return instance;
}

#endif