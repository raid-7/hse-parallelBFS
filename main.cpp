#include <iostream>
#include <vector>
#include <deque>
#include <random>
#include <numeric>
#include <atomic>
#include <chrono>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <tbb/global_control.h>

struct Result
{
    double value;
    double variation;
};

std::ostream& operator<<(std::ostream& os, const Result& result)
{
    os << result.value << " +- " << result.variation;
    return os;
}

Result stats(const std::vector<double>& data) {
    double sum = 0.0;
    for (double x : data) {
        sum += x;
    }
    double mean = sum / data.size();
    double sqSum = 0.0;
    for (double x : data) {
        sqSum += x*x - mean * mean;
    }
    double stddev = data.size() < 2 ? 0.0 : std::sqrt(sqSum / (data.size() - 1));
    return {mean, stddev};
}

class Timer {
public:
    Timer() {
        Reset();
    }

    void Reset()
    {
        StartTime_ = std::chrono::high_resolution_clock::now();
    }

    double Nanoseconds()
    {
        auto finishTime = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(finishTime - StartTime_).count();
    }

    double Microseconds()
    {
        return Nanoseconds() / 1000.;
    }

    double Seconds()
    {
        return Microseconds() / 1'000'000.;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> StartTime_;
};

template <class V>
std::ostream& operator <<(std::ostream& stream, const std::vector<V>& v) {
    for (const auto& x : v) {
        stream << x << ' ';
    }
    return stream;
}

template <class F, class V>
Result time(size_t numRuns, F func, const V& arg) {
    std::vector<double> results;
    for (size_t i = 0; i < numRuns; ++i) {
        Timer t;
        func(arg);
        results.push_back(t.Microseconds() / 1'000'000.0);
    }
    return stats(results);
}

class CubeGraph {
public:
    struct Vertex {
        int x, y, z;

        auto operator <=>(const Vertex&) const = default;
    };

    explicit CubeGraph(int side)
            : side(side)
    {}

    std::vector<int> adjacent(int i) const {
        std::vector<int> res;
        res.reserve(6);
        Vertex v = fromInt(i);
        for (int* p : std::array{&v.x, &v.y, &v.z}) {
            for (int diff : std::array{-1, +1}) {
                *p += diff;
                Vertex u = v;
                *p -= diff;
                if (isValid(u)) {
                    res.push_back(toInt(u));
                }
            }
        }
        return res;
    }

    size_t adjacentCount(int i) const {
        Vertex v = fromInt(i);
        return (v.x > 0) + (v.y > 0) + (v.z > 0) + (v.x < side - 1) + (v.y < side - 1) + (v.z < side - 1);
    }

    inline size_t count() const {
        return side * side * side;
    }

    std::vector<size_t> computeDistancesToOrigin() const {
        std::vector<size_t> result(count());
        for (int i = 0; i < side * side * side; ++i) {
            Vertex v = fromInt(i);
            result[i] = v.x + v.y + v.z;
        }
        return result;
    }

    inline Vertex fromInt(int v) const {
        return {
            v % side,
            (v / side) % side,
            (v / (side * side)) % side
        };
    }

    inline int toInt(Vertex v) const {
        return v.x + v.y * side + v.z * side * side;
    }

private:
    inline bool isValid(Vertex v) const {
        return v.x >= 0 && v.y >= 0 && v.z >= 0
               && v.x < side && v.y < side && v.z < side;
    }

private:
    int side;
};

std::ostream& operator <<(std::ostream& stream, CubeGraph::Vertex v) {
    return stream << "V[" << v.x << ", " << v.y << ", " << v.z << "]";
}

template <class FIn>
void doSerialPrefixSum(FIn in, size_t size, std::vector<size_t>& out) {
    for (size_t i = 0; i < size; ++i) {
        out[i + 1] = out[i] + in(i);
    }
}

template <class FIn>
void doParallelPrefixSum(FIn in, size_t size, std::vector<size_t>& out) {
    tbb::parallel_scan(tbb::blocked_range<size_t>(0, size),
       0,
       [&](tbb::blocked_range<size_t> r, size_t sum, bool is_final_scan) {
           for (size_t i = r.begin(); i < r.end(); ++i) {
               sum += in(i);
               if  (is_final_scan)
                   out[i + 1] = sum;
           }
           return sum;
       },
       [](size_t left, size_t right) {
           return left + right;
       }
    );
}


template <class FIn>
void prefixSum(FIn in, size_t size, std::vector<size_t>& out) {
    out.resize(size + 1);
    out[0] = 0;

    if (size < 25'000) {
        doSerialPrefixSum(in, size, out);
    } else {
        doParallelPrefixSum(in, size, out);
    }
}

template <size_t limit = 25'000, class Func>
void maybeParallelFor(size_t size, Func func) {
    if (size < limit) {
        for (size_t i = 0; i < size; ++i)
            func(i);
    } else {
        tbb::parallel_for(size_t(0), size, func);
    }
}

/*
 * vertices of g are indexed from 0 to g.count()-1
 * g has methods:
 * - count -> size_t
 * - adjacent -> random access container
 * - adjacentCount -> size_t
 */
template <class Graph>
std::vector<size_t> parallelBfs(const Graph& g, int startVertex) {
    struct Mark {
        alignas(8) std::atomic<bool> value{false};

        bool reserve() {
            return !value.exchange(true);
        }
    };

    std::vector<Mark> marks(g.count());
    marks[startVertex].reserve();
    std::vector<size_t> result(g.count() /*, std::numeric_limits<size_t>::max() */);
    std::vector<int> frontier = {startVertex};
    std::vector<int> newFrontier = {};
    std::vector<size_t> degPrefSum;
    size_t dist = 0;

    while (!frontier.empty()) {
        prefixSum([&](size_t i) {
            int v = frontier[i];
            return v < 0 ? 0 : g.adjacentCount(v);
        }, frontier.size(), degPrefSum);

        newFrontier.resize(degPrefSum.back(), -1);

        maybeParallelFor<512>(frontier.size(), [&](size_t i) {
            int v = frontier[i];
            if (v < 0)
                return;
            size_t offset = degPrefSum[i];
            const auto& adjacent = g.adjacent(v);
            maybeParallelFor(adjacent.size(), [&](size_t j) {
                int u = adjacent[j];
                if (marks[u].reserve())
                    newFrontier[offset + j] = u;
            });
            result[v] = dist;
            frontier[i] = -1;
        });

        std::swap(frontier, newFrontier);
        ++dist;
    }

    return result;
}

template <class Graph>
std::vector<size_t> bfs(const Graph& g, int startVertex) {
    std::deque<int> queue = {startVertex};
    std::vector<size_t> result(g.count(), std::numeric_limits<size_t>::max());
    result[startVertex] = 0;
    while (!queue.empty()) {
        int v = queue.front();
        queue.pop_front();
        size_t d = result[v];
        for (int u : g.adjacent(v)) {
            if (d + 1 < result[u]) {
                result[u] = d + 1;
                queue.push_back(u);
            }
        }
    }
    return result;
}



int main() {
    tbb::global_control concurrencyLimit(tbb::global_control::max_allowed_parallelism, 4);

    CubeGraph g(500);

    std::cout << "Data generated." << std::endl;
//    std::cout << "Verifying correctness:" << std::endl;
//    {
//        auto fakeRes = g.computeDistancesToOrigin();
//        {
//            auto serialRes = bfs(g, 0);
//            std::cout << "Serial version correct: " << std::boolalpha << (serialRes == fakeRes) << std::endl;
//        }
//        {
//            auto parallelRes = parallelBfs(g, 0);
//            std::cout << "Parallel version correct: " << std::boolalpha << (parallelRes == fakeRes) << std::endl;
//        }
//    }

//    std::cout << "Running serial version:" << std::endl;
//    std::cout << "Serial: " << time(3, [](const CubeGraph& g) {
//        return bfs(g, 0);
//    }, g) << std::endl;
    std::cout << "Running parallel version:" << std::endl;
    std::cout << "Parallel: " << time(3, [](const CubeGraph& g) {
        return parallelBfs(g, 0);
    }, g) << std::endl;
    return 0;
}
