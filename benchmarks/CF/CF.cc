// This code is part of the project "Theoretically Efficient Parallel Graph
// Algorithms Can Be Fast and Scalable", presented at Symposium on Parallelism
// in Algorithms and Architectures, 2018.
// Copyright (c) 2018 Laxman Dhulipala, Guy Blelloch, and Julian Shun
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all  copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Usage:
// numactl -i all ./PageRank -s -m -rounds 3 twitter_SJ
// flags:
//   optional:
//     -eps : the epsilon to use for convergence (1e-6 by default)
//     -rounds : the number of times to run the algorithm
//     -c : indicate that the graph is compressed
//     -m : indicate that the graph should be mmap'd
//     -s : indicate that the graph is symmetric
//     -k : latent vector dimension
//     -step : stepsize 
//     -lambda : lambda, regularization parameter

#include "CF.h"

namespace gbbs {

  template<class Graph>
  double CF_runner(Graph &G, commandLine P) {
    std::cout << "### Application: CF" << std::endl;
    std::cout << "### Graph: " << P.getArgument(0) << std::endl;
    std::cout << "### Threads: " << num_workers() << std::endl;
    std::cout << "### n: " << G.n << std::endl;
    std::cout << "### m: " << G.m << std::endl;
    std::cout << "### Params: -eps = " << P.getOptionDoubleValue("-eps", 0.000001)
              << std::endl;

    uint k = P.getOptionIntValue("-k", 20);
    double step = P.getOptionDoubleValue("-step", 0.00000035);
    double lambda = P.getOptionDoubleValue("-lambda", 0.001);
    std::cout << "### k: " << k << std::endl;
    std::cout << "### step size: " << step << std::endl;
    std::cout << "### lambda: " << lambda << std::endl;
    
    std::cout << "### ------------------------------------" << std::endl;

    // 0.00000001

    timer t;
    t.start();
    double eps = P.getOptionDoubleValue("-eps", 0.000001);
    double local_eps = P.getOptionDoubleValue("-leps", 0.01);
    size_t iters = P.getOptionLongValue("-iters", 100);
    auto ret = CollaborativeFiltering(G, k, step, lambda, iters);

    double tt = t.stop();

    std::cout << "### Running Time: " << tt << std::endl;
    return tt;
  }

}// namespace gbbs

generate_weighted_main(gbbs::CF_runner, false);