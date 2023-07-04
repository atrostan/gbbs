#pragma once

#include "gbbs/edge_map_reduce.h"
#include "gbbs/gbbs.h"

#include <math.h>

namespace gbbs {
  template<class Graph>
  sequence<double> CollaborativeFiltering(Graph &G,
                                          uint K,
                                          double step,
                                          double lambda,
                                          size_t max_iters = 100) {
    // using W = typename Graph::weight_type;
    const uintE n = G.n;
    auto frontier = sequence<bool>(n, true);
    vertexSubset Frontier(n, n, std::move(frontier));

    auto EM = EdgeMap<double, Graph>(
            G, std::make_tuple(UINT_E_MAX, static_cast<double>(0)),
            (size_t) G.m / 1000);

    auto latent_curr = sequence<double>(K * n, 0.5);
    auto error = sequence<double>(K * n, 0.5);

    // todo add flag for optional rand init of latent vector
    auto cond_f = [&](const uintE &v) { return true; };

    // update latent vector based on neighbours' data
    auto map_f = [&](const uintE &d, const uintE &s, const double &edgeLen) -> double {
      double estimate = 0;
      long current_offset = K * d, ngh_offset = K * s;
      double *cur_latent = &latent_curr[current_offset], *ngh_latent = &latent_curr[ngh_offset];
      for (int i = 0; i < K; i++) {
        estimate += cur_latent[i] * ngh_latent[i];
      }
      double err = edgeLen - estimate;

      double *cur_error = &error[current_offset];
      for (int i = 0; i < K; i++) {
        cur_error[i] += ngh_latent[i] * err;
      }
      return 1;
    };

    auto reduce_f = [&](double l, double r) { return l + r; };

    auto apply_f = [&](
                           std::tuple<uintE, double> k) -> std::optional<std::tuple<uintE, double>> {
      const uintE &i = std::get<0>(k);
      const double &contribution = std::get<1>(k);
      for (int j = 0; j < K; j++) {
        latent_curr[K * i + j] += step * (-lambda * latent_curr[K * i + j] + error[K * i + j]);
        error[K * i + j] = 0.0;
      }
      return std::nullopt;
    };
    timer ttt;
    ttt.start();
    uint iter = 0;
    for (iter = 0; iter < max_iters; iter++) {
      //edgemap to accumulate error for each node
      //vertexmap to update the latent vectors
      timer tt;
      tt.start();
      EM.template edgeMapReduce_dense<double, double>(
              Frontier, cond_f, map_f, reduce_f, apply_f, 0.0, no_output);
      tt.stop();
      tt.next("em time");
    }
    double total_time = ttt.stop();

    std::cout << "iter = " << iter << std::endl;
    std::cout << "total_time = " << total_time << std::endl;
    return latent_curr;
  }
}// namespace gbbs