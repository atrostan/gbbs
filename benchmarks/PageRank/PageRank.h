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
#pragma once

#include "gbbs/edge_map_reduce.h"
#include "gbbs/gbbs.h"

#include <math.h>

namespace gbbs {

  typedef float ScoreT;

  template<class Graph>
  struct PR_F {
    using W = typename Graph::weight_type;
    ScoreT *p_curr, *p_next;
    Graph &G;
    PR_F(ScoreT *_p_curr, ScoreT *_p_next, Graph &G)
        : p_curr(_p_curr), p_next(_p_next), G(G) {}
    inline bool update(
            const uintE &s, const uintE &d,
            const W &wgh) {// update function applies PageRank equation
      p_next[d] += p_curr[s] / G.get_vertex(s).out_degree();
      return 1;
    }
    inline bool updateAtomic(const uintE &s, const uintE &d,
                             const W &wgh) {// atomic Update
      gbbs::fetch_and_add(&p_next[d], p_curr[s] / G.get_vertex(s).out_degree());
      return 1;
    }
    inline bool cond(intT d) { return cond_true(d); }
  };

  // vertex map function to update its p value according to PageRank equation
  struct PR_Vertex_F {
    ScoreT damping;
    ScoreT addedConstant;
    ScoreT *p_curr;
    ScoreT *p_next;
    PR_Vertex_F(ScoreT *_p_curr, ScoreT *_p_next, ScoreT _damping, intE n)
        : damping(_damping),
          addedConstant((1 - _damping) * (1 / (ScoreT) n)),
          p_curr(_p_curr),
          p_next(_p_next) {}
    inline bool operator()(uintE i) {
      p_next[i] = damping * p_next[i] + addedConstant;
      return 1;
    }
  };

  // resets p
  struct PR_Vertex_Reset {
    ScoreT *p_curr;
    PR_Vertex_Reset(ScoreT *_p_curr) : p_curr(_p_curr) {}
    inline bool operator()(uintE i) {
      p_curr[i] = 0.0;
      return 1;
    }
  };

  template<class Graph>
  sequence<ScoreT> PageRank_edgeMap(Graph &G, ScoreT eps = 0.000001,
                                    size_t max_iters = 100) {
    const uintE n = G.n;
    const ScoreT damping = 0.85;

    ScoreT one_over_n = 1 / (ScoreT) n;
    auto p_curr = sequence<ScoreT>(n, one_over_n);
    auto p_next = sequence<ScoreT>(n, static_cast<ScoreT>(0));
    auto frontier = sequence<bool>(n, true);

    // read from special array of just degrees

    auto degrees = sequence<uintE>::from_function(
            n, [&](size_t i) { return G.get_vertex(i).out_degree(); });

    vertexSubset Frontier(n, n, std::move(frontier));

    size_t iter = 0;
    while (iter++ < max_iters) {
      gbbs_debug(timer t; t.start(););
      // SpMV
      edgeMap(G, Frontier, PR_F<Graph>(p_curr.begin(), p_next.begin(), G), 0,
              no_output);
      vertexMap(Frontier,
                PR_Vertex_F(p_curr.begin(), p_next.begin(), damping, n));

      // Check convergence: compute L1-norm between p_curr and p_next
      auto differences = parlay::delayed_seq<ScoreT>(
              n, [&](size_t i) { return fabs(p_curr[i] - p_next[i]); });
      ScoreT L1_norm = parlay::reduce(differences);
      if (L1_norm < eps) break;

      gbbs_debug(std::cout << "L1_norm = " << L1_norm << std::endl;);
      // Reset p_curr
      parallel_for(0, n, [&](size_t i) { p_curr[i] = static_cast<ScoreT>(0); });
      std::swap(p_curr, p_next);

      gbbs_debug(t.stop(); t.next("iteration time"););
    }
    auto max_pr = parlay::reduce_max(p_next);
    std::cout << "max_pr = " << max_pr << std::endl;
    return p_next;
  }

  template<class Graph>
  sequence<ScoreT> PageRank(Graph &G, ScoreT eps = 0.000001,
                            size_t max_iters = 100) {
    using W = typename Graph::weight_type;
    const uintE n = G.n;
    const ScoreT damping = 0.85;
    const ScoreT addedConstant = (1 - damping) * (1 / static_cast<ScoreT>(n));

    ScoreT one_over_n = 1 / (ScoreT) n;
    auto p_curr = sequence<ScoreT>(n, one_over_n);
    auto p_next = sequence<ScoreT>(n, static_cast<ScoreT>(0));
    auto frontier = sequence<bool>(n, true);
    auto p_div = sequence<ScoreT>::from_function(n, [&](size_t i) -> ScoreT {
      return one_over_n / static_cast<ScoreT>(G.get_vertex(i).out_degree());
    });
    auto p_div_next = sequence<ScoreT>(n);


    // read from special array of just degrees

    auto degrees = sequence<uintE>::from_function(
            n, [&](size_t i) { return G.get_vertex(i).out_degree(); });

    vertexSubset Frontier(n, n, std::move(frontier));
    auto EM = EdgeMap<ScoreT, Graph>(
            G, std::make_tuple(UINT_E_MAX, static_cast<ScoreT>(0)),
            (size_t) G.m / 1000);

    auto cond_f = [&](const uintE &v) { return true; };
    auto map_f = [&](const uintE &d, const uintE &s, const W &wgh) -> ScoreT {
      return p_div[s];
      //    return p_curr[s] / degrees[s];
    };
    auto reduce_f = [&](ScoreT l, ScoreT r) { return l + r; };
    auto apply_f = [&](
                           std::tuple<uintE, ScoreT> k) -> std::optional<std::tuple<uintE, ScoreT>> {
      const uintE &u = std::get<0>(k);
      const ScoreT &contribution = std::get<1>(k);
      p_next[u] = damping * contribution + addedConstant;
      p_div_next[u] = p_next[u] / static_cast<ScoreT>(degrees[u]);
      return std::nullopt;
    };

    size_t iter = 0;
    timer ttt;
    ttt.start();
    while (iter++ < max_iters) {
      timer t;
      t.start();
      // SpMV
      timer tt;
      tt.start();
      EM.template edgeMapReduce_dense<ScoreT, ScoreT>(
              Frontier, cond_f, map_f, reduce_f, apply_f, 0.0, no_output);
      tt.stop();
      tt.next("em time");

      // Check convergence: compute L1-norm between p_curr and p_next
      // auto differences = parlay::delayed_seq<ScoreT>(n, [&](size_t i) {
      //   auto d = p_curr[i];
      //   p_curr[i] = 0;
      //   return fabs(d - p_next[i]);
      // });
      // ScoreT L1_norm = parlay::reduce(differences, parlay::plus<ScoreT>());
      // if (L1_norm < eps) break;
      // gbbs_debug(std::cout << "L1_norm = " << L1_norm << std::endl;);

      // Reset p_curr and p_div
      std::swap(p_curr, p_next);
      std::swap(p_div, p_div_next);
      t.stop();
      t.next("iteration time");
    }
    double total_time = ttt.stop();

    auto max_pr = parlay::reduce_max(p_next);
    std::cout << "iter = " << iter << std::endl;
    std::cout << "total_time = " << total_time << std::endl;
    std::cout << "max_pr = " << max_pr << std::endl;
    return p_next;
  }

  namespace delta {

    struct delta_and_degree {
      ScoreT delta;
      ScoreT delta_over_degree;
    };

    template<class Graph>
    struct PR_Delta_F {
      using W = typename Graph::weight_type;
      Graph &G;
      delta_and_degree *Delta;
      ScoreT *nghSum;
      PR_Delta_F(Graph &G, delta_and_degree *_Delta, ScoreT *_nghSum)
          : G(G), Delta(_Delta), nghSum(_nghSum) {}
      inline bool update(const uintE &s, const uintE &d, const W &wgh) {
        ScoreT oldVal = nghSum[d];
        nghSum[d] += Delta[s].delta_over_degree;// Delta[s].delta/Delta[s].degree;
                                                // // V[s].out_degree();
        return oldVal == 0;
      }
      inline bool updateAtomic(const uintE &s, const uintE &d, const W &wgh) {
        volatile ScoreT oldV, newV;
        do {// basically a fetch-and-add
          oldV = nghSum[d];
          newV = oldV + Delta[s].delta_over_degree;// Delta[s]/V[s].out_degree();
        } while (!gbbs::atomic_compare_and_swap(&nghSum[d], oldV, newV));
        return oldV == 0.0;
      }
      inline bool cond(uintE d) { return cond_true(d); }
    };

    template<class Graph, class E>
    void sparse_or_dense(Graph &G, E &EM, vertexSubset &Frontier,
                         delta_and_degree *Delta, ScoreT *nghSum, const flags fl) {
      using W = typename Graph::weight_type;

      if (Frontier.size() > G.n / 5) {
        Frontier.toDense();

        auto cond_f = [&](size_t i) { return true; };
        auto map_f = [&](const uintE &s, const uintE &d, const W &wgh) -> ScoreT {
          if (Frontier.d[d]) {
            return Delta[d].delta_over_degree;// Delta[d]/G.V[d].out_degree();
          } else {
            return static_cast<ScoreT>(0);
          }
        };
        auto reduce_f = [&](ScoreT l, ScoreT r) { return l + r; };
        auto apply_f = [&](std::tuple<uintE, ScoreT> k)
                -> std::optional<std::tuple<uintE, gbbs::empty>> {
          const uintE &u = std::get<0>(k);
          const ScoreT &contribution = std::get<1>(k);
          nghSum[u] = contribution;
          return std::nullopt;
        };
        ScoreT id = 0.0;

        flags dense_fl = fl;
        dense_fl ^= in_edges;// todo: check
        timer dt;
        dt.start();
        EM.template edgeMapReduce_dense<gbbs::empty, ScoreT>(
                Frontier, cond_f, map_f, reduce_f, apply_f, id, dense_fl | no_output);

        dt.stop();
        dt.next("dense time");
      } else {
        edgeMap(G, Frontier, PR_Delta_F<Graph>(G, Delta, nghSum), G.m / 2,
                no_output);
      }
    }

    template<class G>
    struct PR_Vertex_F_FirstRound {
      ScoreT damping, addedConstant, one_over_n, epsilon2;
      ScoreT *p;
      delta_and_degree *Delta;
      ScoreT *nghSum;
      G &get_degree;
      PR_Vertex_F_FirstRound(ScoreT *_p, delta_and_degree *_Delta, ScoreT *_nghSum,
                             ScoreT _damping, ScoreT _one_over_n, ScoreT _epsilon2,
                             G &get_degree)
          : damping(_damping),
            addedConstant((1 - _damping) * _one_over_n),
            one_over_n(_one_over_n),
            epsilon2(_epsilon2),
            p(_p),
            Delta(_Delta),
            nghSum(_nghSum),
            get_degree(get_degree) {}
      inline bool operator()(uintE i) {
        ScoreT pre_init = damping * nghSum[i] + addedConstant;
        p[i] += pre_init;
        ScoreT new_delta =
                pre_init - one_over_n;// subtract off delta from initialization
        Delta[i].delta = new_delta;
        Delta[i].delta_over_degree = new_delta / get_degree(i);
        return (new_delta > epsilon2 * p[i]);
      }
    };

    template<class G>
    auto make_PR_Vertex_F_FirstRound(ScoreT *p, delta_and_degree *delta,
                                     ScoreT *nghSum, ScoreT damping,
                                     ScoreT one_over_n, ScoreT epsilon2,
                                     G &get_degree) {
      return PR_Vertex_F_FirstRound<G>(p, delta, nghSum, damping, one_over_n,
                                       epsilon2, get_degree);
    }

    template<class G>
    struct PR_Vertex_F {
      ScoreT damping, epsilon2;
      ScoreT *p;
      delta_and_degree *Delta;
      ScoreT *nghSum;
      G &get_degree;
      PR_Vertex_F(ScoreT *_p, delta_and_degree *_Delta, ScoreT *_nghSum,
                  ScoreT _damping, ScoreT _epsilon2, G &get_degree)
          : damping(_damping),
            epsilon2(_epsilon2),
            p(_p),
            Delta(_Delta),
            nghSum(_nghSum),
            get_degree(get_degree) {}
      inline bool operator()(uintE i) {
        ScoreT new_delta = nghSum[i] * damping;
        Delta[i].delta = new_delta;
        Delta[i].delta_over_degree = new_delta / get_degree(i);

        if (fabs(Delta[i].delta) > epsilon2 * p[i]) {
          p[i] += new_delta;
          return 1;
        } else
          return 0;
      }
    };

    template<class G>
    auto make_PR_Vertex_F(ScoreT *p, delta_and_degree *delta, ScoreT *nghSum,
                          ScoreT damping, ScoreT epsilon2, G &get_degree) {
      return PR_Vertex_F<G>(p, delta, nghSum, damping, epsilon2, get_degree);
    }

    template<class Graph>
    sequence<ScoreT> PageRankDelta(Graph &G, ScoreT eps = 0.000001,
                                   ScoreT local_eps = 0.01,
                                   size_t max_iters = 100) {
      const long n = G.n;
      const ScoreT damping = 0.85;

      ScoreT one_over_n = 1 / (ScoreT) n;
      auto p = sequence<ScoreT>(n);
      auto Delta = sequence<delta_and_degree>(n);
      auto nghSum = sequence<ScoreT>(n);
      auto frontier = sequence<bool>(n);
      parallel_for(0, n, [&](size_t i) {
        uintE degree = G.get_vertex(i).out_degree();
        p[i] = 0.0;                 // one_over_n;
        Delta[i].delta = one_over_n;// initial delta propagation from each vertex
        Delta[i].delta_over_degree = one_over_n / degree;
        nghSum[i] = 0.0;
        frontier[i] = 1;
      });

      auto get_degree = [&](size_t i) { return G.get_vertex(i).out_degree(); };
      auto EM = EdgeMap<ScoreT, Graph>(G, std::make_tuple(UINT_E_MAX, (ScoreT) 0.0),
                                       (size_t) G.m / 1000);
      vertexSubset Frontier(n, n, std::move(frontier));
      auto all = sequence<bool>(n, true);
      vertexSubset All(n, n, std::move(all));// all vertices

      size_t round = 0;
      while (round++ < max_iters) {
        timer t;
        t.start();
        sparse_or_dense(G, EM, Frontier, Delta.begin(), nghSum.begin(), no_output);
        vertexSubset active =
                (round == 1)
                        ? vertexFilter(All, delta::make_PR_Vertex_F_FirstRound(
                                                    p.begin(), Delta.begin(), nghSum.begin(),
                                                    damping, one_over_n, local_eps, get_degree))
                        : vertexFilter(All, delta::make_PR_Vertex_F(
                                                    p.begin(), Delta.begin(), nghSum.begin(),
                                                    damping, local_eps, get_degree));

        // Check convergence: compute L1-norm between p_curr and p_next
        auto differences = parlay::delayed_seq<ScoreT>(
                n, [&](size_t i) { return fabs(Delta[i].delta); });
        ScoreT L1_norm = parlay::reduce(differences, parlay::plus<ScoreT>());
        if (L1_norm < eps) break;
        gbbs_debug(std::cout << "L1_norm = " << L1_norm << std::endl;);

        // Reset
        parallel_for(0, n, [&](size_t i) { nghSum[i] = static_cast<ScoreT>(0); });

        Frontier = std::move(active);
        gbbs_debug(t.stop(); t.next("iteration time"););
      }
      auto max_pr = parlay::reduce_max(p);
      std::cout << "max_pr = " << max_pr << std::endl;

      std::cout << "Num rounds = " << round << std::endl;
      return p;
    }
  }// namespace delta

}// namespace gbbs
