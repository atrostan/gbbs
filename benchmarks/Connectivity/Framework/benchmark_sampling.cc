#include "benchmarks/Connectivity/common.h"

#include "framework.h"
#include "sampling.h"
#include "utils.h"

#include "utils/benchmark.h"

void print_result(commandLine& P, std::string sampling_method, size_t rounds, double sampling_time, double pct_covered, size_t k=0) {
  std::cout << "{" << std::endl;
  std::cout << "  \"test_type\": \"sampling_result\"," << std::endl;
  std::cout << "  \"graph\" : \"" << P.getArgument(0) << "\"," << std::endl;
  std::cout << "  \"sampling_method\" : " << "\"" << sampling_method << "\"," << std::endl;
  std::cout << "  \"rounds\" : " << rounds << "," << std::endl;
  std::cout << "  \"sampling_time\" : " << sampling_time << "," << std::endl;
  std::cout << "  \"pct_covered\" : " << pct_covered << "," << std::endl;
  std::cout << "  \"k\" : " << k << std::endl;
  std::cout << "}" << std::endl;
}

template <class Graph>
void TestBFSSampling(Graph& G, commandLine& P) {
  long rounds = P.getOptionLongValue("-r", 5);
  using BFS = BFSSamplingTemplate<Graph>;
  auto sampler = BFS(G, P);
  std::vector<double> pcts;

  auto test = [&] () {
    timer t; t.start();
    auto parents = sampler.initial_components();
    double sampling_time = t.stop();

    auto [frequent_comp, pct] = sample_frequent_element(parents);
    pcts.emplace_back(pct);

    return sampling_time;
  };
  auto [mint, maxt, medt] = benchmark::run_multiple(rounds, test);
  auto medpct = benchmark::median(pcts);

  print_result(P, "bfs", rounds, medt, medpct);
}

template <class Graph>
void TestLDDSampling(Graph& G, commandLine& P) {
  long rounds = P.getOptionLongValue("-r", 5);
  using LDD = LDDSamplingTemplate<Graph>;
  auto sampler = LDD(G, P);
  std::vector<double> pcts;
  auto test = [&] () {
    timer t; t.start();
    auto parents = sampler.initial_components();
    double sampling_time = t.stop();

    auto [frequent_comp, pct] = sample_frequent_element(parents);
    pcts.emplace_back(pct);

    return sampling_time;
  };

  auto [mint, maxt, medt] = benchmark::run_multiple(rounds, test);
  auto medpct = benchmark::median(pcts);

  print_result(P, "ldd", rounds, medt, medpct);
}

template <class Graph>
double TestKOutSampling(Graph& G, commandLine& P) {
  long rounds = P.getOptionLongValue("-r", 5);
  size_t n = G.n;
  std::vector<double> pcts;
  auto test = [&] () {
    auto fc = find_variants::find_compress;
    auto unite = unite_variants::UniteND<decltype(fc)>(n, fc);
    using KOut = KOutSamplingTemplate<decltype(fc), decltype(unite), Graph>;
    auto sampler = KOut(G, fc, unite, P);

    timer t; t.start();
    auto parents = sampler.initial_components();
    double sampling_time = t.stop();

    auto [frequent_comp, pct] = sample_frequent_element(parents);
    pcts.emplace_back(pct);
    return sampling_time;
  };

  auto [mint, maxt, medt] = benchmark::run_multiple(rounds, test);
  auto medpct = benchmark::median(pcts);

  print_result(P, "kout-hybrid", rounds, medt, medpct);
}

template <class Graph>
double TestKOutAfforestSampling(Graph& G, commandLine& P) {
  long rounds = P.getOptionLongValue("-r", 5);
  size_t n = G.n;
  std::vector<double> pcts;
  auto test = [&] () {
    auto fc = find_variants::find_compress;
    auto unite = unite_variants::UniteND<decltype(fc)>(n, fc);
    using KOut = KOutSamplingTemplate<decltype(fc), decltype(unite), Graph>;
    auto sampler = KOut(G, fc, unite, P);

    timer t; t.start();
    auto parents = sampler.initial_components_afforest();
    double sampling_time = t.stop();

    auto [frequent_comp, pct] = sample_frequent_element(parents);
    pcts.emplace_back(pct);
    return sampling_time;
  };

  auto [mint, maxt, medt] = benchmark::run_multiple(rounds, test);
  auto medpct = benchmark::median(pcts);

  print_result(P, "kout-afforest", rounds, medt, medpct);
}

template <class Graph>
double TestKOutPureSampling(Graph& G, commandLine& P) {
  long rounds = P.getOptionLongValue("-r", 5);
  size_t n = G.n;
  std::vector<double> pcts;
  auto test = [&] () {
    auto fc = find_variants::find_compress;
    auto unite = unite_variants::UniteND<decltype(fc)>(n, fc);
    using KOut = KOutSamplingTemplate<decltype(fc), decltype(unite), Graph>;
    auto sampler = KOut(G, fc, unite, P);

    timer t; t.start();
    auto parents = sampler.initial_components_pure();
    double sampling_time = t.stop();

    auto [frequent_comp, pct] = sample_frequent_element(parents);
    pcts.emplace_back(pct);
    return sampling_time;
  };

  auto [mint, maxt, medt] = benchmark::run_multiple(rounds, test);
  auto medpct = benchmark::median(pcts);

  print_result(P, "kout", rounds, medt, medpct);
}

template <class Graph>
double Sampler(Graph& G, commandLine& P) {
  std::cout << "[" << std::endl;
  TestBFSSampling<Graph>(G, P);
  std::cout << "," << endl;
  TestLDDSampling<Graph>(G, P);
  std::cout << "," << endl;
  TestKOutSampling<Graph>(G, P);
  std::cout << "," << endl;
  TestKOutAfforestSampling<Graph>(G, P);
  std::cout << "," << endl;
  TestKOutPureSampling<Graph>(G, P);
  std::cout << "]" << std::endl;
}

generate_symmetric_once_main(Sampler, false);
