#ifndef _KINTERSECT_
#define _KINTERSECT_

#pragma once

#include <math.h>

#include "bucket.h"
#include "edge_map_reduce.h"
#include "ligra.h"
#include "pbbslib/dyn_arr.h"
#include "simdinter/include/intersection.h"

struct nop_f {
void operator()(pbbs::sequence<uintE> s) const {
    return;
  }
};

struct lstintersect_par_struct {
  template <template <typename W> class vertex, class W, class S>
  std::tuple<sequence<uintE>, size_t> operator()(graph<vertex<W>>& DG, uintE vtx, S induced, bool save = true) const {
    return lstintersect_par(DG, vtx, induced, save);
  }
};

template <template <typename W> class vertex, class W, class S>
inline std::tuple<sequence<uintE>, size_t> lstintersect_par(graph<vertex<W>>& DG, uintE vtx, S induced, bool save = true) {
  auto vtx_seq = pbbslib::make_sequence<uintE>((uintE*)(DG.V[vtx].getOutNeighbors()), DG.V[vtx].getOutDegree());
  size_t index = 0;
  if (!save) {
    auto merge_f = [&] (uintE ngh) {};
    return std::make_tuple(pbbs::sequence<uintE>(), intersection::merge(vtx_seq, induced, merge_f));
  }

  auto out = sequence<uintE>::no_init(std::min(induced.size(), vtx_seq.size()));
  auto merge_f = [&] (uintE ngh) {
    out[index] = ngh;
    index++;
  };
  intersection::merge(vtx_seq, induced, merge_f);
  out.shrink(index);
  return std::make_tuple(out, out.size());
  // want to intersect V[base[i]] outneighbors
  // figure out which one represents the smallest
  // intersect each against the smallest
  // any element in the smallest that has num-1 marks is in our merged list
}

struct lstintersect_set_struct {
  template <template <typename W> class vertex, class W, class S>
  std::tuple<std::vector<uintE>, size_t> operator()(graph<vertex<W>>& DG, uintE vtx, S induced, bool save = true) const {
    return lstintersect_set(DG, vtx, induced, save);
  }
};

// make sure set intersection is stable
template <template <typename W> class vertex, class W, class S>
inline std::tuple<std::vector<uintE>, size_t> lstintersect_set(graph<vertex<W>>& DG, uintE vtx, S induced, bool save = true) {
  auto vtx_seq = pbbslib::make_sequence<uintE>((uintE*)(DG.V[vtx].getOutNeighbors()), DG.V[vtx].getOutDegree());
  std::vector<uintE> out;
  std::set_intersection(induced.begin(), induced.end(), vtx_seq.begin(), vtx_seq.end(), std::back_inserter(out));
  return std::make_tuple(out, out.size());
}

struct lstintersect_vec_struct {
  template <template <typename W> class vertex, class W, class S>
  std::tuple<sequence<uintE>, size_t> operator()(graph<vertex<W>>& DG, uintE vtx, S induced, bool save = true) const {
    return lstintersect_vec(DG, vtx, induced, save);
  }
};

// TODO radix sort in place
template <template <typename W> class vertex, class W, class S>
inline std::tuple<sequence<uintE>, size_t> lstintersect_vec(graph<vertex<W>>& DG, uintE vtx, S induced, bool save = true) {
  auto vtx_seq = pbbslib::make_sequence<uintE>((uintE*)(DG.V[vtx].getOutNeighbors()), DG.V[vtx].getOutDegree());
  auto out = sequence<uintE>::no_init(std::min(induced.size(), vtx_seq.size()));
  SIMDCompressionLib::intersectionfunction inter = SIMDCompressionLib::IntersectionFactory::getFromName("simd");
  size_t out_size = inter(induced.begin(), induced.size(), vtx_seq.begin(), vtx_seq.size(), out.begin());
  if (!save) {
    out.~sequence();
    return std::make_tuple(pbbs::sequence<uintE>(), out_size);
  }
  out.shrink(out_size);
  return std::make_tuple(out, out_size);
}


template <template <typename W> class vertex, class W>
inline sequence<uintE> kintersect(graph<vertex<W>>& DG, sequence<uintE> base, size_t num) {
  if (num == 1) {
    uintT deg = DG.V[base[0]].getOutDegree();
    uintE* ngh = (uintE*)(DG.V[base[0]].getOutNeighbors());
    return pbbslib::make_sequence<uintE>(ngh, deg);
  }

  // find base index with min outdeg
  auto base_idxs = sequence<size_t>::no_init(num);
  parallel_for (0,num,[&] (size_t i) { base_idxs[i] = i; });
  auto base_deg_f = [&](size_t i, size_t j) -> size_t {
    return DG.V[base[i]].getOutDegree() < DG.V[base[j]].getOutDegree() ? i : j;
  };
  size_t min_base = pbbslib::reduce(base_idxs, pbbslib::make_monoid(base_deg_f, 0));
  size_t min_deg = DG.V[base[min_base]].getOutDegree();
  auto min_seq = pbbslib::make_sequence<uintE>((uintE*)(DG.V[base[min_base]].getOutNeighbors()), min_deg);

  // set up array to mark where min seq intersects with other outneighbors in base
  auto marks = sequence<bool>(min_deg * (num - 1), false);
  // mark intersections
  parallel_for (0,num,[&] (size_t i) {
    if (i != min_base) {
    size_t idx = i > min_base ? i - 1 : i;
    auto merge_f = [&] (uintE ngh) {
      size_t j = pbbslib::binary_search(min_seq, ngh, std::less<size_t>());
      marks[min_deg * idx + j] = true;
    };
    auto a_seq = pbbslib::make_sequence<uintE>((uintE*)(DG.V[base[i]].getOutNeighbors()), DG.V[base[i]].getOutDegree());
    intersection::merge(min_seq, a_seq, merge_f);
    }
  });

  // merge marks to determine which intersects all (can make more efficient here)
  auto merge_seq = sequence<bool>(min_deg, true);
  parallel_for (0,min_deg,[&] (size_t i) {
    for (size_t j=0; j < num - 1; ++j) {
      merge_seq[i] = merge_seq[i] && marks[min_deg * j + i];
    }
  });
  // pack out intersected vertices
  auto out = pbbslib::pack(min_seq, merge_seq);
  return out;
  // want to intersect V[base[i]] outneighbors
  // figure out which one represents the smallest
  // intersect each against the smallest
  // any element in the smallest that has num-1 marks is in our merged list

}

#endif