#include "kernels/topk.h"
#include "test_utils.hpp"

#include <algorithm>
#include <utility>

static void cpu_topk(const std::vector<float>& x, int k,
                     std::vector<float>& vals, std::vector<int>& idx) {
  std::vector<std::pair<float, int>> p(x.size());
  for (size_t i = 0; i < x.size(); ++i) p[i] = {x[i], (int)i};
  // Descending by value; indices are unique for the test inputs (no ties).
  std::sort(p.begin(), p.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  vals.resize(k);
  idx.resize(k);
  for (int i = 0; i < k; ++i) {
    vals[i] = p[i].first;
    idx[i] = p[i].second;
  }
}

void test_topk() {
  const int n = 4096, k = 8;
  const auto x = ktest::randu(n, -1000.0f, 1000.0f, 99 /* seed */);

  ktest::DeviceArray<float> dx(n), dvals(k);
  ktest::DeviceArray<int> didx(k);
  dx.upload(x);
  ck::topk_forward(dx.ptr, n, dvals.ptr, didx.ptr, k);
  const auto got_vals = dvals.download();
  const auto got_idx = didx.download();

  std::vector<float> ref_vals;
  std::vector<int> ref_idx;
  cpu_topk(x, k, ref_vals, ref_idx);

  for (int i = 0; i < k; ++i) {
    KT_CHECK(got_vals[i] == ref_vals[i]);  // exact: values are copied, not computed
    KT_CHECK(got_idx[i] == ref_idx[i]);
  }
  std::printf("  [OK] topk (k=%d, n=%d)\n", k, n);
}
