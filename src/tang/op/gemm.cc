/*!
 * \file tl/tang/op/gemm.cc
 * \brief TANG implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"
#include "op/utils.h"
#include "support/check.h"
#include "tang/target_utils.h"

#include <cmath>
#include <limits>

namespace tvm {
namespace tl {
namespace tang {
namespace {

constexpr const char *kTangTMMA = "tang.tmma";
constexpr const char *kTangWGMMA = "tang.wgmma";
constexpr const char *kTangTCGEN5 = "tang.tcgen5";

std::pair<int, int> ComputeWarpPartition(const GemmWarpPolicyNode &policy,
                                         int M, int N, int block_size,
                                         Target target, String gemm_inst) {
  ICHECK(TargetIsTang(target));
  constexpr int kWarpSize = 32;
  constexpr int kMPerWarp = 8;
  constexpr int kNPerWarp = 8;
  int num_warps = block_size / kWarpSize;
  int m_warp = 1;
  int n_warp = 1;

  if (gemm_inst == kTangTCGEN5) {
    policy.m_warp = 1;
    policy.n_warp = num_warps;
    return {1, num_warps};
  }
  if (gemm_inst == kTangWGMMA) {
    ICHECK_EQ(num_warps % 4, 0)
        << "TANG stcuv2 WGMMA requires a whole number of four-warp groups";
  } else {
    ICHECK_EQ(gemm_inst, kTangTMMA);
  }

  ICHECK_EQ(block_size % kWarpSize, 0)
      << "TANG GEMM requires a whole number of 32-thread warps";
  ICHECK_EQ(M % kMPerWarp, 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK_EQ(N % kNPerWarp, 0)
      << "N must be divisible by " << kNPerWarp << ", but got " << N;

  if (policy.IsFree()) {
    m_warp = policy.m_warp;
    n_warp = policy.n_warp;
  } else if (policy.IsFullRow()) {
    m_warp = num_warps;
    if (M % (m_warp * kMPerWarp) != 0) {
      m_warp = M / kMPerWarp;
      n_warp = num_warps / m_warp;
    }
  } else if (policy.IsFullCol()) {
    n_warp = num_warps;
    if (N % (n_warp * kNPerWarp) != 0) {
      n_warp = N / kNPerWarp;
      m_warp = num_warps / n_warp;
    }
  } else if (policy.IsSquare()) {
    int max_m_warps = M / kMPerWarp;
    float ideal_ratio = N > 0 ? static_cast<float>(M) / N : 1.0f;
    float best_balance = std::numeric_limits<float>::max();
    for (int m = 1; m <= max_m_warps && m <= num_warps; ++m) {
      int n = num_warps / m;
      if (m * n != num_warps)
        continue;
      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * kNPerWarp);
      if (m_per_warp < 1 || n_per_warp < 1)
        continue;
      float balance = std::abs(m_per_warp / n_per_warp - ideal_ratio);
      if (balance < best_balance) {
        best_balance = balance;
        m_warp = m;
        n_warp = n;
      }
    }
  } else {
    ICHECK(false) << "Unknown TANG GemmWarpPolicy";
  }

  ICHECK_GT(m_warp, 0);
  ICHECK_GT(n_warp, 0);
  ICHECK_EQ(m_warp * n_warp, num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp=" << m_warp
      << ", n_warp=" << n_warp << ", num_warps=" << num_warps;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

String SelectInst(const GemmNode &op, int block_size, Target target) {
  ICHECK(TargetIsTang(target));
  if (TargetTangIsSTCUV2(target)) {
    if (op.c_.scope() == "shared.tmem") {
      return kTangTCGEN5;
    }
    if (op.c_.scope() == "shared" || op.c_.scope() == "shared.dyn") {
      return kTangWGMMA;
    }
  }
  // A must be shared for the same reason B must: GemmTensorOp::body() takes
  // both as shared bases and issues its own shared->register loads (LoadA /
  // LoadB). There is no register-source (RS) TMMA variant, and gemm_tmma.py
  // only registers a layout for A when its scope is shared, so an A in
  // local.fragment would otherwise lower to a template that reinterprets
  // register addresses as shared ones and silently produce NaN/inf.
  ICHECK(IsSharedBuffer(op.a_))
      << "TANG TMMA requires A in shared memory, but got " << op.a_.scope();
  ICHECK(IsSharedBuffer(op.b_))
      << "TANG TMMA requires B in shared memory, but got " << op.b_.scope();
  ICHECK(op.c_.scope() == "local.fragment")
      << "TANG TMMA requires C in local.fragment, but got " << op.c_.scope();
  return kTangTMMA;
}

bool ReuseExistingSharedLayout(String gemm_inst) {
  ICHECK(gemm_inst == kTangTMMA || gemm_inst == kTangWGMMA ||
         gemm_inst == kTangTCGEN5);
  return false;
}

bool RegisterTangGemm() {
  RegisterGemmImpl(GemmImpl{
      "tang.Gemm",
      TargetIsTang,
      SelectInst,
      ComputeWarpPartition,
      ReuseExistingSharedLayout,
  });
  return true;
}

const bool tang_gemm_registered = RegisterTangGemm();

} // namespace
} // namespace tang
} // namespace tl
} // namespace tvm
