#pragma once
#include "gemm_tmma.h"

// tcgen5 templates depend on stcuv2-only cccl headers
// (cccl/tang/__ptx/instructions/tc_mma.h ...), which are not available to the
// stcu (v1) ptcc toolchain. Only pull them in for stcuv2 builds, where the
// compiler defines TANG_STCUV2. This keeps stcu kernels that don't use these
// GEMM paths from failing on the missing headers.
//
// gemm_wgmma.h was removed (WGMMA not supported); gemm_tcgen5.h bodies are
// currently stubbed as TODO.
#ifdef TANG_STCUV2
#include "gemm_tcgen5.h"
#endif
