/*!
 * \file verify_s3_shared_swizzle.cc
 * \brief Reject bulk copies of one shared buffer that disagree on its swizzle.
 *
 * Background
 * ----------
 * A shared buffer's swizzle is a property of the *buffer*, but
 * ``tang_swizzle_atom_bytes`` hangs off a *copy statement*
 * (``CopyNode::annotations``). ``LowerSTCUV2BulkCopy`` (src/tang/op/copy.cc)
 * picks the regime from two local signals only -- whether *this* copy carries
 * the annotation, and whether the buffer is a GEMM operand -- so two copies of
 * the same buffer can pick different regimes:
 *
 *   T.copy(In, S, annotations={"tang_swizzle_atom_bytes": 32})  # swizzled 3D
 *   T.copy(S, Out)                                              # unswizzled 1D
 *
 * The write then permutes the data and the read does not, silently scrambling
 * the buffer.
 *
 * Making the annotation drive layout inference instead -- so that one buffer
 * can only ever have one layout and both copies necessarily agree -- needs a
 * C++ constructor for the TANG atom8/32/64 layouts, which are currently
 * generated in Python. Until that exists, this pass reports the disagreement
 * and names both use sites.
 *
 * What is checked / not checked
 * -----------------------------
 * Checked: every *bulk* copy (global<->shared, i.e. the ones
 * ``LowerSTCUV2BulkCopy`` handles) of one shared buffer must agree on
 * (annotation present, atom bytes).
 *
 * Not checked: copies on the other paths (tmem<->shared, fragment<->shared) do
 * not go through that decision at all, so they are ignored -- the validated
 * cpt2s drain annotates only its shared->global leg on purpose.
 *
 * This pass does not compare an annotation with a GEMM-assigned operand
 * layout; that requires layout-map information.
 */
#include "op/copy.h"
#include "op/operator.h"
#include "support/check.h"
#include "tang/target_utils.h"

#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
using tvm::transform::PassContext;

namespace {

/*! \brief The annotation that states a bulk copy's swizzle atom. */
constexpr const char *kAtomAnnotation = "tang_swizzle_atom_bytes";

/*! \brief One bulk copy of a shared buffer, as the check sees it. */
struct SwizzleUse {
  bool is_store;  // shared -> global
  bool annotated; // carries tang_swizzle_atom_bytes
  int atom_bytes; // only meaningful when annotated
  Span span;
};

bool IsSharedScope(const Buffer &buffer) {
  return buffer.scope() == "shared" || buffer.scope() == "shared.dyn";
}

/*!
 * \brief Whether this copy leaves the bulk path, mirroring
 * ``IsTangBulkCopyDisabled`` in src/tang/op/copy.cc. Copies that do not reach
 * ``LowerSTCUV2BulkCopy`` never consult the annotation, so they must not be
 * part of the comparison.
 */
bool IsBulkCopyDisabled(const CopyNode &copy) {
  bool disabled = PassContext::Current()
                      ->GetConfig<Bool>("tl.disable_tma_lower", Bool(false))
                      .value();
  if (auto value = copy.annotations.Get("disable_tma")) {
    if (const auto *flag = value->as<IntImmNode>()) {
      disabled |= flag->value != 0;
    }
  }
  return disabled;
}

std::string Where(const Span &span) {
  if (!span.defined()) {
    return "<unknown location>";
  }
  std::ostringstream os;
  os << span->source_name << ":" << span->line << ":" << span->column;
  return os.str();
}

std::string Describe(const SwizzleUse &use) {
  std::string text =
      use.is_store ? "shared->global store" : "global->shared load";
  text += use.annotated ? ", annotated tang_swizzle_atom_bytes = " +
                              std::to_string(use.atom_bytes) + " (swizzled)"
                        : ", no swizzle annotation (unswizzled 1D)";
  return text + " at " + Where(use.span);
}

class SharedSwizzleChecker : public StmtExprVisitor {
public:
  void VisitExpr_(const CallNode *op) final {
    if (Optional<TileOperator> tile_op = ParseOperator(GetRef<Call>(op));
        tile_op.defined()) {
      const auto *copy = tile_op.value().as<CopyNode>();
      if (copy != nullptr) {
        Record(*copy, op->span);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  /*! \brief Report every disagreement found while visiting. */
  void Report() const {
    // ``order_`` rather than ``uses_``: an unordered_map over buffer *pointers*
    // reaches whichever conflicting buffer its hash happens to hit first, and
    // that changes from run to run (ASLR), so the same source could name a
    // different buffer each time it is compiled. First-seen order -- the order
    // the IR is visited in -- is stable, and keeps the first copy of each
    // buffer as the reference the others are compared against.
    for (const BufferNode *buffer : order_) {
      const std::vector<SwizzleUse> &uses = uses_.at(buffer);
      if (uses.size() < 2) {
        continue;
      }
      const SwizzleUse &first = uses.front();
      for (size_t i = 1; i < uses.size(); ++i) {
        const SwizzleUse &other = uses[i];
        const bool agrees =
            first.annotated == other.annotated &&
            (!first.annotated || first.atom_bytes == other.atom_bytes);
        ICHECK(agrees)
            << "TANG STCUV2: two bulk copies of shared buffer '" << buffer->name
            << "' disagree about its swizzle.\n"
            << "  - " << Describe(first) << "\n"
            << "  - " << Describe(other) << "\n"
            << "A swizzle is a property of the buffer, but "
               "`tang_swizzle_atom_bytes` is attached to a copy statement and "
               "lowering chooses the regime from that statement alone "
               "(src/tang/op/copy.cc). Writing swizzled and reading unswizzled "
               "scrambles the data without any error, so the two copies have "
               "to "
               "agree.\n"
            << "  Annotate every bulk copy of this buffer with the same atom "
               "size, or annotate none of them and use a buffer that is not a "
               "GEMM operand (then both legs take the unswizzled 1D path).";
      }
    }
  }

private:
  void Record(const CopyNode &copy, const Span &span) {
    if (IsBulkCopyDisabled(copy)) {
      return;
    }
    const bool g2s = copy.src.scope() == "global" && IsSharedScope(copy.dst);
    const bool s2g = IsSharedScope(copy.src) && copy.dst.scope() == "global";
    if (!g2s && !s2g) {
      return;
    }
    const Buffer &shared = g2s ? copy.dst : copy.src;
    SwizzleUse use;
    use.is_store = s2g;
    use.annotated = false;
    use.atom_bytes = 0;
    if (auto value = copy.annotations.Get(kAtomAnnotation)) {
      const auto *imm = value->as<IntImmNode>();
      ICHECK(imm) << "TANG STCUV2: tang_swizzle_atom_bytes must be an IntImm";
      use.annotated = true;
      use.atom_bytes = static_cast<int>(imm->value);
    }
    use.span = span;
    std::vector<SwizzleUse> &uses = uses_[shared.get()];
    if (uses.empty()) {
      order_.push_back(shared.get());
    }
    uses.push_back(use);
  }

  std::unordered_map<const BufferNode *, std::vector<SwizzleUse>> uses_;
  /*! \brief The buffers in first-seen order, so Report() is deterministic. */
  std::vector<const BufferNode *> order_;
};

} // namespace

/*!
 * \brief Check that the bulk copies of a shared buffer agree on its swizzle.
 *
 * A pure verification: the module is returned unchanged, and the pass is a
 * no-op on other targets.
 */
tvm::transform::Pass VerifySharedSwizzle() {
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    Optional<Target> opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!opt_target.defined() || !TargetTangIsSTCUV2(opt_target.value())) {
      return f;
    }
    SharedSwizzleChecker checker;
    checker(f->body);
    checker.Report();
    return f;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.tang.VerifySharedSwizzle", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.tang.transform.VerifySharedSwizzle",
                        VerifySharedSwizzle);
}

} // namespace tl
} // namespace tvm
