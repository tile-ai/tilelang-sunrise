/*!
 * \file target/codegen.cc
 */

#include "codegen_tang.h"
#include "backend/common/codegen/codegen_utils.h"
#include "support/check.h"
#include "tang/target_utils.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/cast.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/index_map.h>
#include <tvm/tirx/op.h>

#include <cassert>
#include <cmath>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "arith/pattern_match.h"
#include "op/builtin.h"
#include "transform/common/attr.h"

namespace tvm {
namespace codegen {
using namespace ffi;

struct TANGMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      case 16: {
        if (name == "fabs") {
          return "__habs";
        } else if (name == "round") {
          return "hrint";
        } else {
          return "h" + name;
        }
      }
      default:
        return "";
      }
    } else if (t.is_bfloat16()) {
      if (name == "fabs") {
        return "__habs";
      } else if (name == "round") {
        return "hrint";
      } else {
        return "h" + name;
      }
    } else if (t.is_int() || t.is_uint()) {
      switch (t.bits()) {
      case 32:
        return "__" + name;
      case 64:
        return "__" + name + "ll";
      default:
        return "";
      }
    }
    return "";
  }
};

struct TANGFastMath : public TANGMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float() && t.bits() == 32) {
      return "__" + name + 'f';
    } else {
      return TANGMath::operator()(t, name);
    }
    return "";
  }
};

struct TANGFastMathTan : public TANGMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
      case 64:
        return name;
      // `__tanf` seems to produce some values too deviant from numpy tan
      // version. So, let's use just `tanf` instead.
      case 32:
        return name + 'f';
      case 16:
        return 'h' + name;
      default:
        return "";
      }
    }
    return "";
  }
};

struct TANGIEEEMath {
  std::string operator()(DataType t, std::string name,
                         std::string rounding_mode) const {
    if (t.is_float() && t.bits() == 32) {
      return "__" + name + "_" + rounding_mode;
    } else if (t.is_float() && t.bits() == 64) {
      return "__d" + name + "_" + rounding_mode;
    }
    return "";
  }
};

static std::optional<std::string>
get_stcu_s2_atomic_func_name(const CallNode *op) {
  if (!op->op.same_as(builtin::call_pure_extern()) &&
      !op->op.same_as(builtin::call_extern())) {
    return std::nullopt;
  }
  std::string func_name;
  if (const auto *str_imm = op->args[0].as<StringImmNode>()) {
    func_name = str_imm->value;
  } else if (const auto *global_var = op->args[0].as<GlobalVarNode>()) {
    func_name = global_var->name_hint;
  } else {
    return std::nullopt; // not a valid function name
  }

  static const std::unordered_set<std::string> atomic_func_set = {
      "atomicAdd", "atomicAddUint", "atomicSub",  "atomicSubUint",
      "atomicMax", "atomicMaxUint", "atomicExch", "atomicExchUint",
      "atomicInc", "atomicIncUint", "atomicDec",  "atomicDecUint",
      "atomicCAS", "atomicCASUint", "atomicMin",  "atomicMinUint",
      "atomicXor", "atomicXorUint", "atomicOr",   "atomicOrUint",
      "atomicAnd", "atomicAndUint"};

  if (atomic_func_set.count(func_name)) {
    return func_name;
  } else {
    return std::nullopt;
  }
}

static std::optional<DataType>
GetAtomicAccessPtrElementType(const PrimExpr &expr) {
  const auto *ptr_call = expr.as<CallNode>();
  if (ptr_call == nullptr) {
    return std::nullopt;
  }
  if (ptr_call->op.same_as(builtin::address_of())) {
    const auto *buffer_load = ptr_call->args[0].as<BufferLoadNode>();
    ICHECK(buffer_load) << "address_of arg must be BufferLoad";
    return buffer_load->buffer->dtype;
  }
  if (ptr_call->op.same_as(builtin::tvm_access_ptr())) {
    ICHECK(!ptr_call->args.empty());
    return ptr_call->args[0].dtype();
  }
  if (ptr_call->op.same_as(tl::access_ptr())) {
    ICHECK_EQ(ptr_call->args.size(), 3U)
        << "tl.access_ptr expects 3 args: (BufferLoad, extent, rw_mask)";
    const auto *buffer_load = ptr_call->args[0].as<BufferLoadNode>();
    ICHECK(buffer_load) << "tl.access_ptr arg0 must be BufferLoad";
    return buffer_load->buffer->dtype;
  }
  return std::nullopt;
}

static std::string GetFP8Type(DataType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.is_scalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "_2";
  } else if (lanes == 4) {
    vec = "_4";
  } else if (lanes == 8) {
    vec = "_8";
  } else if (lanes == 16) {
    vec = "_16";
  } else if (lanes == 32) {
    vec = "_32";
  } else {
    LOG(FATAL)
        << "Only support scalar and vector types of width (2, 4, 8, 16, 32) "
           "for FP8";
  }
  if (type.is_float8_e4m3fn() || type.is_float8_e4m3fnuz() ||
      type.is_float8_e4m3()) {
    stream << "fp8_e4" << vec << "_t";
  } else if (type.is_float8_e5m2() || type.is_float8_e5m2fnuz() ||
             type.is_float8_e5m2()) {
    stream << "fp8_e5" << vec << "_t";
  } else {
    LOG(FATAL) << "Unsupported FP8 type in TANG codegen but got " << type;
  }
  return stream.str();
}

static std::string GetFP6Type(DataType type) {
  // TANG has no usable 6-bit scalar type. fp6 tensor-core operands are
  // bit-packed (128 elements -> 128*6/8 = 96 bytes) and the shared/global rows
  // are padded to a swizzle-valid byte stride; the buffer is handled purely as
  // a byte stream (the fp6-ness lives only in the mma descriptor's EleType).
  // Represent fp6 by a 1-byte container, mirroring the fp4 (uchar) treatment,
  // so the generated code type-checks under ptcc (which rejects the CUDA
  // ``__nv_fp6_*`` names).
  if (type.code() != DataType::kFloat6_e2m3fn &&
      type.code() != DataType::kFloat6_e3m2fn) {
    LOG(FATAL) << "Unsupported FP6 type in TANG codegen";
  }
  std::stringstream stream;
  int32_t lanes = type.lanes();
  if (type.is_scalar()) {
    stream << "uchar";
  } else if (lanes <= 16) {
    stream << "uchar" << lanes;
  } else {
    LOG(FATAL) << "Only support scalar and vector (<=16) fp6 in TANG codegen";
  }
  return stream.str();
}

static std::string GetFP4Type(DataType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.is_scalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "x2";
  } else if (lanes == 4) {
    vec = "x4";
  } else if (lanes == 8) {
    vec = "x8";
  } else if (lanes == 16) {
    vec = "x16";
  } else {
    LOG(FATAL)
        << "Only support scalar and vector types of width (2, 4) for FP4";
  }
  if (type.code() != DataType::kFloat4_e2m1fn) {
    LOG(FATAL) << "Unsupported FP4 type in TANG codegen";
  }
  // TANG has no usable 4-bit scalar type, and fp4 tensor-core operands are
  // physically packed 2-per-byte. Represent an fp4 element by a 1-byte
  // container (two fp4 share one ``uchar``); the buffer-index logic divides
  // element indices by 2 (see GetBufferRef). Vector lanes map to the wider
  // containers.
  if (type.is_scalar()) {
    stream << "uchar";
  } else if (lanes == 2) {
    stream << "uchar"; // fp4x2 -> 1 byte
  } else if (lanes == 4) {
    stream << "ushort"; // fp4x4 -> 2 bytes
  } else {
    stream << "uchar" << vec;
  }
  return stream.str();
}

CodeGenTileLangTANG::CodeGenTileLangTANG() {
  restrict_keyword_ = "__restrict__";
  vid_global_barrier_state_ =
      name_supply_->FreshName("__tvm_global_barrier_state");
  vid_global_barrier_expect_ = name_supply_->FreshName("__barrier_expect");
  ICHECK_EQ(vid_global_barrier_state_, "__tvm_global_barrier_state");
}

void CodeGenTileLangTANG::PrintFuncPrefix(std::ostream &os) {
  bool use_cache3 = tvm::transform::PassContext::Current()
                        ->GetConfig<Bool>(tvm::tl::kUseAsyncCop4, Bool(false))
                        .value()
                        ->value;
  os << "extern \"C\" __global__";
  if (use_cache3) {
    os << " __cache3__";
  }
  os << " ";
}

/*!
 * \brief Assert that a pts_{load,store}_async call respects the operand
 * order contract: args[0]=shared, args[1]=global, args[2]=bytes.
 *
 * This is NOT {dst, src} — shared/global is the only description that
 * holds for both load (global→shared) and store (shared→global).
 *
 * The glb_base/subtrahend resolution unconditionally reads args[1] as
 * the global operand; a swap resolves the shared buffer instead and
 * emits a wrong global address with no compile error, just wrong data.
 * Arity checks cannot catch this ({shared,global,bytes} and {global,
 * shared,bytes} both have 3 args), so we assert on storage scopes.
 *
 * \note Uses LOG(FATAL) so the check survives -DNDEBUG release builds.
 */
static void CheckAsyncCopyOperandOrder(const tirx::CallNode *call,
                                       const char *op_name) {
  auto scope_of = [](const PrimExpr &arg) -> std::string {
    // Operands are built by AddressOffset() as address_of(BufferLoad(dummy)),
    // where the dummy buffer wraps the original handle Var, so the pointer
    // type annotation (and thus the storage scope) is preserved.
    const auto *c = arg.as<tirx::CallNode>();
    if (!c || !c->op.same_as(tirx::builtin::address_of()) ||
        c->args.size() != 1) {
      return "";
    }
    const auto *load = c->args[0].as<tirx::BufferLoadNode>();
    if (!load)
      return "";
    const auto *ptr = load->buffer->data->type_annotation.as<PointerTypeNode>();
    return ptr ? std::string(ptr->storage_scope) : std::string("");
  };

  const std::string shared_scope = scope_of(call->args[0]);
  const std::string global_scope = scope_of(call->args[1]);
  const bool arg0_is_shared =
      (shared_scope == "shared" || shared_scope == "shared.dyn");
  // An unannotated scope prints as "", which is how plain global pointers can
  // appear; only reject a positively-identified shared buffer in args[1].
  const bool arg1_is_shared =
      (global_scope == "shared" || global_scope == "shared.dyn");

  if (!arg0_is_shared || arg1_is_shared) {
    LOG(FATAL) << op_name
               << ": operand order contract violated. Expected {shared, global,"
                  " bytes} (args[0] on a shared buffer, args[1] on the global"
                  " one), but got args[0] scope=\""
               << shared_scope << "\", args[1] scope=\"" << global_scope
               << "\". The glb_base resolution below reads args[1] as the"
                  " global operand; a swapped order silently emits a wrong"
                  " global address. Fix the injection site in"
                  " src/tang/transform/inject_pts_async_copy.cc.";
  }
}

/**
 * \brief Detect function parameters that must not be const-qualified.
 *
 * AnnotateReadOnlyParams marks parameters const when they are never loaded
 * in the TIR body. This pass catches the blind spots -- parameters written
 * through paths invisible to read-only analysis:
 *   1. async DMA  -- the destination operand is written by the hardware copy
 *   2. atomicAdd   -- atomic read-modify-write requires a non-const pointer
 *
 * Only the *written* operand is excluded from const; the source operand of
 * an async copy is read-only and correctly receives const.
 *
 * \par Operand-order contract (kept in sync with inject_pts_async_copy.cc)
 *   pts_load_async  (global -> shared):  args = {dst, src, bytes}  -> dst is
 * args[0] pts_store_async (shared -> global):  args = {src, dst, bytes}  -> dst
 * is args[1]
 *
 *   WARNING: reversing the per-builtin dst index silently const-qualifies a
 *   written buffer. ptcc accepts const on these builtins, so the defect
 *   produces wrong results with no compile error.
 */
class AsyncLoadParamCollector {
private:
  std::unordered_set<const tirx::VarNode *> params_set; // Function parameters
  std::unordered_set<const tirx::VarNode *> buffer_data_set; // Buffer data vars

public:
  explicit AsyncLoadParamCollector(
      const std::unordered_set<const tirx::VarNode *> &params,
      const std::unordered_set<const tirx::VarNode *> &buffer_data)
      : params_set(params), buffer_data_set(buffer_data) {}

  void Collect(const Stmt &stmt) {
    PostOrderVisit(stmt, [this](const ObjectRef &node) {
      const auto *call = node.as<tirx::CallNode>();
      if (!call)
        return;
      if (get_stcu_s2_atomic_func_name(call).has_value() &&
          call->args.size() > 1) {
        // call_extern layout: (name, address, operands...)
        MarkBufferArgAsAtomicAdd(call->args[1]);
      } else if (call->op.same_as(tl::pts_load_async())) {
        // global -> shared: args = {dst, src, bytes}. Only dst is written.
        // Contract with inject_pts_async_copy.cc: 3 non-predicated args.
        ICHECK_GE(call->args.size(), 3)
            << "pts_load_async: expected at least 3 args {dst, src, bytes}, "
            << "got " << call->args.size();
        MarkBufferArgAsAsyncLoad(call->args[0]);
        VisitExpr(call->args[0]);
      } else if (call->op.same_as(tl::pts_store_async())) {
        // shared -> global: args = {src, dst, bytes} -- note the order is the
        // reverse of pts_load_async. dst is the global buffer being written, so
        // it is the one that must stay non-const.
        // Contract with inject_pts_async_copy.cc: 3 non-predicated args.
        ICHECK_GE(call->args.size(), 3)
            << "pts_store_async: expected at least 3 args {src, dst, bytes}, "
            << "got " << call->args.size();
        MarkBufferArgAsAsyncLoad(call->args[1]);
        VisitExpr(call->args[1]);
      } else if ((call->op.same_as(tl::atomic_add_elem_op()) ||
                  call->op.same_as(tl::atomic_add_ret_elem_op()) ||
                  call->op.same_as(tl::atomic_addx2_elem_op()) ||
                  call->op.same_as(tl::atomic_addx4_elem_op()) ||
                  call->op.same_as(tl::atomic_max_elem_op()) ||
                  call->op.same_as(tl::atomic_max_ret_elem_op()) ||
                  call->op.same_as(tl::atomic_min_elem_op()) ||
                  call->op.same_as(tl::atomic_min_ret_elem_op()) ||
                  call->op.same_as(tl::atomic_or_elem_op()) ||
                  call->op.same_as(tl::atomic_sub_elem_op()) ||
                  call->op.same_as(tl::atomic_exch_elem_op()) ||
                  call->op.same_as(tl::atomic_inc_elem_op()) ||
                  call->op.same_as(tl::atomic_dec_elem_op()) ||
                  call->op.same_as(tl::atomic_cas_elem_op()) ||
                  call->op.same_as(tl::atomic_and_elem_op()) ||
                  call->op.same_as(tl::atomic_xor_elem_op()) ||
                  call->op.same_as(tl::atomic_store_elem_op())) &&
                 !call->args.empty()) {
        MarkBufferArgAsAtomicAdd(call->args[0]);
      }
    });
  }

private:
  // Detect atomicAdd from tir.call_extern("atomicAdd", ptr, ...)
  void DetectAtomicAddFromCallExtern(const tirx::CallNode *call) {
    if (call->args.empty())
      return;
    auto *str_arg = call->args[0].as<tirx::StringImmNode>();
    if (!str_arg)
      return;

    std::string func_name = str_arg->value;
    if (func_name != "atomicAdd" &&
        func_name.find("atomicAdd") == std::string::npos)
      return;

    // Mark buffer arguments (skip first arg which is the function name)
    for (size_t i = 1; i < call->args.size(); ++i) {
      MarkBufferArgAsAtomicAdd(call->args[i]);
    }
  }

  // Detect parameters used in atomic_add_elem_op
  void DetectParamsInAtomicAddElemOp(const tirx::CallNode *call) {
    if (call->args.empty())
      return;
    MarkBufferArgAsAtomicAdd(call->args[0]); // First arg is destination
  }

  // Extract buffer data pointer from various expression types
  const tirx::VarNode *ExtractBufferData(const PrimExpr &arg) const {
    // tvm_access_ptr(buf_var, ...) -> returns buf_var
    if (auto *call = arg.as<tirx::CallNode>()) {
      if (call->op.same_as(tirx::builtin::tvm_access_ptr()) &&
          call->args.size() >= 2) {
        if (auto *var = call->args[1].as<tirx::VarNode>()) {
          return var;
        }
      }
      // address_of(BufferLoad(...)) -> returns buffer->data
      if (call->op.same_as(tirx::builtin::address_of()) &&
          call->args.size() == 1) {
        if (auto *load = call->args[0].as<tirx::BufferLoadNode>()) {
          return load->buffer->data.get();
        }
      }
    }
    // BufferLoad(...) -> returns buffer->data
    if (auto *load = arg.as<tirx::BufferLoadNode>()) {
      return load->buffer->data.get();
    }
    // Direct Var
    if (auto *var = arg.as<tirx::VarNode>()) {
      return var;
    }
    return nullptr;
  }

  // Check if buffer is a tracked parameter or buffer data
  bool IsTrackedBuffer(const tirx::VarNode *buffer_data) const {
    return buffer_data && (params_set.count(buffer_data) ||
                           buffer_data_set.count(buffer_data));
  }

  void MarkBufferArgAsAsyncLoad(const PrimExpr &arg) {
    const tirx::VarNode *buffer_data = ExtractBufferData(arg);
    if (IsTrackedBuffer(buffer_data)) {
      used_in_async_load_.insert(buffer_data);
    }
  }

  void MarkBufferArgAsAtomicAdd(const PrimExpr &arg) {
    const tirx::VarNode *buffer_data = ExtractBufferData(arg);
    if (IsTrackedBuffer(buffer_data)) {
      used_in_atomic_add_.insert(buffer_data);
    }
  }

  // Defense-in-depth recursive visitor for the async DMA destination
  // operand. MarkBufferArgAsAsyncLoad's ExtractBufferData handles the
  // standard address_of(BufferLoad(...)) IR from the injection site, but
  // if the injection site ever wraps the operand in additional expression
  // layers (Cast, Add, nested Call that ExtractBufferData does not
  // recognise), the recursive traversal catches the inner buffer anyway.
  void VisitExpr(const PrimExpr &e) {
    if (auto *var = e.as<tirx::VarNode>()) {
      if (params_set.count(var)) {
        used_in_async_load_.insert(var);
      }
      return;
    }
    const tirx::VarNode *buffer_data = ExtractBufferData(e);
    if (buffer_data && buffer_data_set.count(buffer_data)) {
      used_in_async_load_.insert(buffer_data);
      return;
    }
    if (auto *call = e.as<tirx::CallNode>()) {
      for (const auto &arg : call->args) {
        VisitExpr(arg);
      }
    }
  }

public:
  std::unordered_set<const tirx::VarNode *> used_in_async_load_;
  std::unordered_set<const tirx::VarNode *> used_in_atomic_add_;
};

class LaunchConfigExtractor : public tirx::StmtVisitor {
private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->var->name_hint == "threadIdx.x" ||
          iv->thread_tag == "threadIdx.x") {
        threadIdx_x_ext = op->value;
      } else if (iv->var->name_hint == "threadIdx.y" ||
                 iv->thread_tag == "threadIdx.y") {
        threadIdx_y_ext = op->value;
      } else if (iv->var->name_hint == "threadIdx.z" ||
                 iv->thread_tag == "threadIdx.z") {
        threadIdx_z_ext = op->value;
      }
    } else if (op->attr_key == tl::attr::kMinBlocksPerSM) {
      if (const IntImmNode *v = op->value.as<IntImmNode>()) {
        min_blocks_per_sm = v->value;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

public:
  PrimExpr threadIdx_x_ext = Integer(1);
  PrimExpr threadIdx_y_ext = Integer(1);
  PrimExpr threadIdx_z_ext = Integer(1);
  int64_t min_blocks_per_sm = 1;
};

void CodeGenTileLangTANG::PrintExtraAttrs(const PrimFunc &f) {
  LaunchConfigExtractor extractor;
  extractor(f->body);
  arith::Analyzer analyzer;
  PrimExpr threadIdx_ext =
      analyzer.Simplify(extractor.threadIdx_x_ext * extractor.threadIdx_y_ext *
                        extractor.threadIdx_z_ext);
  if (const IntImmNode *const threadIdx_ext_int =
          threadIdx_ext.as<IntImmNode>()) {
    if (threadIdx_ext_int->value == 1) {
      // unable to extract the number of threads per block, hence directly
      // return
      return;
    }
    stream << " __launch_bounds__(" << threadIdx_ext_int->value << ", "
           << extractor.min_blocks_per_sm << ")";
  }
}

std::string CodeGenTileLangTANG::Finish() {
  decl_stream << "#include <tang.h>\n";
  decl_stream << "#include <tang_runtime.h>\n";
  if (need_mma_h_) {
    decl_stream << "#include <mma.h>\n";
  }

  if (need___clang_tang_builtin_vars_h) {
    decl_stream << "#include <__clang_tang_builtin_vars.h>\n";
  }
  if (need___clang_tang_fp16_h) {
    decl_stream << "#include <__clang_tang_fp16.h>\n";
  }
  if (need___clang_tang_bf16_h) {
    decl_stream << "#include <__clang_tang_bf16.h>\n";
  }

  decl_stream << "#include <tl_templates/tang/reduce.h>\n";
  decl_stream << "#include <tl_templates/tang/threadblock_swizzle.h>\n";
  decl_stream << "#include <tl_templates/tang/gemm.h>\n";
  decl_stream << "#include <tl_templates/tang/debug.h>\n";
  decl_stream << "#include <tl_templates/tang/intrin.h>\n";
  decl_stream << "#include <tl_templates/tang/rng.h>\n";

  if (need_tcgen05_common_h_) {
    // Public PTX umbrella header instead of the private
    // <cccl/tang/__ptx/instructions/*.h> headers (tc_mma / tc_ldst*): it
    // re-exports tc_ldst.h (all 16x256b/32x32b/... variants) and tc_mma.h.
    decl_stream << "#include <cccl/tang/ptx>\n";
  }

  if (need_cp_async_bulk_h_) {
    decl_stream << "#include <tl_templates/tang/copy_fcp_g_s.h>\n";
  }

  if (need_global_barrier_) {
    decl_stream << "__device__ unsigned " << vid_global_barrier_state_
                << " = 0;\n";
  }
  if (need_cooperative_groups_) {
    // Note: sync_grids lives in cooperative_groups::details (an internal
    // namespace) and is not re-exported by the public <cooperative_groups.h>.
    // The public API (this_grid().sync()) would add runtime overhead from
    // grid_group construction, so we use the internal symbol directly.
    // Since sync_grids is a static inline function (~15 lines), the risk of
    // a silent ABI break is negligible; a header rename would produce a
    // compile-time error, not a runtime failure.
    decl_stream << "#include <cooperative_groups/details/sync.h>\n";
    decl_stream << "using cooperative_groups::details::sync_grids;\n";
    decl_stream << "__device__ volatile cooperative_groups::details::barrier_t "
                   "bar0 = 0;\n";
  }

  decl_stream << "\n";

  return CodeGenC::Finish();
}

void CodeGenTileLangTANG::VisitStmt_(const tirx::ForNode *op) {
  // NOTE: A previous revision hoisted the loop-invariant (threadIdx) part of
  // the store index out of the unrolled epilogue loop here. That transform was
  // an assembly-level no-op because the backend compiler already extracts the
  // same loop-invariant address computation, while it made the store (B)
  // address asymmetric with the un-hoisted load (A) address. It was removed to
  // keep read/write addressing symmetric; leave hoisting to the compiler.
  std::string extent =
      PrintExpr(arith::Analyzer().Simplify(op->extent + op->min));
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  std::string start = PrintExpr(op->min);

  if (op->kind == tirx::ForKind::kUnrolled) {
    if (unroll_factor.count(op->loop_var.get())) {
      stream << "#pragma unroll "
             << PrintExpr(unroll_factor[op->loop_var.get()]) << "\n";
    } else if (op->body.as<BufferStoreNode>() ||
               op->body.as<BufferLoadNode>()) {
      stream << "#pragma unroll \n";
    } else {
      // Uniform or non-uniform index: use partial unroll to limit register
      // pressure.  Non-uniform indices may require full unroll for correctness
      // on some compilers; if needed, set pragma_unroll_factor on the loop.
      stream << "#pragma unroll 8\n";
    }
    PrintIndent();
  }

  stream << "for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = " << start << "; " << vid << " < " << extent
         << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenTileLangTANG::BindThreadIndex(const IterVar &iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] =
      CastFromTo(iv->thread_tag, DataType::UInt(32), iv->var.dtype());
}

void CodeGenTileLangTANG::PrintType(DataType t, std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK(t.is_scalar()) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }

  if (t == tl::CuTensorMapType()) {
    os << "CUtensorMap";
    return;
  }

  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
    case 16:
      enable_fp16_ = true;
      if (t.is_scalar()) {
        os << "__fp16";
      } else if (lanes <= 8) {
        // Emit TANG code to access fp16 vector elements.
        //
        // half4 is stored as uint2
        //
        // h4.x is emitted as *(half2*)(&(u2.x)).x
        // h4.y is emitted as *(half2*)(&(u2.x)).y
        // h4.z is emitted as *(half2*)(&(u2.y)).x
        // h4.w is emitted as *(half2*)(&(u2.y)).y
        //
        ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
        os << "uint" << lanes / 2;
      } else if (lanes <= 16) {
        ICHECK_EQ(lanes % 4, 0) << "only support (mod 4 = 0) lanes for half "
                                   "type of more than 8 lanes";
        os << "ulonglong" << lanes / 4;
      } else {
        fail = true;
      }
      break;
    case 32:
      if (lanes <= 4) {
        os << "float";
      } else if (lanes <= 8) {
        // Emit TANG code to access fp32 vector elements for 4 < lanes <= 8.
        //
        // float8 is stored as ulonglong4
        //
        // f8.v1 is emitted as *(float2*)(&(ul4.x)).x
        // f8.v2 is emitted as *(float2*)(&(ul4.x)).y
        //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for float type with lanes > 4";
        os << "ulonglong" << lanes / 2;
      } else {
        fail = true;
      }
      break;
    case 64:
      os << "double";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && (t.is_scalar() || t.bits() == 16))
      return;
    if (!fail && (lanes > 4 && lanes <= 8 && t.bits() == 32))
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_bfloat16()) {
    enable_bf16_ = true;
    if (t.is_scalar()) {
      os << "__bf16";
    } else if (lanes <= 8) {
      ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
      os << "uint" << lanes / 2;
    } else if (lanes <= 16) {
      ICHECK_EQ(lanes % 4, 0) << "only support (mod 4 = 0) lanes for half type "
                                 "of more than 8 lanes";
      os << "ulonglong" << lanes / 4;
    } else {
      fail = true;
    }
    if (!fail)
      return;
  } else if (t.is_float8()) {
    enable_fp8_ = true;
    os << GetFP8Type(t);
    return;
  } else if (t.is_float6()) {
    enable_fp6_ = true;
    if (t.lanes() <= 4) {
      os << GetFP6Type(t);
    }
    return;
  } else if (t.is_float4()) {
    enable_fp4_ = true;
    if (t.lanes() <= 4) {
      os << GetFP4Type(t);
    }
    return;
  } else if (t == DataType::Bool()) {
    os << "bool";
    return;
  } else if (t.is_vector_bool()) {
    // TANG does not support bool vectors.
    // Use ushort vectors to represent instead.
    int n = t.lanes();
    if (n <= 4) {
      os << "ushort" << n;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "u";
    }
    switch (t.bits()) {
    case 1: {
      if (t.is_scalar()) {
        os << "int";
        return;
      } else if (t.lanes() == 8) {
        os << "int8_t";
        return;
      } else if (t.lanes() == 16) {
        os << "int16_t";
        return;
      } else if (t.lanes() == 32) {
        os << "int";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t << " to TANG type!";
      }
    }
    case 4: {
      if (t.is_scalar()) {
        os << "int";
        return;
      } else if (t.lanes() == 4) {
        os << "int16_t";
        return;
      } else if (t.lanes() == 8) {
        // directly 8 4-bit int in integer.
        os << "int";
        return;
      } else if (t.lanes() == 16) {
        os << "int2";
        return;
      } else if (t.lanes() == 32) {
        os << "int4";
        return;
      } else if (t.lanes() == 64) {
        os << "int8";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t << " to TANG type!";
      }
    }
    case 8: {
      if (t.lanes() == 4) {
        // directly 4 8 bit int in integer.
        enable_int8_ = true;

        // We use int for int8x4 instead of char4 because using char4 is
        // likely to produce extra instructions to pack four int8 elements
        // into 32-bit data.
        os << "int";
        return;
      } else if (t.lanes() == 8) {
        enable_int8_ = true;
        os << "int2";
        return;
      } else if (t.lanes() == 16) {
        enable_int8_ = true;
        os << "int4";
        return;
      } else if (t.lanes() == 32) {
        enable_int8_ = true;
        os << "longlong4";
        return;
      } else if (!t.is_uint() && t.is_scalar()) {
        os << "signed char";
        break;
      } else {
        os << "char";
        break;
      }
    }
    case 16: {
      if (t.is_scalar()) {
        os << "short";
      } else if (t.lanes() <= 4) {
        os << "short" << lanes;
      } else if (t.lanes() <= 8) {
        // Emit TANG code to access int16 vector elements.
        //
        // short4 is stored as int2
        //
        // s4.x is emitted as *(short2*)(&(i2.x)).x
        // s4.y is emitted as *(short2*)(&(i2.x)).y
        // s4.z is emitted as *(short2*)(&(i2.y)).x
        // s4.w is emitted as *(short2*)(&(i2.y)).y
        //
        ICHECK_EQ(t.lanes() % 2, 0)
            << "only support even lane for shorT type with lanes > 4";
        os << "int" << t.lanes() / 2;
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 32: {
      if (t.is_scalar()) {
        os << "int";
      } else if (t.lanes() <= 4) {
        os << "int" << t.lanes();
      } else if (t.lanes() <= 8) {
        // Emit TANG code to access int32 vector elements for 4 < lanes <= 8.
        //
        // int8 is stored as longlong4
        //
        // i8.v1 is emitted as *(int2*)(&(l4.x)).x
        // i8.v2 is emitted as *(int2*)(&(l4.x)).y
        //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for int32 type with lanes > 4";
        os << "longlong" << lanes / 2;
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 64: {
      if (t.is_scalar()) {
        os << "int64_t";
      } else if (t.lanes() == 2) {
        os << "longlong2";
      } else if (t.lanes() == 3) {
        os << "longlong3";
      } else if (t.lanes() == 4) {
        os << "longlong4";
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1) {
      return;
    }
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to TANG type";
}

void CodeGenTileLangTANG::PrintVecBinaryOp(const std::string &op, DataType t,
                                           PrimExpr lhs, PrimExpr rhs,
                                           std::ostream &os) { // NOLINT(*)
  // Fast-path for packed FP32x2 arithmetic (stcuv2 only).
  Target cur_target = Target::Current(/*allow_not_defined=*/true);
  bool target_supports_f32x2_packed =
      cur_target.defined() && tl::TargetTangIsSTCUV2(cur_target);
  if (target_supports_f32x2_packed && t.is_float() && t.bits() == 32 &&
      t.lanes() == 2) {
    if (op == "+") {
      os << "tl::fadd2(" << PrintExpr(lhs) << ", " << PrintExpr(rhs) << ")";
      return;
    }
    if (op == "*") {
      os << "tl::fmul2(" << PrintExpr(lhs) << ", " << PrintExpr(rhs) << ")";
      return;
    }
  }

  // Declare the result.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  int ssa_scope = BeginScope();
  {
    // Unpack into individual ops.
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.dtype());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.dtype());

    for (int i = 0, lanes = t.lanes(); i < lanes; ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
  }
  EndScope(ssa_scope);
  os << sret;
}

void CodeGenTileLangTANG::PrintVecConstructor(DataType t, std::ostream &os) {
  os << "make_";
  PrintType(t, os);
}

void CodeGenTileLangTANG::PrintVecElemLoad(const std::string &vec, DataType t,
                                           int i,
                                           std::ostream &os) { // NOLINT(*)
  if (t.is_scalar()) {
    os << vec;
    return;
  }

  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < 256 / t.bits());
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    std::string type_name = t.is_int() ? "char" : "unsigned char";
    if (t.lanes() == 2 || t.lanes() == 3) {
      os << vec << "." << access[i % t.lanes()];
    } else if (t.lanes() <= 16) {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      os << "((" << type_name << ")(" << ac << " >> " << i % 4 * 8 << "))";
    } else {
      ICHECK(t.lanes() == 32);
      std::string ac = vec + "." + access[i / 8];
      os << "((" << type_name << ")(" << ac << " >> " << i % 8 * 8 << "))";
    }
  } else if (t.is_float16()) {
    if (t.lanes() <= 8) {
      os << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->"
         << access[i % 2];
    } else {
      os << "(((half2*)(&(" << vec << "." << access[i / 4] << "))) + "
         << (i / 2 % 2) << ")->" << access[i % 2];
    }
  } else if (t.is_bfloat16()) {
    need___clang_tang_bf16_h = true;
    if (t.lanes() <= 8) {
      os << "((__tang_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->"
         << access[i % 2];
    } else {
      os << "(((__tang_bfloat162*)(&(" << vec << "." << access[i / 4]
         << "))) + " << (i / 2 % 2) << ")->" << access[i % 2];
    }
  } else if (t.is_float8()) {
    os << vec;
    // fp8_e5_32_t
    if (t.lanes() >= 32)
      os << "." << access[i / 16];
    // fp8_e5_16_t
    if (t.lanes() >= 16)
      os << "." << access[(i % 16) / 8];
    // fp8_e5_8_t
    if (t.lanes() >= 8)
      os << "." << access[(i % 8) / 4];
    // fp8_e5_4_t or fp8_e5_2_t
    os << "." << access[i % 4];
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    os << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2]
       << ")))->" << access[i % 2];
  } else {
    os << vec << "." << access[i];
  }
}

void CodeGenTileLangTANG::PrintVecElemStore(const std::string &vec, DataType t,
                                            int i, const std::string &value) {
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < 256 / t.bits());
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (t.lanes() == 2 || t.lanes() == 3) {
      stream << vec << '.' << access[i % t.lanes()] << "="
             << "(" << value << ");\n";
    } else if (t.lanes() <= 16) {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      stream << ac << "=";
      // Do not read the first undef lane.
      if (i != 0) {
        stream << ac << " & ~(0x000000ff << " << i % 4 * 8 << ") |";
      }
      stream << "(" << value << " << " << i % 4 * 8 << ");\n";
    } else {
      ICHECK(t.lanes() == 32);
      std::string ac = vec + "." + access[i / 8];
      stream << ac << "=";
      // Do not read the first undef lane.
      if (i != 0) {
        stream << ac << " & ~(0x000000ff << " << i % 8 * 8 << ") |";
      }
      stream << "(" << value << " << " << i % 8 * 8 << ");\n";
    }
  } else if (t.is_float16()) {
    if (t.lanes() <= 8) {
      stream << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->"
             << access[i % 2] << " = " << value << ";\n";
    } else {
      stream << "(((half2*)(&(" << vec << "." << access[i / 4] << "))) + "
             << (i / 2 % 2) << ")->" << access[i % 2] << " = " << value
             << ";\n";
    }
  } else if (t.is_bfloat16()) {
    need___clang_tang_bf16_h = true;
    if (t.lanes() <= 8) {
      stream << "((__tang_bfloat162*)(&(" << vec << "." << access[i / 2]
             << ")))->" << access[i % 2] << " = " << value << ";\n";
    } else {
      stream << "(((__tang_bfloat162*)(&(" << vec << "." << access[i / 4]
             << "))) + " << (i / 2 % 2) << ")->" << access[i % 2] << " = "
             << value << ";\n";
    }
  } else if (t.is_float8()) {
    stream << vec;
    // fp8_e5_32_t
    if (t.lanes() >= 32)
      stream << "." << access[i / 16];
    // fp8_e5_16_t
    if (t.lanes() >= 16)
      stream << "." << access[(i % 16) / 8];
    // fp8_e5_8_t
    if (t.lanes() >= 8)
      stream << "." << access[(i % 8) / 4];
    // fp8_e5_4_t or fp8_e5_2_t
    stream << "." << access[i % 4] << " = " << value << ";\n";
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    stream << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2]
           << ")))->" << access[i % 2] << " = " << value << ";\n";
  } else {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  }
}

void CodeGenTileLangTANG::PrintStorageSync(const CallNode *op) {
  auto args = op->args;
  const std::string &sync = args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    // DO nothing.
  } else if (sync == "shared" || sync == "shared.dyn") {
    this->PrintIndent();
    if (args.size() == 1) {
      this->stream << "__syncthreads();\n";
    } else {
      // For partial sync (barrier_id, thread_count), fallback to full
      // __syncthreads() since TANG does not support partial barriers.
      this->stream << "__syncthreads();\n";
    }
  } else if (sync == "global") {
    if (!need_global_barrier_) {
      need_global_barrier_ = true;
    }
    // global synchronizer
    std::string is_load = PrintExpr(op->args[1]);
    std::string num_blocks = PrintExpr(op->args[2]);
    this->PrintIndent();
    // In theory only threadfence is needed
    // but we observed problems with only threadfence
    this->stream << "__threadfence_system();\n";
    this->PrintIndent();
    this->stream << "if (" << is_load << ") {\n";
    int wb = this->BeginScope();
    this->PrintIndent();
    this->stream << "atomicAdd(&" << vid_global_barrier_state_ << ", 1);\n";
    this->PrintIndent();
    std::string ptr = name_supply_->FreshName("pf");
    this->stream << "volatile unsigned* " << ptr << " = &"
                 << vid_global_barrier_state_ << ";\n";
    this->PrintIndent();
    this->stream << vid_global_barrier_expect_ << " += " << num_blocks << ";\n";
    this->PrintIndent();
    this->stream << "while (" << ptr << "[0] < " << vid_global_barrier_expect_
                 << ");\n";
    this->EndScope(wb);
    this->PrintIndent();
    this->stream << "}\n";
    this->PrintIndent();
    this->stream << "__syncthreads();\n";
  }
}

void CodeGenTileLangTANG::PrintStorageScope(const std::string &scope,
                                            std::ostream &os) { // NOLINT(*)
  ICHECK_NE(scope, "global")
      << "Cannot allocate global memory when targeting TANG. You must pass "
         "all global arrays as input instead";
  if (scope == "shared" || scope == "shared.barrier" ||
      scope == "shared.tmem" || scope == "shared.tmem_addr") {
    os << "__shared__ ";
  } else if (scope == "shared.dyn") {
    Target cur_target = Target::Current(/*allow_not_defined=*/true);
    bool is_stcuv2 = cur_target.defined() && tl::TargetTangIsSTCUV2(cur_target);
    if (is_stcuv2) {
      // S3 (stcuv2): emit the merged "dynamic" shared buffer as a *static*
      // 512-byte-aligned array rather than an `extern __shared__` one. The
      // TC-Gen5 TensorCore requires its shared-memory operands to be 512-byte
      // aligned; an `extern __shared__` region begins right after any static
      // shared memory (barriers, tmem_addr, ...), so its base is generally not
      // 512-aligned (observed base address 0x8) and the TensorCore aborts.
      // Since the buffer is reserved statically, LowerDeviceKernelLaunch omits
      // the dynamic-shared launch parameter for stcuv2 so the launch requests 0
      // dynamic bytes -- avoiding a double-counting of the shared-memory
      // budget.
      os << "__shared__ __align__(512) ";
    } else {
      // S2 (stcu): real dynamic shared memory, whose size is supplied at launch
      // via dyn_shmem_size, exactly like the CUDA backend.
      os << "extern __shared__ __align__(1024) ";
    }
  }
}

std::string CodeGenTileLangTANG::CastFromTo(std::string value, DataType from,
                                            DataType target) {
  if (from == target)
    return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")";
  if (from.is_float16() && (target.is_int() || target.is_uint()) &&
      target.bits() == 8) {
    os << "(";
    if (target.is_uint()) {
      os << "u";
    }
    os << "int)";
  }
  if ((from.is_float16() || from.is_bfloat16()) && target.is_float8()) {
    os << "(float)";
  }
  os << value << ")";
  return os.str();
}

void CodeGenTileLangTANG::VisitExpr_(const CastNode *op, std::ostream &os) {
  DataType from_ty = op->value.dtype();
  DataType target_ty = op->dtype;
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  if (from_ty.is_float16() || target_ty.is_float16()) {
    need___clang_tang_fp16_h = true;
  }
  if (from_ty.is_bfloat16() || target_ty.is_bfloat16()) {
    need___clang_tang_bf16_h = true;
  }

  // Emit simple C-style type conversion.
  if (from_ty.is_scalar()) {
    if (from_ty.is_float16() && target_ty.is_bfloat16()) {
      os << "__float2bfloat16_rn(__half2float(" << PrintExpr(op->value) << "))";
      return;
    }
    if (from_ty.is_bfloat16() && target_ty.is_float16()) {
      os << "__float2half_rn(__bfloat162float(" << PrintExpr(op->value) << "))";
      return;
    }
    return CodeGenC::VisitExpr_(op, os);
  }

  // We could emit make_float4 like calls, but the emitted code looks
  // too compact to read. Emit this as vectorized unary ops.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(target_ty, stream);
  stream << ' ' << sret << ";\n";
  std::string src = SSAGetID(PrintExpr(op->value), from_ty);

  // Handle conversion between float16 and float32
  if (from_ty.is_float16() && target_ty.is_float() && target_ty.bits() == 32) {
    // Use __half22float2 for vectorized conversion (half2 -> float2)
    if (from_ty.lanes() == 2 && target_ty.lanes() == 2) {
      // half2 -> float2
      PrintIndent();
      stream << sret << " = __half22float2(*(half2*)(&(" << src << ")));\n";
      os << sret;
      return;
    } else if (from_ty.lanes() == 4 && target_ty.lanes() == 4) {
      // half4 -> float4
      PrintIndent();
      stream << "((float2*)(&" << sret << "))[0] = "
             << "__half22float2(*(half2*)(&(" << src << ")));\n";
      PrintIndent();
      stream << "((float2*)(&" << sret << "))[1] = "
             << "__half22float2(*((half2*)(&(" << src << "))+1));\n";
      os << sret;
      return;
    } else if (from_ty.lanes() == 8 && target_ty.lanes() == 8) {
      // half8 -> float8
      PrintIndent();
      stream << "((float2*)(&" << sret << "))[0] = "
             << "__half22float2(*(half2*)(&(" << src << ")));\n";
      PrintIndent();
      stream << "((float2*)(&" << sret << "))[1] = "
             << "__half22float2(*((half2*)(&(" << src << "))+1));\n";
      PrintIndent();
      stream << "((float2*)(&" << sret << "))[2] = "
             << "__half22float2(*((half2*)(&(" << src << "))+2));\n";
      PrintIndent();
      stream << "((float2*)(&" << sret << "))[3] = "
             << "__half22float2(*((half2*)(&(" << src << "))+3));\n";
      os << sret;
      return;
    }
  } else if (from_ty.is_float() && from_ty.bits() == 32 &&
             target_ty.is_float16()) {
    // Use __float22half2_rn for vectorized conversion (float2 -> half2)
    if (from_ty.lanes() == 2 && target_ty.lanes() == 2) {
      // float2 -> half2
      PrintIndent();
      stream << "*(half2*)(&(" << sret << ")) = __float22half2_rn(*(float2*)(&("
             << src << ")));\n";
      os << sret;
      return;
    } else if (from_ty.lanes() == 4 && target_ty.lanes() == 4) {
      // float4 -> half4
      PrintIndent();
      stream << "((half2*)(&" << sret << "))[0] = "
             << "__float22half2_rn(*(float2*)(&(" << src << ")));\n";
      PrintIndent();
      stream << "((half2*)(&" << sret << "))[1] = "
             << "__float22half2_rn(*((float2*)(&(" << src << "))+1));\n";
      os << sret;
      return;
    } else if (from_ty.lanes() == 8 && target_ty.lanes() == 8) {
      // float8 -> half8
      PrintIndent();
      stream << "((half2*)(&" << sret << "))[0] = "
             << "__float22half2_rn(*(float2*)(&(" << src << ")));\n";
      PrintIndent();
      stream << "((half2*)(&" << sret << "))[1] = "
             << "__float22half2_rn(*((float2*)(&(" << src << "))+1));\n";
      PrintIndent();
      stream << "((half2*)(&" << sret << "))[2] = "
             << "__float22half2_rn(*((float2*)(&(" << src << "))+2));\n";
      PrintIndent();
      stream << "((half2*)(&" << sret << "))[3] = "
             << "__float22half2_rn(*((float2*)(&(" << src << "))+3));\n";
      os << sret;
      return;
    }
  }

  // Handle conversion between bfloat16 and float32
  if (from_ty.is_bfloat16() && target_ty.is_float() && target_ty.bits() == 32) {
    // Use __bfloat1622float2 for vectorized conversion (bfloat162 -> float2)
    need___clang_tang_bf16_h = true;
    if (from_ty.lanes() == 2 && target_ty.lanes() == 2) {
      // bfloat162 -> float2
      PrintIndent();
      stream << sret
             << " = __bfloat1622float2(*reinterpret_cast<__tang_bfloat162*>(&("
             << src << ")));\n";
      os << sret;
      return;
    } else if (from_ty.lanes() == 4 && target_ty.lanes() == 4) {
      // bfloat162x2 -> float4
      PrintIndent();
      stream << "((float2*)(&" << sret << "))[0] = "
             << "__bfloat1622float2(*reinterpret_cast<__tang_bfloat162*>(&("
             << src << ")));\n";
      PrintIndent();
      stream << "((float2*)(&" << sret << "))[1] = "
             << "__bfloat1622float2(*(reinterpret_cast<__tang_bfloat162*>(&("
             << src << "))+1));\n";
      os << sret;
      return;
    } else if (from_ty.lanes() == 8 && target_ty.lanes() == 8) {
      // bfloat162x4 -> float8
      PrintIndent();
      stream << "((float2*)(&" << sret << "))[0] = "
             << "__bfloat1622float2(*reinterpret_cast<__tang_bfloat162*>(&("
             << src << ")));\n";
      PrintIndent();
      stream << "((float2*)(&" << sret << "))[1] = "
             << "__bfloat1622float2(*(reinterpret_cast<__tang_bfloat162*>(&("
             << src << "))+1));\n";
      PrintIndent();
      stream << "((float2*)(&" << sret << "))[2] = "
             << "__bfloat1622float2(*(reinterpret_cast<__tang_bfloat162*>(&("
             << src << "))+2));\n";
      PrintIndent();
      stream << "((float2*)(&" << sret << "))[3] = "
             << "__bfloat1622float2(*(reinterpret_cast<__tang_bfloat162*>(&("
             << src << "))+3));\n";
      os << sret;
      return;
    }
  } else if (from_ty.is_float() && from_ty.bits() == 32 &&
             target_ty.is_bfloat16()) {
    // Use __float22bfloat162_rn for vectorized conversion (float2 -> bfloat162)
    need___clang_tang_bf16_h = true;
    if (from_ty.lanes() == 2 && target_ty.lanes() == 2) {
      // float2 -> bfloat162
      PrintIndent();
      stream << "*reinterpret_cast<__tang_bfloat162*>(&(" << sret
             << ")) = __float22bfloat162_rn(*(float2*)(&(" << src << ")));\n";
      os << sret;
      return;
    } else if (from_ty.lanes() == 4 && target_ty.lanes() == 4) {
      // float4 -> bfloat162x2
      PrintIndent();
      stream << "(reinterpret_cast<__tang_bfloat162*>(&" << sret << "))[0] = "
             << "__float22bfloat162_rn(*(float2*)(&(" << src << ")));\n";
      PrintIndent();
      stream << "(reinterpret_cast<__tang_bfloat162*>(&" << sret << "))[1] = "
             << "__float22bfloat162_rn(*((float2*)(&(" << src << "))+1));\n";
      os << sret;
      return;
    } else if (from_ty.lanes() == 8 && target_ty.lanes() == 8) {
      // float8 -> bfloat162x4
      PrintIndent();
      stream << "(reinterpret_cast<__tang_bfloat162*>(&" << sret << "))[0] = "
             << "__float22bfloat162_rn(*(float2*)(&(" << src << ")));\n";
      PrintIndent();
      stream << "(reinterpret_cast<__tang_bfloat162*>(&" << sret << "))[1] = "
             << "__float22bfloat162_rn(*((float2*)(&(" << src << "))+1));\n";
      PrintIndent();
      stream << "(reinterpret_cast<__tang_bfloat162*>(&" << sret << "))[2] = "
             << "__float22bfloat162_rn(*((float2*)(&(" << src << "))+2));\n";
      PrintIndent();
      stream << "(reinterpret_cast<__tang_bfloat162*>(&" << sret << "))[3] = "
             << "__float22bfloat162_rn(*((float2*)(&(" << src << "))+3));\n";
      os << sret;
      return;
    }
  }

  // Convert directly between float16 and bfloat16 through float32 pairs.
  if ((from_ty.is_float16() && target_ty.is_bfloat16()) ||
      (from_ty.is_bfloat16() && target_ty.is_float16())) {
    const int lanes = from_ty.lanes();
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      for (int i = 0; i < lanes / 2; ++i) {
        PrintIndent();
        if (from_ty.is_float16()) {
          stream << "(reinterpret_cast<__tang_bfloat162*>(&" << sret << "))["
                 << i << "] = __float22bfloat162_rn(__half22float2("
                 << "(reinterpret_cast<half2*>(&" << src << "))[" << i
                 << "]));\n";
        } else {
          stream << "(reinterpret_cast<half2*>(&" << sret << "))[" << i
                 << "] = __float22half2_rn(__bfloat1622float2("
                 << "(reinterpret_cast<__tang_bfloat162*>(&" << src << "))["
                 << i << "]));\n";
        }
      }
      os << sret;
      return;
    }
  }

  // Handle conversion from float32 to float8 (E4M3/E5M2)
  if (from_ty.is_float() && from_ty.bits() == 32 &&
      (target_ty.is_float8_e4m3() || target_ty.is_float8_e5m2())) {
    // FP32 -> FP8: Use __nv_cvt_float2_to_fp8x2 for vectorized conversion
    // (float2 -> fp8x2)
    if (from_ty.lanes() == 2 && target_ty.lanes() == 2) {
      // float2 -> fp8x2
      PrintIndent();
      stream << "*reinterpret_cast<__nv_fp8x2_storage_t*>(&(" << sret
             << ")) = __nv_cvt_float2_to_fp8x2(*reinterpret_cast<float2*>(&("
             << src << ")), __NV_SATFINITE, "
             << (target_ty.is_float8_e4m3() ? "__NV_E4M3" : "__NV_E5M2")
             << ");\n";
      os << sret;
      return;
    } else if (from_ty.lanes() == 4 && target_ty.lanes() == 4) {
      // float4 -> fp8x4
      PrintIndent();
      stream << "((__nv_fp8x2_storage_t*)(&" << sret << "))[0] = "
             << "__nv_cvt_float2_to_fp8x2(*(float2*)(&(" << src
             << ")), __NV_SATFINITE, "
             << (target_ty.is_float8_e4m3() ? "__NV_E4M3" : "__NV_E5M2")
             << ");\n";
      PrintIndent();
      stream << "((__nv_fp8x2_storage_t*)(&" << sret << "))[1] = "
             << "__nv_cvt_float2_to_fp8x2(*((float2*)(&(" << src
             << "))+1), __NV_SATFINITE, "
             << (target_ty.is_float8_e4m3() ? "__NV_E4M3" : "__NV_E5M2")
             << ");\n";
      os << sret;
      return;
    } else if (from_ty.lanes() == 8 && target_ty.lanes() == 8) {
      // float8 -> fp8x8
      PrintIndent();
      stream << "((__nv_fp8x2_storage_t*)(&" << sret << "))[0] = "
             << "__nv_cvt_float2_to_fp8x2(*(float2*)(&(" << src
             << ")), __NV_SATFINITE, "
             << (target_ty.is_float8_e4m3() ? "__NV_E4M3" : "__NV_E5M2")
             << ");\n";
      PrintIndent();
      stream << "((__nv_fp8x2_storage_t*)(&" << sret << "))[1] = "
             << "__nv_cvt_float2_to_fp8x2(*((float2*)(&(" << src
             << "))+1), __NV_SATFINITE, "
             << (target_ty.is_float8_e4m3() ? "__NV_E4M3" : "__NV_E5M2")
             << ");\n";
      PrintIndent();
      stream << "((__nv_fp8x2_storage_t*)(&" << sret << "))[2] = "
             << "__nv_cvt_float2_to_fp8x2(*((float2*)(&(" << src
             << "))+2), __NV_SATFINITE, "
             << (target_ty.is_float8_e4m3() ? "__NV_E4M3" : "__NV_E5M2")
             << ");\n";
      PrintIndent();
      stream << "((__nv_fp8x2_storage_t*)(&" << sret << "))[3] = "
             << "__nv_cvt_float2_to_fp8x2(*((float2*)(&(" << src
             << "))+3), __NV_SATFINITE, "
             << (target_ty.is_float8_e4m3() ? "__NV_E4M3" : "__NV_E5M2")
             << ");\n";
      os << sret;
      return;
    }
  }

  // Fallback: elementwise cast
  for (int i = 0, lanes = from_ty.lanes(); i < lanes; ++i) {
    std::ostringstream val;
    val << "(";
    PrintType(target_ty.element_of(), val);
    val << ")(";
    PrintVecElemLoad(src, from_ty, i, val);
    val << ")";
    PrintVecElemStore(sret, target_ty, i, val.str());
  }

  os << sret;
}

void CodeGenTileLangTANG::VisitExpr_(const MinNode *op, std::ostream &os) {
  // TODO(wt): Consider vectorized reduction and impl for other dtypes
  DataType t = op->dtype;

  // Emit tl::MinOp{} so that NaN semantics (ignoring vs propagation) are
  // handled in reduce.h, not duplicated in the codegen.
  if ((t.is_bfloat16() || t.is_float16() || t.is_float()) && t.is_scalar()) {
    os << "tl::MinOp{}(" << PrintExpr(op->a) << ", " << PrintExpr(op->b) << ")";
    return;
  }

  // For all other scalar types (int, uint), use default implementation
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangTANG::VisitExpr_(const MaxNode *op, std::ostream &os) {
  // TODO(wt): Consider vectorized reduction and impl for other dtypes
  DataType t = op->dtype;

  // Emit tl::MaxOp{} so that NaN semantics (ignoring vs propagation) are
  // handled in reduce.h, not duplicated in the codegen.
  if ((t.is_bfloat16() || t.is_float16() || t.is_float()) && t.is_scalar()) {
    os << "tl::MaxOp{}(" << PrintExpr(op->a) << ", " << PrintExpr(op->b) << ")";
    return;
  }

  // For all other scalar types (int, uint), use default implementation
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangTANG::PrintCallExtern(Type ret_type, String global_symbol,
                                          const Array<PrimExpr> &args,
                                          bool skip_first_arg,
                                          std::ostream &os) { // NOLINT(*)
  DataType ret_dtype = GetRuntimeDataType(ret_type);
  if (ret_dtype.is_fixed_length_vector()) {
    //
    // Emit an unsupported vector call
    //
    // v = intrin_f((float4*)A[0], (float4*)B[0])
    //
    // as
    //
    // float4 __ret;
    // {
    //   float4 __arg0 = ((float4*)A)[0];
    //   float4 __arg1 = ((float4*)B)[0];
    //   __ret.x = intrin_f(__arg0.x, __arg1.x);
    //   __ret.y = intrin_f(__arg0.y, __arg1.y);
    //   __ret.z = intrin_f(__arg0.z, __arg1.z);
    //   __ret.w = intrin_f(__arg0.w, __arg1.w);
    // }
    // v = __ret;
    //
    // Declare the result vector.
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(ret_dtype, stream);
    stream << ' ' << sret << ";\n";
    {
      // Load arguments.
      std::vector<std::string> sargs;
      size_t arg_begin = static_cast<size_t>(skip_first_arg);
      for (size_t i = arg_begin; i < args.size(); ++i) {
        std::string val = SSAGetID(PrintExpr(args[i]), args[i].dtype());
        sargs.push_back(std::move(val));
      }

      // Emit a scalar call for each lane.
      for (int i = 0; i < ret_dtype.lanes(); ++i) {
        std::ostringstream scall;
        scall << global_symbol << "(";
        for (size_t j = 0; j < sargs.size(); ++j) {
          if (j > 0)
            scall << ", ";
          PrintVecElemLoad(sargs[j], args[arg_begin + j].dtype(), i, scall);
        }
        scall << ")";
        PrintVecElemStore(sret, ret_dtype, i, scall.str());
      }
    }
    os << sret;
  } else {
    CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg,
                              os);
  }
}

// Print a reference expression to a buffer.
std::string CodeGenTileLangTANG::GetBufferRef(DataType t,
                                              const BufferNode *buffer,
                                              PrimExpr index) {
  const VarNode *buffer_var = buffer->data.get();
  std::ostringstream os;
  std::string vid = GetVarID(buffer_var);
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  // bool is_vol = IsVolatile(buffer_var);
  // always false for tl cutlass backend.
  bool is_vol = false;

  auto ptr_cast = [this, is_vol, scope](DataType pointed_to) {
    std::ostringstream ptr_os;
    ptr_os << "(";
    if (is_vol) {
      ptr_os << "volatile ";
    }
    if (!scope.empty() && IsScopePartOfType()) {
      PrintStorageScope(scope, ptr_os);
    }
    PrintType(pointed_to, ptr_os);
    ptr_os << "*)";
    return ptr_os.str();
  };

  DataType buffer_element_dtype = buffer->dtype;

  std::string buffer_str = vid;
  if (!HandleTypeMatch(buffer_var, buffer_element_dtype) || is_vol) {
    std::stringstream temp;
    temp << "(" << ptr_cast(buffer_element_dtype) << vid << ")";
    buffer_str = temp.str();
  }
  if (scope.empty()) {
    scope = GetPtrStorageScope(buffer->data);
  }
  if (scope == "local.var") {
    os << vid;
    return os.str();
  }
  std::string index_str = PrintExpr(index);
  if (t.is_float4_e2m1fn()) {
    // fp4 is represented by a 1-byte container holding two packed elements
    // (PrintType emits ``uchar``), so a scalar element index maps to a byte
    // index by dividing by 2 (= 8 bits / 4 bits). Vector lanes already span a
    // whole container, so divide by the lane count.
    int div_factor = (t.lanes() == 1) ? (8 / t.bits()) : t.lanes();
    os << "*("
       << "(" << ptr_cast(t) << vid << ")"
       << " + " << index_str << " / " << div_factor << ")";
  } else if (t.bits() == 4 || (t.bits() == 1 && t.is_int())) {
    // This is a special case, because CodegenTANG::PrintType()
    // returns "int" for bool and for 4-bit integers. In most cases,
    // we divide by the number of lanes to determine the index.
    // However, the backing type for scalar int4 and scalar bool is
    // int32.  Therefore, we need to divide by the ratio of their
    // sizes in that case.
    int div_factor = (t.lanes() == 1) ? (32 / t.bits()) : t.lanes();

    os << "*("
       << "(" << ptr_cast(t) << vid << ")"
       << " + " << index_str << " / " << div_factor << ")";
  } else if (t == buffer_element_dtype) {
    os << buffer_str << "[" << index_str << "]";
  } else {
    os << "*" << ptr_cast(t) << "(" << buffer_str << " + " << index_str << ")";
  }

  return os.str();
}

std::string CodeGenTileLangTANG::GetVecLoad(DataType t,
                                            const BufferNode *buffer,
                                            PrimExpr base) {
  const VarNode *buffer_var = buffer->data.get();
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  if (scope.empty()) {
    scope = GetPtrStorageScope(buffer->data);
  }

  if (scope != "global" || t.bits() * t.lanes() <= 256) {
    return this->CodeGenC::GetVecLoad(t, buffer, base);
  }
  ICHECK_EQ(t.bits() * t.lanes(), 256)
      << "Unsupported vector load size: " << t.bits() * t.lanes();
  auto buffer_ref = this->GetBufferRef(t, buffer, base);
  std::ostringstream os;
  os << "tl::ld_global_256(&(" << buffer_ref << "))";
  return os.str();
}

void CodeGenTileLangTANG::PrintVecStore(const BufferNode *buffer, DataType t,
                                        PrimExpr base,
                                        const std::string &value) {
  const VarNode *buffer_var = buffer->data.get();
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  if (scope.empty()) {
    scope = GetPtrStorageScope(buffer->data);
  }

  if (scope != "global" || t.bits() * t.lanes() <= 256) {
    this->CodeGenC::PrintVecStore(buffer, t, base, value);
    return;
  }
  ICHECK_EQ(t.bits() * t.lanes(), 256)
      << "Unsupported vector load size: " << t.bits() * t.lanes();
  auto buffer_ref = this->GetBufferRef(t, buffer, base);
  this->PrintIndent();
  this->stream << "tl::st_global_256(&(" << buffer_ref << "), " << value
               << ");\n";
}

/**
 * @brief Emit TANG/TensorLib-specific code for a call expression.
 *
 * This visitor handles CallNode intrinsics and builtins that require emitting
 * TANG/TL-specific code (inline PTX/ASM sequences, TensorLanguage runtime
 * calls, WMMA/TMA helpers, barriers, cp.async primitives, index-map based
 * stores, reinterpret/packing helpers, and various mma/ldmatrix patterns). The
 * function writes the generated code to the provided output stream and falls
 * back to the C codegen for unrecognized calls.
 *
 * The method recognizes and emits code for (non-exhaustive): cp.async and its
 * commit/wait variants, tma_load/store and im2col variants, ptX
 * ldmatrix/stmatrix helpers, mbarrier APIs, cooperative grid sync, WMMA/legacy
 * MMA intrinsics (fill/load/store/mma/bmma/ptx_mma/ptx_mma_sp), low-level PTX
 * asm helpers (ldg32, cp_async bulk/init/arrive/wait barriers), reinterpret
 * paths for special small-float encodings (e.g., float4 e2m1fn), tl::tl_gemm
 * and related external calls, and other TL runtime calls.
 *
 * Side effects:
 * - Emits to `os` and the internal codegen output stream.
 * - May set internal feature flags (e.g., need_cooperative_groups_,
 * need_mma_h_, need_cast_smem_ptr_to_int_, enable_sparse_gemm_).
 * - May open/close SSA scopes and mutate internal variable mappings.
 * - May call LOG(FATAL) / CHECK / ICHECK on invalid or unsupported argument
 *   patterns.
 *
 * @param op The call node to generate code for; the function inspects op->op
 *           and op->args to determine the appropriate emission.
 * @param os  Output stream to receive expression-level output when the caller
 *            expects an expression result (some paths write directly to the
 *            member stream instead).
 */
void CodeGenTileLangTANG::VisitExpr_(const CallNode *op, std::ostream &os) {
  auto print_extern_call_stmt = [&](std::string name, size_t start = 0,
                                    size_t end = 0) {
    // Cache context into a private ss, otherwise the let node may generate
    // within the function call arguments.
    std::ostringstream ss;

    for (size_t i = start; i < op->args.size() - end; i++) {
      if (i > start)
        ss << ", ";
      ss << this->PrintExpr(op->args[i]);
    }

    this->PrintIndent();
    this->stream << name << "(";
    this->stream << ss.str();
    this->stream << ");\n";
  };
  auto print_mbarrier_obj = [&](PrimExpr barrier_id) {
    std::ostringstream ss;
    if (barrier_id.as<IntImmNode>()) {
      // incase the barrier_id is an integer, we need to print the barrier_id as
      // an integer
      ss << mbarrier_name_ << "[" << barrier_id << "]";
    } else {
      // otherwise may be a T.get_mbarrier() call or BufferLoad Node
      // we need to print the barrier_id as a string
      ss << this->PrintExpr(barrier_id);
    }
    return ss.str();
  };
  if (op->op.same_as(tl::max_nan()) || op->op.same_as(tl::min_nan())) {
    ICHECK_EQ(op->args.size(), 2U);
    ICHECK(op->dtype.is_scalar());
    os << (op->op.same_as(tl::max_nan()) ? "tl::MaxOpNan{}("
                                         : "tl::MinOpNan{}(")
       << PrintExpr(op->args[0]) << ", " << PrintExpr(op->args[1]) << ")";
    return;
  }
  auto print_pointer = [&](const PrimExpr &expr) {
    if (const auto *call = expr.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_access_ptr())) {
        ICHECK_GE(call->args.size(), 3U);
        return "(" + PrintExpr(call->args[1]) + " + " +
               PrintExpr(call->args[2]) + ")";
      }
      if (call->op.same_as(builtin::address_of())) {
        ICHECK_EQ(call->args.size(), 1U);
        return "&(" + PrintExpr(call->args[0]) + ")";
      }
    }
    if (expr.as<BufferLoadNode>()) {
      return "&(" + PrintExpr(expr) + ")";
    }
    return PrintExpr(expr);
  };
  auto print_atomic_binary = [&](const char *name, bool returns_value) {
    ICHECK(op->args.size() == 2 || op->args.size() == 3)
        << name << " expects address, value, and an optional memory order";
    need___clang_tang_builtin_vars_h = true;
    if (std::string(name) == "atomicAdd") {
      DataType value_type = op->args[1].dtype();
      bool supported = (value_type.is_float() && value_type.bits() == 32) ||
                       (value_type.is_int() && value_type.bits() == 32) ||
                       (value_type.is_uint() && value_type.bits() == 32);
      ICHECK(supported)
          << "TANG atomicAdd only supports float32, int32, or uint32, but got "
          << value_type;
    }
    bool use_unsigned = op->args[1].dtype().is_uint();
    if (auto value = op->annotations.Get("uint_atomic")) {
      if (const auto *flag = value->as<IntImmNode>()) {
        use_unsigned = flag->value != 0;
      }
    }
    std::ostringstream call;
    call << name << "(";
    if (use_unsigned) {
      call << "(unsigned int*)(" << print_pointer(op->args[0])
           << "), (unsigned int)(" << PrintExpr(op->args[1]) << ")";
    } else {
      call << print_pointer(op->args[0]) << ", " << PrintExpr(op->args[1]);
    }
    call << ")";
    if (returns_value) {
      os << call.str();
    } else {
      PrintIndent();
      stream << call.str() << ";\n";
    }
  };
  if (op->op.same_as(builtin::ptx_cp_async())) {
    // NOT used, will be deleted in the future.
    std::string dst = this->PrintExpr(op->args[0]);
    std::string dst_offset = this->PrintExpr(op->args[1]);
    std::string src = this->PrintExpr(op->args[2]);
    std::string src_offset = this->PrintExpr(op->args[3]);
    std::string size = this->PrintExpr(op->args[4]);
    // use size of argument list to indicate whether or not to use predicated
    // cp.async
    if (op->args.size() == 5) {
      this->PrintIndent();
      this->stream << "tl::cp_async_gs<" << size << ">(" << dst << "+"
                   << dst_offset << ", " << src << "+" << src_offset << ");\n";
    } else {
      std::string condition = this->PrintExpr(op->args[5]);
      this->PrintIndent();
      this->stream << "tl::cp_async_gs_conditional<" << size << ">(" << dst
                   << "+" << dst_offset << ", " << src << "+" << src_offset
                   << ", " << condition << ");\n";
    }
  } else if (op->op.same_as(builtin::ptx_commit_group())) {
    // assert(false && "Not Used in STCU.");
  } else if (op->op.same_as(builtin::ptx_wait_group())) {
    // assert(false && "Not Used in STCU.");
  } else if (op->op.same_as(tl::atomic_add_elem_op())) {
    print_atomic_binary("atomicAdd", false);
    return;
  } else if (op->op.same_as(tl::atomic_add_ret_elem_op())) {
    print_atomic_binary("atomicAdd", true);
    return;
  } else if (op->op.same_as(tl::atomic_max_elem_op())) {
    print_atomic_binary("atomicMax", false);
    return;
  } else if (op->op.same_as(tl::atomic_max_ret_elem_op())) {
    print_atomic_binary("atomicMax", true);
    return;
  } else if (op->op.same_as(tl::atomic_min_elem_op())) {
    print_atomic_binary("atomicMin", false);
    return;
  } else if (op->op.same_as(tl::atomic_min_ret_elem_op())) {
    print_atomic_binary("atomicMin", true);
    return;
  } else if (op->op.same_as(tl::atomic_or_elem_op())) {
    print_atomic_binary("atomicOr", false);
    return;
  } else if (op->op.same_as(tl::atomic_sub_elem_op())) {
    print_atomic_binary("atomicSub", false);
    return;
  } else if (op->op.same_as(tl::atomic_exch_elem_op())) {
    print_atomic_binary("atomicExch", false);
    return;
  } else if (op->op.same_as(tl::atomic_inc_elem_op())) {
    print_atomic_binary("atomicInc", false);
    return;
  } else if (op->op.same_as(tl::atomic_dec_elem_op())) {
    print_atomic_binary("atomicDec", false);
    return;
  } else if (op->op.same_as(tl::atomic_and_elem_op())) {
    print_atomic_binary("atomicAnd", false);
    return;
  } else if (op->op.same_as(tl::atomic_xor_elem_op())) {
    print_atomic_binary("atomicXor", false);
    return;
  } else if (op->op.same_as(tl::atomic_cas_elem_op())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "atomic_cas_elem_op expects address, compare, value, memory order";
    need___clang_tang_builtin_vars_h = true;
    bool use_unsigned = op->args[1].dtype().is_uint();
    if (auto value = op->annotations.Get("uint_atomic")) {
      if (const auto *flag = value->as<IntImmNode>()) {
        use_unsigned = flag->value != 0;
      }
    }
    PrintIndent();
    stream << "atomicCAS(";
    if (use_unsigned) {
      stream << "(unsigned int*)(" << print_pointer(op->args[0])
             << "), (unsigned int)(" << PrintExpr(op->args[1])
             << "), (unsigned int)(" << PrintExpr(op->args[2]) << ")";
    } else {
      stream << print_pointer(op->args[0]) << ", " << PrintExpr(op->args[1])
             << ", " << PrintExpr(op->args[2]);
    }
    stream << ");\n";
    return;
  } else if (op->op.same_as(tl::atomic_load_elem_op())) {
    LOG(FATAL) << "TANG STCU S2 does not support atomic_load; the Tang 0.25 "
                  "compiler cannot select an atomic load instruction";
  } else if (op->op.same_as(tl::atomic_store_elem_op())) {
    LOG(FATAL) << "TANG STCU S2 does not support atomic_store; the Tang 0.25 "
                  "compiler cannot select an atomic store instruction";
  } else if (op->op.same_as(tl::atomic_addx2_elem_op()) ||
             op->op.same_as(tl::atomic_addx4_elem_op())) {
    ICHECK(op->args.size() == 2 || op->args.size() == 3)
        << "vector atomic add expects destination and source pointers";
    need___clang_tang_builtin_vars_h = true;
    std::string dst = print_pointer(op->args[0]);
    std::string src = print_pointer(op->args[1]);
    int lanes = op->op.same_as(tl::atomic_addx2_elem_op()) ? 2 : 4;
    auto dst_type = GetAtomicAccessPtrElementType(op->args[0]);
    ICHECK(dst_type.has_value())
        << "vector atomic add expects a typed destination access pointer";
    if (dst_type.value().is_float16() || dst_type.value().is_bfloat16()) {
      LOG(FATAL) << "TANG vector atomicAdd does not support float16 or "
                    "bfloat16; the local atomic contract has no packed-CAS "
                    "fallback";
    }
    for (int lane = 0; lane < lanes; ++lane) {
      PrintIndent();
      stream << "atomicAdd(" << dst << " + " << lane << ", *(" << src << " + "
             << lane << "));\n";
    }
    return;
  } else if (auto atomic_func = get_stcu_s2_atomic_func_name(op);
             atomic_func.has_value()) {
    need___clang_tang_builtin_vars_h = true;
    std::string func_name = atomic_func.value();
    if (func_name == "atomicAdd" || func_name == "atomicAddUint") {
      // FIXME:? if has 4 args, the last args[3] is mmeory order which is
      // droppped here.
      ICHECK(op->args.size() == 3 || op->args.size() == 4)
          << "atomicAdd expects 2 or 3 arguments (address, value, [memory "
             "order]), but got "
          << op->args.size() - 1;
      DataType val_type = op->args[1].dtype();
      bool is_uint32 = (val_type.is_uint() && val_type.bits() == 32);
      bool is_int32 = (val_type.is_int() && val_type.bits() == 32);
      bool is_float32 = (val_type.is_float() && val_type.bits() == 32);
      // Type check: atomicAdd supports float32, int32, uint32
      ICHECK(is_float32 || is_int32 || is_uint32)
          << "atomicAdd only supports float32, int32, or uint32, but got "
          << val_type;
      bool is_uint = (func_name == "atomicAddUint");
      if (is_uint) {
        os << "atomicAdd((unsigned int*)(&(";
        this->PrintExpr(op->args[1], os);
        os << ")), (unsigned int)(";
        this->PrintExpr(op->args[2], os);
        os << "))";
      } else {
        os << "atomicAdd(&(";
        this->PrintExpr(op->args[1], os);
        os << "), ";
        this->PrintExpr(op->args[2], os);
        os << ")";
      }
    } else if (func_name == "atomicSub" || func_name == "atomicSubUint") {
      ICHECK(op->args.size() == 3 || op->args.size() == 4)
          << "atomicSub expects 2 arguments (address and value), but got "
          << op->args.size() - 1;
      DataType val_type = op->args[1].dtype();
      bool is_uint32 = (val_type.is_uint() && val_type.bits() == 32);
      bool is_int32 = (val_type.is_int() && val_type.bits() == 32);
      // FIXME: drop last arg, arg[3] is memory order
      // Type check: atomicSub supports only int32 and uint32 (no float)
      ICHECK(is_int32 || is_uint32)
          << "atomicSub only supports int32 or uint32, but got " << val_type;

      bool is_uint = (func_name == "atomicSubUint");
      if (is_uint) {
        os << "atomicSub((unsigned int*)(&(";
        this->PrintExpr(op->args[1], os);
        os << ")), (unsigned int)(";
        this->PrintExpr(op->args[2], os);
        os << "))";
      } else {
        os << "atomicSub(&(";
        this->PrintExpr(op->args[1], os);
        os << "), ";
        this->PrintExpr(op->args[2], os);
        os << ")";
      }
    } else if (func_name == "atomicMax" || func_name == "atomicMaxUint" ||
               func_name == "atomicMin" || func_name == "atomicMinUint") {
      ICHECK(op->args.size() == 3 || op->args.size() == 4)
          << "atomicMax/atomicMin expects 2 arguments (address and value), but "
             "got "
          << op->args.size() - 1;
      // FIXME: drop last arg, arg[3] is memory order
      DataType val_dtype = op->args[2]->dtype;
      if (!(val_dtype.is_int() && val_dtype.bits() == 32) &&
          !(val_dtype.is_uint() && val_dtype.bits() == 32)) {
        LOG(FATAL)
            << "atomicMax on stcu s2 only supports int32 or uint32, but got "
            << val_dtype;
      }
      if (func_name == "atomicMax" || func_name == "atomicMaxUint") {
        os << "atomicMax";
      } else {
        os << "atomicMin";
      }
      if (func_name == "atomicMaxUint" || func_name == "atomicMinUint") {
        // Emit: atomicMax((unsigned int*)(&(address)), (unsigned int)(value))
        os << "((" << "unsigned int*" << ")(&(";
        this->PrintExpr(op->args[1], os);
        os << ")), (";
        os << "unsigned int" << ")(";
        this->PrintExpr(op->args[2], os);
        os << "))";
      } else {
        os << "(&(";
        this->PrintExpr(op->args[1], os);
        os << "), ";
        this->PrintExpr(op->args[2], os);
        os << ")";
      }
    } else if (func_name == "atomicExch" || func_name == "atomicExchUint") {
      // Check argument count: (ret_type, address, value)
      ICHECK(op->args.size() == 3 || op->args.size() == 4)
          << "atomicExch expects 2 arguments (address and value), but got "
          << op->args.size() - 1;
      DataType val_dtype = op->args[2]->dtype;
      DataType addr_dtype; // The dtype of the element pointed to by address
      addr_dtype = val_dtype;

      // Validate supported types for atomicExch
      bool is_supported = (val_dtype.is_int() && val_dtype.bits() == 32) ||
                          (val_dtype.is_uint() && val_dtype.bits() == 32) ||
                          (val_dtype.is_float() && val_dtype.bits() == 32);

      if (!is_supported) {
        LOG(FATAL) << "atomicExch only supports int32, uint32, and float32, "
                   << "but value argument has dtype: " << val_dtype;
      }

      // check that the return type matches val_dtype
      if (op->dtype != val_dtype) {
        LOG(WARNING)
            << "Return type (" << op->dtype << ") does not match value type ("
            << val_dtype
            << ") in atomicExch call. This may cause undefined behavior.";
      }

      if (func_name == "atomicExchUint") {
        // Force cast to unsigned int*
        os << "atomicExch((unsigned int*)(&(";
        this->PrintExpr(op->args[1], os);
        os << ")), static_cast<unsigned int>(";
        this->PrintExpr(op->args[2], os);
        os << "))";
      } else {
        // Emit: atomicExch(&(address_expr), value_expr)
        os << "atomicExch(&(";
        this->PrintExpr(op->args[1], os);
        os << "), ";
        this->PrintExpr(op->args[2], os);
        os << ")";
      }
    } else if (func_name == "atomicInc" || func_name == "atomicIncUint") {
      ICHECK(op->args.size() == 3 || op->args.size() == 4)
          << func_name << " expects 2 arguments (address and value), but got "
          << op->args.size() - 1;

      DataType val_dtype = op->args[2]->dtype;
      bool is_uint_variant = (func_name == "atomicIncUint");
      if (!(val_dtype.is_int() && val_dtype.bits() == 32) &&
          !(val_dtype.is_uint() && val_dtype.bits() == 32)) {
        LOG(FATAL)
            << "atomicIncUint requires 32-bit integer type (int32/uint32), got "
            << val_dtype;
      }
      if (is_uint_variant) {
        os << "atomicInc((unsigned int*)(&(";
        this->PrintExpr(op->args[1], os);
        os << ")), static_cast<unsigned int>(";
        this->PrintExpr(op->args[2], os);
        os << "))";
      } else {
        // Emit normally
        os << "atomicInc(&(";
        this->PrintExpr(op->args[1], os);
        os << "), ";
        this->PrintExpr(op->args[2], os);
        os << ")";
      }
    } else if (func_name == "atomicDec" || func_name == "atomicDecUint") {
      ICHECK(op->args.size() == 3 || op->args.size() == 4)
          << func_name << " expects 2 arguments (address and value), but got "
          << op->args.size() - 1;

      DataType val_dtype = op->args[2]->dtype;
      bool is_uint_variant = (func_name == "atomicDecUint");
      if (!(val_dtype.is_int() && val_dtype.bits() == 32) &&
          !(val_dtype.is_uint() && val_dtype.bits() == 32)) {
        LOG(FATAL) << "atomicDec requires int32/uint32, but got " << val_dtype;
      }
      if (is_uint_variant) {
        // Cast both pointer and value to unsigned int
        os << "atomicDec((unsigned int*)(&(";
        this->PrintExpr(op->args[1], os); // address
        os << ")), static_cast<unsigned int>(";
        this->PrintExpr(op->args[2], os); // value
        os << "))";
      } else {
        os << "atomicDec(&(";
        this->PrintExpr(op->args[1], os);
        os << "), ";
        this->PrintExpr(op->args[2], os);
        os << ")";
      }
    } else if (func_name == "atomicCAS" || func_name == "atomicCASUint") {
      // TIR call: T.call_extern(ret_type, "atomicCAS", addr, compare, val)
      ICHECK(op->args.size() == 4 || op->args.size() == 5)
          << func_name
          << " expects 3 arguments (address, compare, value), but got "
          << op->args.size() - 1;

      DataType val_dtype = op->args[3]->dtype; // value's dtype
      bool is_uint_variant = (func_name == "atomicCASUint");
      if (!(val_dtype.is_int() && val_dtype.bits() == 32) &&
          !(val_dtype.is_uint() && val_dtype.bits() == 32)) {
        LOG(FATAL)
            << "atomicCASUint requires 32-bit integer type (int32/uint32), got "
            << val_dtype;
      }
      if (op->dtype != val_dtype) {
        LOG(WARNING) << "Return type (" << op->dtype << ") != operand type ("
                     << val_dtype << ") in atomicCAS";
      }
      if (is_uint_variant) {
        // Emit with cast to unsigned int for all integer args
        os << "atomicCAS((unsigned int*)(&(";
        this->PrintExpr(op->args[1], os); // address
        os << ")), static_cast<unsigned int>(";
        this->PrintExpr(op->args[2], os); // compare
        os << "), static_cast<unsigned int>(";
        this->PrintExpr(op->args[3], os); // value
        os << "))";
      } else {
        os << "atomicCAS(&(";
        this->PrintExpr(op->args[1], os); // address
        os << "), ";
        this->PrintExpr(op->args[2], os); // compare
        os << ", ";
        this->PrintExpr(op->args[3], os); // value
        os << ")";
      }
    } else if (func_name == "atomicOr" || func_name == "atomicOrUint" ||
               func_name == "atomicXor" || func_name == "atomicXorUint" ||
               func_name == "atomicAnd" || func_name == "atomicAndUint") {

      ICHECK(op->args.size() == 3 || op->args.size() == 4)
          << "Atomic bitwise op expects 2 arguments (address and value), but "
             "got "
          << op->args.size() - 1;

      DataType val_type = op->args[1].dtype();
      // Only 32-bit unsigned int is natively supported by CUDA atomic bitwise
      // ops
      bool is_uint32 = (val_type.is_uint() && val_type.bits() == 32);
      bool is_int32 = (val_type.is_int() && val_type.bits() == 32);

      // Note: CUDA's atomicOr/And/Xor officially only support unsigned int
      // (32-bit). Some compilers allow signed int as an extension, but we
      // enforce uint32 for safety.
      ICHECK(is_uint32 || is_int32) << "Atomic bitwise operations (Or/Xor/And) "
                                       "only support int32 or uint32, but got "
                                    << val_type;

      // Determine base operation name (strip "Uint" suffix if present)
      std::string base_name;
      bool is_uint = false;
      if (func_name.find("Uint") != std::string::npos) {
        is_uint = true;
        if (func_name == "atomicOrUint")
          base_name = "atomicOr";
        else if (func_name == "atomicXorUint")
          base_name = "atomicXor";
        else if (func_name == "atomicAndUint")
          base_name = "atomicAnd";
        else
          LOG(FATAL) << "Unknown atomic bitwise function: " << func_name;
      } else {
        base_name = func_name;
      }

      if (is_uint) {
        // Emit cast to unsigned int* for all cases (CUDA requires it)
        os << base_name << "((unsigned int*)(&(";
        this->PrintExpr(op->args[1], os);
        os << ")), (unsigned int)(";
        this->PrintExpr(op->args[2], os);
        os << "))";
      } else {
        os << base_name << "((&(";
        this->PrintExpr(op->args[1], os);
        os << ")), (";
        this->PrintExpr(op->args[2], os);
        os << "))";
      }
    }
    return;
  } else if (op->op.same_as(builtin::tvm_tuple())) {
    // Handle tir.tvm_tuple: evaluate all args for side effects, return last one
    // This matches the semantics of tvm_tuple in TIR.
    ICHECK(!op->args.empty()) << "tvm_tuple must have at least one argument";
    // Emit all arguments except the last as standalone statements (for side
    // effects)
    for (size_t i = 0; i < op->args.size() - 1; ++i) {
      this->PrintExpr(op->args[i], os);
      os << ";\n"; // Force as statement to preserve side effect
    }
    // Return the last expression as the result value
    this->PrintExpr(op->args.back(), os);
    return;
  } else if (op->op.same_as(tl::pts_syncthreads())) {
    this->PrintIndent();
    this->stream << "__syncthreads();\n";
    return;
  } else if (op->op.same_as(tl::pts_load_async())) {
    // use size of argument list to indicate whether or not to use predicated
    // cp.async
    if (op->args.size() == 3) {
      CheckAsyncCopyOperandOrder(op, "pts_load_async");
      std::string dst = this->PrintExpr(op->args[0]);
      std::string src = this->PrintExpr(op->args[1]);
      const auto *size_imm = op->args[2].as<IntImmNode>();
      if (!size_imm) {
        // Non-constant size: emit simple async_load without chunk splitting.
        std::string size = this->PrintExpr(op->args[2]);
        this->PrintIndent();
        this->stream << "async_load(" << dst << ", " << src << ", " << size
                     << ");\n";
        return;
      }
      int size_val = static_cast<int>(size_imm->value);
      // Hardware only supports 4-byte async DMA; split wider copies into
      // multiple 4-byte calls with incremented pointer offsets.
      // When size > 4 (cop4 widened by InjectPTSAsyncCopy), emit inline asm
      // with async_load.u32.cop4 for 4 outstanding DMA transactions.
      // Pattern from taBLAS: "r" constraint on integer byte offsets.
      // Read cop4 preference from pass config (default false: opt-in for HW
      // that supports it).
      bool use_cop4 = tvm::transform::PassContext::Current()
                          ->GetConfig<Bool>(tvm::tl::kUseAsyncCop4, Bool(false))
                          .value()
                          ->value;
      // Extract global buffer VarNode from AddressOffset IR to determine the
      // PTX base register ([p0_p1], [p2_p3], ...) for inline asm, instead of
      // string-matching on the printed variable name.
      std::string glb_base = "[p0_p1]";
      std::string src_var = "A";
      bool glb_resolved = false;
      if (const auto *call = op->args[1].as<CallNode>()) {
        if (call->op.same_as(builtin::address_of()) && call->args.size() == 1) {
          if (const auto *load = call->args[0].as<BufferLoadNode>()) {
            const auto *var = load->buffer->data.get();
            auto it = func_param_index_.find(var);
            if (it != func_param_index_.end()) {
              int idx = it->second;
              glb_base = "[p" + std::to_string(idx * 2) + "_p" +
                         std::to_string(idx * 2 + 1) + "]";
              src_var = GetVarID(var);
              glb_resolved = true;
            }
          }
        }
      }
      // cop4 inline asm encodes the global base register (glb_base) and
      // subtracts src_var; both are only valid when the global operand was
      // resolved to a kernel parameter above. If not, emitting the asm would
      // silently use the wrong base register ([p0_p1]) / wrong subtrahend
      // ("A"), computing a wrong global address. Fail loudly instead.
      ICHECK(!use_cop4 || glb_resolved)
          << "cop4 async_load: could not resolve global source to a kernel "
             "parameter (expected address_of(BufferLoad) on a param buffer); "
             "refusing to emit inline asm with a default base register";
      int chunks = (size_val + 3) / 4;
      for (int c = 0; c < chunks; c++) {
        this->PrintIndent();
        if (use_cop4) {
          // taBLAS inline asm pattern: [glb_base] + glb_offset, [zero] +
          // shm_offset Both offsets are BYTE offsets, so use << [0] (no shift).
          // Use (char*) casts so pointer arithmetic is byte-granular regardless
          // of the underlying element type (fp16*, float*, etc.).
          std::string shm_ptr =
              (c == 0) ? dst
                       : "(char*)(" + dst + ") + " + std::to_string(c * 4);
          std::string glb_ptr =
              (c == 0) ? src
                       : "(char*)(" + src + ") + " + std::to_string(c * 4);
          // glb_off: byte offset from parameter base via pointer subtraction
          // shm_off: byte offset within shared memory (buf_shmem base = 0)
          this->stream << "asm volatile(\"async_load.u32.cop4 " << glb_base
                       << " + %0 << [0] + [0], [zero] + %1 << [0]\\n\\t\""
                       << " :: \"r\"((int)((char*)(" << glb_ptr << ") - (char*)"
                       << src_var << "))"
                       << ", \"r\"((int)(size_t)(" << shm_ptr << "))"
                       << " : \"memory\");\n";
        } else {
          int chunk_size = (c == chunks - 1 && size_val % 4) ? size_val % 4 : 4;
          // Byte-granular pointer arithmetic via (char*), matching the store
          // fallback below. The previous form (dst + c*2, (__fp16*)src + c*2)
          // used ELEMENT offsets and a hard-coded fp16 cast, so it only
          // computed the intended 4-byte-per-chunk stride for 2-byte dtypes;
          // fp32/int8 multi-element copies desynced the two sides and read/
          // wrote wrong addresses. c*4 is the byte offset of chunk c.
          std::string shm_ptr =
              (c == 0) ? dst
                       : "(char*)(" + dst + ") + " + std::to_string(c * 4);
          std::string glb_ptr =
              (c == 0) ? src
                       : "(char*)(" + src + ") + " + std::to_string(c * 4);
          this->stream << "async_load(" << shm_ptr << ", " << glb_ptr << ", "
                       << chunk_size << ");\n";
        }
      }
    } else {
      // Predicated async load (4 args: dst, src, bytes, predicate) is not
      // lowered here yet. This MUST NOT be a bare assert(): release builds
      // compile with -DNDEBUG, which deletes it, and control then falls
      // through to the `return` below -- silently dropping the whole DMA
      // statement instead of failing. LOG(FATAL) survives NDEBUG.
      LOG(FATAL) << "pts_load_async: predicated form (" << op->args.size()
                 << " args) is not supported by the TANG C codegen; expected 3 "
                    "args {dst_shared, src_global, bytes}";
    }
    return;
  } else if (op->op.same_as(tl::pts_store_async())) {
    // args: src (shared), dst (global), size
    if (op->args.size() == 3) {
      CheckAsyncCopyOperandOrder(op, "pts_store_async");
      std::string src = this->PrintExpr(op->args[0]);
      std::string dst = this->PrintExpr(op->args[1]);
      const auto *size_imm = op->args[2].as<IntImmNode>();
      if (!size_imm) {
        std::string size = this->PrintExpr(op->args[2]);
        this->PrintIndent();
        this->stream << "async_store(" << src << ", " << dst << ", " << size
                     << ");\n";
        return;
      }
      int size_val = static_cast<int>(size_imm->value);
      // Emit inline asm async_store.u32.cop4 for cop4 DMA (shared → global).
      // Same pattern as pts_load_async but destination is global, source is
      // shared. Extract global buffer VarNode from AddressOffset IR to
      // determine the PTX base register, instead of string-matching on the
      // printed var name.
      std::string glb_base = "[p4_p5]";
      std::string dst_var = "C";
      bool glb_resolved = false;
      if (const auto *call = op->args[1].as<CallNode>()) {
        if (call->op.same_as(builtin::address_of()) && call->args.size() == 1) {
          if (const auto *load = call->args[0].as<BufferLoadNode>()) {
            const auto *var = load->buffer->data.get();
            auto it = func_param_index_.find(var);
            if (it != func_param_index_.end()) {
              int idx = it->second;
              glb_base = "[p" + std::to_string(idx * 2) + "_p" +
                         std::to_string(idx * 2 + 1) + "]";
              dst_var = GetVarID(var);
              glb_resolved = true;
            }
          }
        }
      }
      // See the matching ICHECK in the pts_load_async path: a cop4 store with
      // an unresolved global base would emit inline asm using the default
      // base register ([p4_p5]) / subtrahend ("C") and write to the wrong
      // global address. Fail loudly rather than silently corrupt memory.
      bool use_cop4 = tvm::transform::PassContext::Current()
                          ->GetConfig<Bool>(tvm::tl::kUseAsyncCop4, Bool(false))
                          .value()
                          ->value;
      ICHECK(!use_cop4 || glb_resolved)
          << "cop4 async_store: could not resolve global destination to a "
             "kernel parameter (expected address_of(BufferLoad) on a param "
             "buffer); refusing to emit inline asm with a default base "
             "register";
      int chunks = (size_val + 3) / 4;
      for (int c = 0; c < chunks; c++) {
        this->PrintIndent();
        // Use (char*) casts for byte-granular pointer arithmetic.
        std::string shm_ptr =
            (c == 0) ? src : "(char*)(" + src + ") + " + std::to_string(c * 4);
        std::string glb_ptr =
            (c == 0) ? dst : "(char*)(" + dst + ") + " + std::to_string(c * 4);
        if (use_cop4) {
          this->stream << "asm volatile(\"async_store.u32.cop4 " << glb_base
                       << " + %0 << [0] + [0], [zero] + %1 << [0]\\n\\t\""
                       << " :: \"r\"((int)((char*)(" << glb_ptr << ") - (char*)"
                       << dst_var << "))"
                       << ", \"r\"((int)(size_t)(" << shm_ptr << "))"
                       << " : \"memory\");\n";
        } else {
          int chunk_size = (c == chunks - 1 && size_val % 4) ? size_val % 4 : 4;
          this->stream << "async_store(" << shm_ptr << ", " << glb_ptr << ", "
                       << chunk_size << ");\n";
        }
      }
    } else {
      // No else branch existed here: a predicated store (4 args) fell straight
      // through to the return below, silently dropping the entire DMA. The
      // injection site does push a predicate operand for store as well
      // (inject_pts_async_copy.cc), so this is reachable, not hypothetical.
      LOG(FATAL) << "pts_store_async: predicated form (" << op->args.size()
                 << " args) is not supported by the TANG C codegen; expected 3 "
                    "args {src_shared, dst_global, bytes}";
    }
    return;
  } else if (op->op.same_as(builtin::create_barriers())) {
    this->PrintIndent();
    int barrier_count = Downcast<IntImm>(op->args[0])->value;
    auto mbarrier_storage_name = mbarrier_name_ + "_mem";
    this->stream << "__shared__ uint64_t " << mbarrier_storage_name << "["
                 << barrier_count << "];\n";
    this->PrintIndent();
    this->stream << "auto " << mbarrier_name_ << " = reinterpret_cast<"
                 << mbarrier_dtype_ << "*>(" << mbarrier_storage_name << ");\n";
  } else if (op->op.same_as(tl::get_mbarrier())) {
    ICHECK_EQ(op->args.size(), 1);
    std::string barrier_id = this->PrintExpr(op->args[0]);
    os << mbarrier_name_ + "[" + barrier_id + "]";
  } else if (op->op.same_as(builtin::ptx_arrive_barrier())) {
    if (op->args.size() == 1) {
      this->PrintIndent();
      auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
      this->stream << mbarrier_obj << ".arrive();\n";
    } else if (op->args.size() == 3) {
      this->PrintIndent();
      auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
      auto cta_id = this->PrintExpr(op->args[1]);
      auto pred = this->PrintExpr(op->args[2]);
      this->stream << mbarrier_obj << ".arrive(" << cta_id << ", " << pred
                   << ");\n";
    } else {
      LOG(FATAL) << "Invalid parameter  for tl::arrive_barrier "
                 << op->args.size();
    }
  } else if (op->op.same_as(builtin::ptx_init_barrier_thread_count())) {
    ICHECK_EQ(op->args.size(), 2);
    this->PrintIndent();
    auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
    auto arrive_count = this->PrintExpr(op->args[1]);
    this->stream << mbarrier_obj << ".init(" << arrive_count << ");\n";
  } else if (op->op.same_as(builtin::ptx_arrive_barrier_expect_tx())) {
    if (op->args.size() == 2) {
      this->PrintIndent();
      auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
      auto transaction_bytes = this->PrintExpr(op->args[1]);
      this->stream << mbarrier_obj << ".arrive_and_expect_tx("
                   << transaction_bytes << ");\n";
    } else if (op->args.size() == 4) {
      this->PrintIndent();
      auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
      auto transaction_bytes = this->PrintExpr(op->args[1]);
      auto cta_id = this->PrintExpr(op->args[2]);
      auto pred = this->PrintExpr(op->args[3]);
      this->stream << mbarrier_obj << ".arrive_and_expect_tx("
                   << transaction_bytes << ", " << cta_id << ", " << pred
                   << ");\n";
    } else {
      LOG(FATAL) << "Invalid parameter  for tl::arrive_barrier_expect_tx "
                 << op->args.size();
    }
  } else if (op->op.same_as(builtin::ptx_cp_async_barrier())) {
    print_extern_call_stmt("tl::mbarrier_cp_async_arrive");
  } else if (op->op.same_as(tl::ptx_fence_barrier_init())) {
    print_extern_call_stmt("tl::fence_barrier_init");
  } else if (op->op.same_as(tl::ptx_cp_async_barrier_noinc())) {
    print_extern_call_stmt("tl::mbarrier_cp_async_arrive_noinc");
  } else if (op->op.same_as(tl::mbarrier_expect_tx())) {
    ICHECK_EQ(op->args.size(), 2);
    this->PrintIndent();
    auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
    auto transaction_bytes = this->PrintExpr(op->args[1]);
    this->stream << mbarrier_obj << ".expect_transaction(" << transaction_bytes
                 << ");\n";
  } else if (op->op.same_as(tl::mbarrier_wait_parity())) {
    ICHECK_EQ(op->args.size(), 2);
    this->PrintIndent();
    auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
    auto phase = this->PrintExpr(op->args[1]);
    this->stream << mbarrier_obj << ".wait(" << phase << ");\n";
  } else if (op->op.same_as(tl::no_set_max_nreg())) {
    return;
  } else if (op->op.same_as(tl::tma_load())) {
    std::ostringstream ss;
    ICHECK_GE(op->args.size(), 2);
    auto eviction_policy =
        this->eviction_policy_names_
            [op->args[op->args.size() - 1].as<IntImmNode>()->value];
    // Simplify the code by using the default eviction policy
    if (eviction_policy != "EVICT_NORMAL") {
      ss << "tl::tma_load<tl::CacheHintSm90::" << eviction_policy << ">(";
    } else {
      ss << "tl::tma_load(";
    }
    auto desc = op->args[0];
    ss << this->PrintExpr(desc) << ", ";
    ss << print_mbarrier_obj(op->args[1]) << ", ";
    for (size_t i = 2; i < op->args.size() - 1; i++) {
      if (i > 2)
        ss << ", ";
      ss << this->PrintExpr(op->args[i]);
    }
    ss << ");\n";
    this->PrintIndent();
    this->stream << ss.str();
  } else if (op->op.same_as(tl::tma_load_im2col())) {
    std::stringstream ss;
    auto eviction_policy =
        this->eviction_policy_names_
            [op->args[op->args.size() - 1].as<IntImmNode>()->value];
    if (eviction_policy != "EVICT_NORMAL") {
      ss << "tl::tma_load_im2col<tl::CacheHintSm90::" << eviction_policy << ">";
    } else {
      ss << "tl::tma_load_im2col";
    }
    print_extern_call_stmt(ss.str(), 0, 1);
  } else if (op->op.same_as(tl::tma_store())) {
    std::stringstream ss;
    auto need_reduce = op->args[op->args.size() - 2].as<IntImmNode>()->value;
    if (need_reduce) {
      print_extern_call_stmt("tl::tma_store_add", 0, 2);
      return;
    }
    auto eviction_policy =
        this->eviction_policy_names_
            [op->args[op->args.size() - 1].as<IntImmNode>()->value];
    if (eviction_policy != "EVICT_NORMAL") {
      ss << "tl::tma_store<tl::CacheHintSm90::" << eviction_policy << ">";
    } else {
      ss << "tl::tma_store";
    }
    print_extern_call_stmt(ss.str(), 0, 1);
  } else if (op->op.same_as(tl::ptx_ldmatrix())) {
    int trans = Downcast<IntImm>(op->args[0])->value;
    int num = Downcast<IntImm>(op->args[1])->value;
    std::string func_name = "tl::ptx_ldmatrix_x" + std::to_string(num);
    if (trans == 1)
      func_name += "_trans";
    print_extern_call_stmt(func_name, 2);
  } else if (op->op.same_as(tl::ptx_stmatrix())) {
    int trans = Downcast<IntImm>(op->args[0])->value;
    int num = Downcast<IntImm>(op->args[1])->value;
    std::string func_name = "tl::ptx_stmatrix_x" + std::to_string(num);
    if (trans == 1)
      func_name += "_trans";
    print_extern_call_stmt(func_name, 2);
  } else if (op->op.same_as(tl::fence_proxy_async())) {
    print_extern_call_stmt("tl::fence_proxy_async");
  } else if (op->op.same_as(tl::tma_store_arrive())) {
    print_extern_call_stmt("tl::tma_store_arrive");
  } else if (op->op.same_as(tl::tma_store_wait())) {
    print_extern_call_stmt("tl::tma_store_wait<0>");
  } else if (op->op.same_as(tl::set_max_nreg())) {
    this->PrintIndent();
    int nreg = Downcast<IntImm>(op->args[0])->value;
    int is_inc = Downcast<IntImm>(op->args[1])->value;
    std::string func_name =
        is_inc ? "tl::warpgroup_reg_alloc" : "tl::warpgroup_reg_dealloc";
    this->stream << func_name << "<" << std::to_string(nreg) << ">();\n";
  } else if (op->op.same_as(tl::wait_wgmma())) {
    this->PrintIndent();
    int num_mma = Downcast<IntImm>(op->args[0])->value;
    this->stream << "tl::wait_wgmma<" << std::to_string(num_mma) << ">();\n";
  } else if (op->op.same_as(tl::pack_b16())) {
    os << "__pack_half2(" << this->PrintExpr(op->args[0]) << ", "
       << this->PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::sync_grid())) {
    this->need_cooperative_groups_ = true;
    this->PrintIndent();
    // this->stream << "cooperative_groups::this_grid().sync();\n";
    this->stream << "sync_grids(&bar0);\n";
  } else if (op->op.same_as(tl::sync_warp())) {
    this->PrintIndent();
    this->stream << "__syncwarp(";
    if (!op->args.empty()) {
      this->stream << this->PrintExpr(op->args[0]);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::tang_fill_fragment())) {
    // TODO: implement matrix multiplication using the WMMA APIs.
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 6U);
    os << "stpu::tmma::fill_fragment(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ")";
  } else if (op->op.same_as(builtin::tvm_load_matrix_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::load_matrix_sync(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[6], os);
    os << ")";
  } else if (op->op.same_as(builtin::tvm_store_matrix_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::store_matrix_sync(";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[6], os);
    if (const StringImmNode *str = op->args[7].as<StringImmNode>()) {
      os << ", nvcuda::wmma::mem_" << str->value;
    } else {
      LOG(FATAL) << "Invalid parameters";
    }
    os << ")";
  } else if (op->op.same_as(builtin::tvm_mma_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::mma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(tl::loop_break())) {
    this->PrintIndent();
    this->stream << "break;\n";
  } else if (op->op.same_as(builtin::reinterpret())) {
    DataType tgt_dtype = op->dtype;
    DataType src_dtype = op->args[0]->dtype;
    PrimExpr value = op->args[0];

    // Handle float4_e2m1fn reinterpret
    if (!src_dtype.is_float4_e2m1fn() && !tgt_dtype.is_float4_e2m1fn()) {
      return CodeGenC::VisitExpr_(op, os);
    }
    if (src_dtype == tgt_dtype || tgt_dtype.lanes() * tgt_dtype.bits() ==
                                      src_dtype.lanes() * src_dtype.bits()) {
      return CodeGenC::VisitExpr_(op, os);
    }
    ICHECK_EQ(tgt_dtype.lanes(), src_dtype.lanes())
        << "E2M1 float4 reinterpret expects source and target to have the same "
           "number of lanes. "
        << "Source dtype: " << src_dtype << ", Target dtype: " << tgt_dtype;
    ICHECK_EQ(tgt_dtype.bytes(), src_dtype.bytes())
        << "E2M1 float4 reinterpret expects source and target to have the same "
           "number of bytes. "
        << "Source dtype: " << src_dtype << ", Target dtype: " << tgt_dtype;

    int lanes = tgt_dtype.lanes();

    int ssa_scope = BeginScope();
    if (lanes == 1) {
      // The case of lane=1 is same as the normal reinterpret,
      // except that we allow the src and dst dtype to have different number of
      // bits.
      std::string rhs = SSAGetID(PrintExpr(value), src_dtype);
      os << "(*(";
      this->PrintType(tgt_dtype, os);
      os << " *)(&(" << rhs << ")))";
    } else if (lanes == 2) {
      if (tgt_dtype.is_float4_e2m1fn()) {
        // We view the source as an uint16, and then extract bits of two fp4
        // numbers, and finally reinterpret the result as fp4x2.
        value = tirx::Call(DataType::UInt(16), tirx::builtin::reinterpret(),
                           {value});
        tirx::Var temp_var("temp_var", DataType::UInt(16));
        value =
            tirx::Let(temp_var, value,
                      tirx::Cast(DataType::UInt(8),
                                 (temp_var & IntImm(DataType::UInt(16), 0xF)) |
                                     ((temp_var >> 4) &
                                      IntImm(DataType::UInt(16), 0xF0))));
      } else {
        value = tirx::Cast(DataType::UInt(16),
                           tirx::Call(DataType::UInt(8),
                                      tirx::builtin::reinterpret(), {value}));
        tirx::Var temp_var("temp_var", DataType::UInt(16));
        value =
            tirx::Let(temp_var, value,
                      (temp_var & IntImm(DataType::UInt(16), 0xF)) |
                          ((temp_var & IntImm(DataType::UInt(16), 0xF0)) << 4));
      }
      os << PrintExpr(
          tirx::Call(tgt_dtype, tirx::builtin::reinterpret(), {value}));
    } else if (lanes == 4) {
      if (tgt_dtype.is_float4_e2m1fn()) {
        // We view the source as an uint32, and then extract bits of four fp4
        // numbers, and finally reinterpret the result as fp4x4.
        value = tirx::Call(DataType::UInt(32), tirx::builtin::reinterpret(),
                           {value});
        tirx::Var temp_var("temp_var", DataType::UInt(32));
        value = tirx::Let(
            temp_var, value,
            tirx::Cast(
                DataType::UInt(16),
                (temp_var & IntImm(DataType::UInt(32), 0xF)) |
                    ((temp_var >> 4) & IntImm(DataType::UInt(32), 0xF0)) |
                    ((temp_var >> 8) & IntImm(DataType::UInt(32), 0xF00)) |
                    ((temp_var >> 12) & IntImm(DataType::UInt(32), 0xF000))));
      } else {
        value = tirx::Cast(DataType::UInt(32),
                           tirx::Call(DataType::UInt(16),
                                      tirx::builtin::reinterpret(), {value}));
        tirx::Var temp_var("temp_var", DataType::UInt(32));
        value = tirx::Let(
            temp_var, value,
            (temp_var & IntImm(DataType::UInt(32), 0xF)) |
                ((temp_var & IntImm(DataType::UInt(32), 0xF0)) << 4) |
                ((temp_var & IntImm(DataType::UInt(32), 0xF00)) << 8) |
                ((temp_var & IntImm(DataType::UInt(32), 0xF000)) << 12));
      }
      os << PrintExpr(
          tirx::Call(tgt_dtype, tirx::builtin::reinterpret(), {value}));
    } else {
      LOG(FATAL) << "Invalid number of lanes for float4_e2m1fn reinterpret: "
                 << lanes;
    }
    EndScope(ssa_scope);
  } else if (op->op.same_as(builtin::thread_return())) {
    os << "return";
  } else if (op->op.same_as(tl::tl_gemm())) {
    ICHECK(op->args.size() == 4) << "tl_gemm expects 4 arguments <op_instance, "
                                    "A_ptr, B_ptr, C_ptr>, but got "
                                 << op->args.size();
    auto op_instance = Downcast<StringImm>(op->args[0]);
    this->PrintCallExtern(GetType(tvm::ffi::GetRef<PrimExpr>(op)),
                          op_instance->value, op->args, true, os);
  } else if (op->op.same_as(tl::tl_tang_gemm())) {
    // 4 args for most TANG GEMM variants <op_instance, A_ptr, B_ptr, C_ptr>;
    // the TC-Gen5 variant appends an optional 5th runtime arg (clear_accum);
    // the block-scaled TC-Gen5 variant passes 8 args
    // <op_instance, A_ptr, B_ptr, C_ptr, SFA_ptr, SFB_ptr, SF_tmem,
    // clear_accum>.
    ICHECK(op->args.size() == 4 || op->args.size() == 5 || op->args.size() == 8)
        << "tl_tang_gemm expects 4, 5 or 8 arguments <op_instance, A_ptr, "
           "B_ptr, C_ptr[, clear_accum | SFA_ptr, SFB_ptr, SF_tmem, "
           "clear_accum]>, but got "
        << op->args.size();
    auto op_instance = Downcast<StringImm>(op->args[0]);
    enable_sparse_gemm_ = false;
    this->PrintCallExtern(GetType(tvm::ffi::GetRef<PrimExpr>(op)),
                          op_instance->value, op->args, true, os);
  } else if (op->op.same_as(tl::tl_gemm_sp())) {
    ICHECK(op->args.size() == 5)
        << "tl_gemm_sp expects 5 arguments <op_instance, A_ptr, B_ptr, C_ptr, "
           "E_ptr>, but got "
        << op->args.size();
    auto op_instance = Downcast<StringImm>(op->args[0]);
    enable_sparse_gemm_ = true;
    this->PrintCallExtern(GetType(tvm::ffi::GetRef<PrimExpr>(op)),
                          op_instance->value, op->args, true, os);
  } else if (op->op.same_as(tl::tang_tcgen05_mma_ss())) {
    // TANG stcuv2 TC-Gen5 MMA (A/B from shared descriptors, C in TMEM)
    // Maps to tang::ptx::mma<enable_input_d>(d_addr, a_desc, b_desc, i_desc)
    ICHECK_EQ(op->args.size(), 14U)
        << "tang_tcgen05_mma_ss expects 14 arguments";
    std::string kind_dtype = Downcast<StringImm>(op->args[0])->value;
    std::string a_desc = this->PrintExpr(op->args[1]);
    std::string a_off = this->PrintExpr(op->args[2]);
    std::string b_desc = this->PrintExpr(op->args[3]);
    std::string b_off = this->PrintExpr(op->args[4]);
    std::string c_ref = this->PrintExpr(op->args[5]);
    std::string c_off = this->PrintExpr(op->args[6]);
    std::string desc_val = this->PrintExpr(op->args[7]);
    std::string scale_out = this->PrintExpr(op->args[8]);
    std::string mask0 = this->PrintExpr(op->args[9]);
    std::string mask1 = this->PrintExpr(op->args[10]);
    std::string mask2 = this->PrintExpr(op->args[11]);
    std::string mask3 = this->PrintExpr(op->args[12]);
    bool enable_ws = Downcast<Bool>(op->args[13])->value;

    need_tcgen05_common_h_ = true;
    this->PrintIndent();
    // i_desc: use pre-computed MMA descriptor
    std::string i_desc = desc_val;
    // d_addr: TMEM address for C
    this->stream << "tang::ptx::mma<false>("
                 << "static_cast<tang::ptx::tmem_ptr>(" << c_ref << ") + "
                 << c_off << ", "
                 << "static_cast<uint64_t>(" << a_desc << ") + " << a_off
                 << ", "
                 << "static_cast<uint64_t>(" << b_desc << ") + " << b_off
                 << ", "
                 << "static_cast<uint32_t>(" << i_desc << ")"
                 << ");\n";
  } else if (op->op.same_as(tl::tang_tcgen05_mma_ts())) {
    // TANG stcuv2 TC-Gen5 MMA (A from TMEM, B from shared, C in TMEM)
    // Maps to tang::ptx::mma_atmem<enable_input_d>(d_addr, a_tmem, b_desc,
    // i_desc)
    ICHECK_EQ(op->args.size(), 13U)
        << "tang_tcgen05_mma_ts expects 13 arguments";
    std::string kind_dtype = Downcast<StringImm>(op->args[0])->value;
    std::string a_ref = this->PrintExpr(op->args[1]);
    std::string a_off = this->PrintExpr(op->args[2]);
    std::string b_desc = this->PrintExpr(op->args[3]);
    std::string b_off = this->PrintExpr(op->args[4]);
    std::string c_ref = this->PrintExpr(op->args[5]);
    std::string c_off = this->PrintExpr(op->args[6]);
    std::string desc_val = this->PrintExpr(op->args[7]);
    std::string scale_out = this->PrintExpr(op->args[8]);
    std::string mask0 = this->PrintExpr(op->args[9]);
    std::string mask1 = this->PrintExpr(op->args[10]);
    std::string mask2 = this->PrintExpr(op->args[11]);
    std::string mask3 = this->PrintExpr(op->args[12]);

    need_tcgen05_common_h_ = true;
    this->PrintIndent();
    this->stream << "tang::ptx::mma_atmem<false>("
                 << "static_cast<tang::ptx::tmem_ptr>(" << c_ref << ") + "
                 << c_off << ", "
                 << "static_cast<uint32_t>(" << a_ref << ") + " << a_off << ", "
                 << "static_cast<uint64_t>(" << b_desc << ") + " << b_off
                 << ", "
                 << "static_cast<uint32_t>(" << desc_val << ")"
                 << ");\n";
  } else if (op->op.same_as(tl::tang_init_tensor_memory())) {
    // TANG stcuv2 TMEM allocation
    // Maps to tang::ptx::tc_alloc<N>(ptr) where N is compile-time constant
    ICHECK_EQ(op->args.size(), 2U)
        << "tang_init_tensor_memory expects 2 arguments";
    std::string ptr = this->PrintExpr(op->args[0]);
    std::string num_cols = this->PrintExpr(op->args[1]);
    need_tcgen05_common_h_ = true;
    this->PrintIndent();
    this->stream << "tang::ptx::tc_alloc<" << num_cols << ">(" << ptr << ");\n";
  } else if (op->op.same_as(tl::tang_deallocate_tensor_memory())) {
    // TANG stcuv2 TMEM deallocation
    // Maps to tang::ptx::tc_dealloc<N>(ptr) where N is compile-time constant
    ICHECK_EQ(op->args.size(), 2U)
        << "tang_deallocate_tensor_memory expects 2 arguments";
    std::string ptr = this->PrintExpr(op->args[0]);
    std::string num_cols = this->PrintExpr(op->args[1]);
    need_tcgen05_common_h_ = true;
    this->PrintIndent();
    this->stream << "tang::ptx::tc_dealloc<" << num_cols << ">(" << ptr
                 << ");\n";
  } else if (op->op.same_as(tl::tang_tcgen05_mma_arrive())) {
    ICHECK_EQ(op->args.size(), 1U)
        << "tang_tcgen05_mma_arrive expects 1 argument";
    need_tcgen05_common_h_ = true;
    this->PrintIndent();
    this->stream << "// tcgen05_mma_arrive(" << this->PrintExpr(op->args[0])
                 << ");\n";
  } else if (op->op.same_as(tl::tang_tmem_ld())) {
    // TANG stcuv2 TMEM load: ldt_32x32b_xN(out, taddr)
    // args: [out_ref, taddr, num_elements]
    ICHECK_EQ(op->args.size(), 3U) << "tang_tmem_ld expects 3 arguments";
    std::string out_ref = this->PrintExpr(op->args[0]);
    std::string taddr = this->PrintExpr(op->args[1]);
    int num_elems = Downcast<IntImm>(op->args[2])->value;
    need_tcgen05_common_h_ = true;
    this->PrintIndent();
    this->stream << "ldt_32x32b_x" << num_elems << "_pack("
                 << "*reinterpret_cast<uint32_t (*)[" << num_elems << "]>("
                 << out_ref << "), " << taddr << ");\n";
    this->PrintIndent();
    this->stream << "tang::ptx::fence_ldt();\n";
  } else if (op->op.same_as(tl::tang_tmem_ld_16x256b())) {
    // TANG stcuv2 warp-collective TMEM load: ldt_16x256b_x8_pack(out, taddr)
    // args: [out_ref, taddr, num_elems] — num_elems ignored, always x8
    ICHECK_EQ(op->args.size(), 3U)
        << "tang_tmem_ld_16x256b expects 3 arguments";
    std::string out_ref = this->PrintExpr(op->args[0]);
    std::string taddr = this->PrintExpr(op->args[1]);
    need_tcgen05_common_h_ = true;
    this->PrintIndent();
    this->stream << "ldt_16x256b_x8_pack("
                 << "*reinterpret_cast<uint32_t (*)[32]>(" << out_ref << "), "
                 << taddr << ");\n";
    this->PrintIndent();
    this->stream << "tang::ptx::fence_ldt();\n";
  } else if (op->op.same_as(tl::tang_tmem_ld_16x256b_x16())) {
    // TANG stcuv2 warp-collective TMEM load (fp32): ldt_16x256b_x16(out, taddr)
    // args: [out_ref, taddr, num_elems] — num_elems determines _out[64]
    ICHECK_EQ(op->args.size(), 3U)
        << "tang_tmem_ld_16x256b_x16 expects 3 arguments";
    std::string out_ref = this->PrintExpr(op->args[0]);
    std::string taddr = this->PrintExpr(op->args[1]);
    int num_elems = Downcast<IntImm>(op->args[2])->value;
    need_tcgen05_common_h_ = true;
    this->PrintIndent();
    this->stream << "ldt_16x256b_x16("
                 << "*reinterpret_cast<uint32_t (*)[" << num_elems << "]>("
                 << out_ref << "), " << taddr << ");\n";
    this->PrintIndent();
    this->stream << "tang::ptx::fence_ldt();\n";
  } else if (op->op.same_as(tl::tang_tmem_drain_16x256b_to_global())) {
    // TANG stcuv2 TMEM→global drain.
    // args: [tmem_base_val, global_access_ptr, BM, BN, out_dtype_zero]
    ICHECK_EQ(op->args.size(), 5U)
        << "tang_tmem_drain_16x256b_to_global expects 5 arguments";
    std::string tmem_base = this->PrintExpr(op->args[0]);
    std::string global_ref = this->PrintExpr(op->args[1]);
    int BM = Downcast<IntImm>(op->args[2])->value;
    int BN = Downcast<IntImm>(op->args[3])->value;
    // The 32-bit accumulator in TMEM is fp32 for a float MMA and int32 for an
    // integer MMA. Reinterpret each lane as that accumulator type, then the
    // store narrows/converts to whatever the C buffer holds (fp16/bf16/int32).
    // An integer C buffer ⇒ integer accumulator; otherwise it is fp32.
    std::string acc_ty = op->args[4].dtype().is_int() ? "int32_t" : "float";
    need_tcgen05_common_h_ = true;
    // Generate step-16 warp-collective drain loop matching the reference:
    //   uint32_t _out[64];
    //   if ((int)threadIdx.x < 32) {
    //     for (int i = 0; i < BM; i += 16) {
    //       ldt_16x256b_x16(_out, C_tmem[0] + ((i << 16) | 0));
    //       fence_ldt();
    //       for (int _c = 0; _c < 16; ++_c) {
    //         4 stores per _c: C[(i+tid/4)*BN+...] =
    //         reinterpret_cast<float>(_out[...])
    //       }
    //     }
    //   }
    this->PrintIndent();
    this->stream << "uint32_t _out[64];\n";
    this->PrintIndent();
    this->stream << "if ((int)threadIdx.x < 32) {\n";
    int scope1 = BeginScope();
    this->PrintIndent();
    this->stream << "uint32_t _tid = __laneid();\n";
    this->PrintIndent();
    this->stream << "for (int i = 0; i < " << BM << "; i += 16) {\n";
    int scope2 = BeginScope();
    this->PrintIndent();
    this->stream << "ldt_16x256b_x16(_out, " << tmem_base
                 << " + ((i << 16) | 0));\n";
    this->PrintIndent();
    this->stream << "tang::ptx::fence_ldt();\n";
    this->PrintIndent();
    this->stream << "for (int _c = 0; _c < 16; ++_c) {\n";
    int scope3 = BeginScope();
    this->PrintIndent();
    this->stream << global_ref << "[(i+_tid/4)*" << BN
                 << " + 8*_c+(_tid%4)*2] = "
                 << "reinterpret_cast<" << acc_ty << "&>(_out[4*_c]);\n";
    this->PrintIndent();
    this->stream << global_ref << "[(i+_tid/4)*" << BN
                 << " + 8*_c+(_tid%4)*2+1] = "
                 << "reinterpret_cast<" << acc_ty << "&>(_out[4*_c+1]);\n";
    this->PrintIndent();
    this->stream << global_ref << "[(i+_tid/4+8)*" << BN
                 << " + 8*_c+(_tid%4)*2] = "
                 << "reinterpret_cast<" << acc_ty << "&>(_out[4*_c+2]);\n";
    this->PrintIndent();
    this->stream << global_ref << "[(i+_tid/4+8)*" << BN
                 << " + 8*_c+(_tid%4)*2+1] = "
                 << "reinterpret_cast<" << acc_ty << "&>(_out[4*_c+3]);\n";
    this->EndScope(scope3);
    this->stream << "}\n";
    this->EndScope(scope2);
    this->stream << "}\n";
    this->EndScope(scope1);
    this->stream << "}\n";
  } else if (op->op.same_as(tl::tang_tmem_st())) {
    // TANG stcuv2 TMEM store: stt_32x32b_xN(in, taddr)
    // args: [in_ref, taddr, num_elements]
    ICHECK_EQ(op->args.size(), 3U) << "tang_tmem_st expects 3 arguments";
    std::string in_ref = this->PrintExpr(op->args[0]);
    std::string taddr = this->PrintExpr(op->args[1]);
    int num_elems = Downcast<IntImm>(op->args[2])->value;
    need_tcgen05_common_h_ = true;
    this->PrintIndent();
    this->stream << "stt_32x32b_x" << num_elems << "(" << in_ref << ", "
                 << taddr << ");\n";
  } else if (op->op.same_as(tl::tang_tmem_fence())) {
    // TANG stcuv2 TMEM fence
    // args: [fence_group] (optional, default fg_ldt_default)
    need_tcgen05_common_h_ = true;
    int fg = 3; // default fg_ldt_default
    if (op->args.size() >= 1) {
      fg = Downcast<IntImm>(op->args[0])->value;
    }
    this->PrintIndent();
    this->stream << "tang::ptx::fence_tmem<" << fg << ">();\n";
  } else if (op->op.same_as(tl::tang_cp_async_bulk())) {
    // TANG stcuv2 swizzled async bulk copy (global ↔ shared).
    // args: [direction, dst, src, rows, col_bytes]
    // direction: 0 = g2s (global → shared), 1 = s2g (shared → global)
    // Emits the self-contained, mbarrier-synchronized helper from
    // tl_templates/tang/copy_fcp_g_s.h, which wraps tang::ptx::fcpg2s / fcps2g
    // (sw128bytes_atom32bytes) with a separate global row pitch so K-subtiles
    // of a wider matrix copy correctly.
    // args: [direction, dst, src, rows, smem_col_bytes, gmem_row_bytes]
    ICHECK_EQ(op->args.size(), 6U) << "tang_cp_async_bulk expects 6 arguments";
    int direction = Downcast<IntImm>(op->args[0])->value;
    std::string dst = this->PrintExpr(op->args[1]);
    std::string src = this->PrintExpr(op->args[2]);
    std::string rows = this->PrintExpr(op->args[3]);
    std::string col_bytes = this->PrintExpr(op->args[4]);
    std::string gmem_row_bytes = this->PrintExpr(op->args[5]);
    need_cp_async_bulk_h_ = true;
    this->PrintIndent();
    if (direction == 0) {
      // g2s: dst = shared, src = global
      this->stream << "tl::tang_bulk_g2s_sw128a32("
                   << "reinterpret_cast<void *>(" << dst << "), "
                   << "reinterpret_cast<const void *>(" << src << "), "
                   << "(uint32_t)(" << rows << "), "
                   << "(uint32_t)(" << col_bytes << "), "
                   << "(uint32_t)(" << gmem_row_bytes << ")"
                   << ");\n";
    } else {
      // s2g: dst = global, src = shared
      this->stream << "tl::tang_bulk_s2g_sw128a32("
                   << "reinterpret_cast<void *>(" << dst << "), "
                   << "reinterpret_cast<const void *>(" << src << "), "
                   << "(uint32_t)(" << rows << "), "
                   << "(uint32_t)(" << col_bytes << "), "
                   << "(uint32_t)(" << gmem_row_bytes << ")"
                   << ");\n";
    }
  } else if (op->op.same_as(tl::tang_cp_async_bulk_sw())) {
    // TANG stcuv2 swizzled async bulk copy with explicit swizzle mode.
    // args: [direction, dst, src, rows, col_bytes, gmem_row_bytes,
    //        mbarrier, swizzle_mode]
    ICHECK_EQ(op->args.size(), 8U)
        << "tang_cp_async_bulk_sw expects 8 arguments";
    int direction = Downcast<IntImm>(op->args[0])->value;
    std::string dst = this->PrintExpr(op->args[1]);
    std::string src = this->PrintExpr(op->args[2]);
    std::string rows = this->PrintExpr(op->args[3]);
    std::string col_bytes = this->PrintExpr(op->args[4]);
    std::string gmem_row_bytes = this->PrintExpr(op->args[5]);
    std::string mbarrier = this->PrintExpr(op->args[6]);
    int swizzle_mode = Downcast<IntImm>(op->args[7])->value;
    need_cp_async_bulk_h_ = true;
    this->PrintIndent();
    // Map swizzle mode to the template function suffix
    const char *sw_suffix = "sw128a32"; // default
    switch (swizzle_mode) {
    case 4:
      sw_suffix = "sw128a8";
      break;
    case 5:
      sw_suffix = "sw128a16";
      break;
    case 6:
      sw_suffix = "sw128a32";
      break;
    case 7:
      sw_suffix = "sw128a64";
      break;
    case 8:
      sw_suffix = "sw64a8";
      break;
    case 9:
      sw_suffix = "sw64a16";
      break;
    case 10:
      sw_suffix = "sw64a32";
      break;
    case 1:
      sw_suffix = "sw32a8";
      break;
    case 2:
      sw_suffix = "sw32a16";
      break;
    default:
      sw_suffix = "sw128a32";
      break;
    }
    if (direction == 0) {
      this->stream << "tl::tang_bulk_g2s_" << sw_suffix << "("
                   << "reinterpret_cast<void *>(" << dst << "), "
                   << "reinterpret_cast<const void *>(" << src << "), "
                   << "(uint32_t)(" << rows << "), "
                   << "(uint32_t)(" << col_bytes << "), "
                   << "(uint32_t)(" << gmem_row_bytes << ")"
                   << ");\n";
    } else {
      this->stream << "tl::tang_bulk_s2g_" << sw_suffix << "("
                   << "reinterpret_cast<void *>(" << dst << "), "
                   << "reinterpret_cast<const void *>(" << src << "), "
                   << "(uint32_t)(" << rows << "), "
                   << "(uint32_t)(" << col_bytes << "), "
                   << "(uint32_t)(" << gmem_row_bytes << ")"
                   << ");\n";
    }
  } else if (op->op.same_as(tl::tang_fence_tc())) {
    // TANG stcuv2 TC fence: fence_tc<fg>()
    need_cp_async_bulk_h_ = true;
    int fg = 0; // default fg_tc_default
    if (op->args.size() >= 1) {
      fg = Downcast<IntImm>(op->args[0])->value;
    }
    this->PrintIndent();
    this->stream << "tang::ptx::fence_tc<" << fg << ">();\n";
  } else if (op->op.same_as(tl::tang_fence_tc_arrive())) {
    // TANG stcuv2 TC fence arrive: fence_tc_arrive(bar)
    ICHECK_EQ(op->args.size(), 1U) << "tang_fence_tc_arrive expects 1 argument";
    std::string bar = this->PrintExpr(op->args[0]);
    need_cp_async_bulk_h_ = true;
    this->PrintIndent();
    this->stream << "tang::ptx::fence_tc_arrive(" << bar << ");\n";
  } else if (op->op.same_as(tl::tang_fence_g2s_arrive())) {
    // TANG stcuv2 G2S fence arrive: fence_g2s_arrive(bar)
    ICHECK_EQ(op->args.size(), 1U)
        << "tang_fence_g2s_arrive expects 1 argument";
    std::string bar = this->PrintExpr(op->args[0]);
    need_cp_async_bulk_h_ = true;
    this->PrintIndent();
    this->stream << "tang::ptx::fence_g2s_arrive(" << bar << ");\n";
  } else if (op->op.same_as(tl::tang_sync_wait())) {
    // TANG stcuv2 sync wait: sync_wait(bar, pcnt, ccnt)
    ICHECK_EQ(op->args.size(), 3U) << "tang_sync_wait expects 3 arguments";
    std::string bar = this->PrintExpr(op->args[0]);
    std::string pcnt = this->PrintExpr(op->args[1]);
    std::string ccnt = this->PrintExpr(op->args[2]);
    this->PrintIndent();
    this->stream << "sync_wait(" << bar << ", " << pcnt << ", " << ccnt
                 << ");\n";
  } else if (op->op.same_as(tl::tang_sync_arrive())) {
    // TANG stcuv2 sync arrive: sync_arrive(bar)
    ICHECK_EQ(op->args.size(), 1U) << "tang_sync_arrive expects 1 argument";
    std::string bar = this->PrintExpr(op->args[0]);
    this->PrintIndent();
    this->stream << "sync_arrive(" << bar << ");\n";
  } else if (op->op.same_as(tl::get_lane_idx())) {
    ICHECK_LE(op->args.size(), 1)
        << "tl.get_lane_idx expects at most one argument <warp_size>.";
    os << "tl::get_lane_idx(";
    if (!op->args.empty()) {
      os << PrintExpr(op->args[0]);
    }
    os << ")";
  } else if (op->op.same_as(tl::get_warp_idx_sync())) {
    ICHECK_LE(op->args.size(), 1)
        << "tl.get_warp_idx_sync expects at most one argument <warp_size>.";
    os << "tl::get_warp_idx_sync(";
    if (!op->args.empty()) {
      os << PrintExpr(op->args[0]);
    }
    os << ")";
  } else if (op->op.same_as(tl::get_warp_idx())) {
    ICHECK_LE(op->args.size(), 1)
        << "tl.get_warp_idx expects at most one argument <warp_size>.";
    os << "tl::get_warp_idx(";
    if (!op->args.empty()) {
      os << PrintExpr(op->args[0]);
    }
    os << ")";
  } else if (op->op.same_as(tl::get_warp_group_idx())) {
    ICHECK_LE(op->args.size(), 2)
        << "tl.get_warp_group_idx expects <warp_size, warps_per_group>.";
    os << "tl::get_warp_group_idx(";
    for (size_t i = 0; i < op->args.size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << PrintExpr(op->args[i]);
    }
    os << ")";
  } else if (op->op.same_as(tl::tl_shuffle_elect())) {
    os << "tl::tl_shuffle_elect<" << PrintExpr(op->args[0]) << ">()";
  } else if (op->op.same_as(tl::rng_init())) {
    // The state variable is declared at function scope by the
    // AddFunction pre-scan; here we only emit the initialization call.
    // See src/tl_templates/tang/rng.h for algorithms and the bit-exactness
    // caveat (algorithms match curand; initialisation derives from
    // splitmix64 rather than curand's closed-source skip-ahead).
    ICHECK(rng_state_declared_)
        << "Tang RNG: rng_init was not registered by the AddFunction pre-scan";
    ICHECK(!rng_initialized_)
        << "Tang RNG: only one rng_init is supported per PrimFunc.";
    ICHECK_EQ(op->args.size(), 4U);
    PrintIndent();
    stream << "tl::rng_init_" << tang_random_generator_state_type_ << "(&"
           << tang_random_generator_state_ << ", " << PrintExpr(op->args[0])
           << ", " << PrintExpr(op->args[1]) << ", " << PrintExpr(op->args[2])
           << ");\n";
    rng_initialized_ = true;
  } else if (op->op.same_as(tl::rng_rand())) {
    ICHECK(rng_initialized_) << "Tang RNG: rng_rand called before rng_init.";
    os << "tl::rng_rand_" << tang_random_generator_state_type_ << "(&"
       << tang_random_generator_state_ << ")";
  } else if (op->op.same_as(tl::rng_rand_float())) {
    ICHECK(rng_initialized_)
        << "Tang RNG: rng_rand_float called before rng_init.";
    std::string dist = op->args[0].as<StringImmNode>()->value;
    ICHECK(dist == "uniform" || dist == "normal")
        << "Tang rng_rand_float only supports uniform or normal distribution, "
           "got "
        << dist;
    ICHECK_EQ(op->dtype.bits(), 32)
        << "Tang RNG does not support float64 (S2 has no native double); "
           "call rng_rand_float(bit=32) instead.";
    os << "tl::rng_" << dist << "_float_" << tang_random_generator_state_type_
       << "(&" << tang_random_generator_state_ << ")";
  } else if (op->op.same_as(tl::__exp())) {
    TANGFastMath math_func;
    std::string func_name = math_func(op->dtype, "exp");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__exp10())) {
    TANGFastMath math_func;
    std::string func_name = math_func(op->dtype, "exp10");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__log())) {
    TANGFastMath math_func;
    std::string func_name = math_func(op->dtype, "log");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__log2())) {
    TANGFastMath math_func;
    std::string func_name = math_func(op->dtype, "log2");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__log10())) {
    TANGFastMath math_func;
    std::string func_name = math_func(op->dtype, "log10");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__tan())) {
    TANGFastMath math_func;
    std::string func_name = math_func(op->dtype, "tan");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__cos())) {
    TANGFastMath math_func;
    std::string func_name = math_func(op->dtype, "cos");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__sin())) {
    TANGFastMath math_func;
    std::string func_name = math_func(op->dtype, "sin");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::ieee_add())) {
    TANGIEEEMath math_func;
    std::string rounding_mode = Downcast<StringImm>(op->args[2])->value;
    std::string func_name = math_func(op->dtype, "fadd", rounding_mode);
    os << func_name << "(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::ieee_sub())) {
    TANGIEEEMath math_func;
    std::string rounding_mode = Downcast<StringImm>(op->args[2])->value;
    std::string func_name = math_func(op->dtype, "fsub", rounding_mode);
    os << func_name << "(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::ieee_mul())) {
    TANGIEEEMath math_func;
    std::string rounding_mode = Downcast<StringImm>(op->args[2])->value;
    std::string func_name = math_func(op->dtype, "fmul", rounding_mode);
    os << func_name << "(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::ieee_fmaf())) {
    TANGIEEEMath math_func;
    std::string rounding_mode = Downcast<StringImm>(op->args[3])->value;
    std::string func_name = math_func(op->dtype, "fmaf", rounding_mode);
    os << func_name << "(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ", " << PrintExpr(op->args[2]) << ")";
  } else if (op->op.same_as(tl::ieee_frcp())) {
    TANGIEEEMath math_func;
    std::string rounding_mode = Downcast<StringImm>(op->args[1])->value;
    std::string func_name = math_func(op->dtype, "frcp", rounding_mode);
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::ieee_fsqrt())) {
    TANGIEEEMath math_func;
    std::string rounding_mode = Downcast<StringImm>(op->args[1])->value;
    std::string func_name = math_func(op->dtype, "fsqrt", rounding_mode);
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::ieee_frsqrt())) {
    TANGIEEEMath math_func;
    std::string func_name = math_func(op->dtype, "frsqrt", "rn");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::ieee_fdiv())) {
    TANGIEEEMath math_func;
    std::string rounding_mode = Downcast<StringImm>(op->args[2])->value;
    std::string func_name = math_func(op->dtype, "fdiv", rounding_mode);
    os << func_name << "(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::match_any_sync())) {
    os << "__match_any_sync(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::shfl_sync())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_sync expects <mask, value, src_lane, width>.";
    os << "__shfl_sync(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ", " << PrintExpr(op->args[2]) << ", "
       << PrintExpr(op->args[3]) << ")";
  } else if (op->op.same_as(tl::shfl_xor_sync())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_xor_sync expects <mask, value, lane_mask, width>.";
    os << "__shfl_xor_sync(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ", " << PrintExpr(op->args[2]) << ", "
       << PrintExpr(op->args[3]) << ")";
  } else if (op->op.same_as(tl::shfl_down_sync())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_down_sync expects <mask, value, delta, width>.";
    os << "__shfl_down_sync(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ", " << PrintExpr(op->args[2]) << ", "
       << PrintExpr(op->args[3]) << ")";
  } else if (op->op.same_as(tl::shfl_up_sync())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_up_sync expects <mask, value, delta, width>.";
    os << "__shfl_up_sync(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ", " << PrintExpr(op->args[2]) << ", "
       << PrintExpr(op->args[3]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_sum())) {
    os << "tl::warp_reduce_sum(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_max())) {
    os << "tl::warp_reduce_max(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_min())) {
    os << "tl::warp_reduce_min(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_bitand())) {
    os << "tl::warp_reduce_bitand(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_bitor())) {
    os << "tl::warp_reduce_bitor(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__ldg())) {
    // Explicit read-only cached load. Preferred form: __ldg(BufferLoad(...)).
    // Fallback form: __ldg(buffer, index)
    const BufferLoadNode *bl = nullptr;
    if (!op->args.empty()) {
      bl = op->args[0].as<BufferLoadNode>();
    }
    if (bl == nullptr) {
      LOG(FATAL) << "T.__ldg expects a BufferLoad as the first argument.";
    }
    const BufferNode *buffer = bl->buffer.get();
    ICHECK_EQ(bl->indices.size(), 1)
        << "T.__ldg currently supports flattened 1D buffer accesses.";
    PrimExpr base = bl->indices[0];
    // Emit __ldg(&buffer_ref)
    auto buffer_ref = this->GetBufferRef(op->dtype, buffer, base);
    os << "__ldg(&(" << buffer_ref << "))";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangTANG::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == s_tir::attr::fragment_shape) {
    const VarNode *buffer = op->node.as<VarNode>();
    const StringImmNode *shape_str = op->value.as<StringImmNode>();
    fragment_shapes[buffer] = shape_str->value;
  } else if (op->attr_key == s_tir::attr::fragment_layout) {
    const VarNode *buffer = op->node.as<VarNode>();
    const StringImmNode *layout_str = op->value.as<StringImmNode>();
    fragment_layouts[buffer] = layout_str->value;
  } else if (op->attr_key == s_tir::attr::async_commit_queue_scope) {
    const IntImmNode *queue_id = op->value.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0)
        << "For TANG, the index of an async queue must be 0.";
    this->VisitStmt(op->body);
    auto commit_group = Call(DataType::Void(), builtin::ptx_commit_group(), {});
    this->VisitExpr(commit_group, this->stream);
    return;
  } else if (op->attr_key == s_tir::attr::async_wait_queue_scope) {
    auto wait_attrs = GetAsyncWaitAttributes(op);
    auto queue_id = wait_attrs.first.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0)
        << "For TANG, the index of an async queue must be 0.";
    auto wait_cnt = wait_attrs.second;
    auto wait_group =
        Call(DataType::Void(), builtin::ptx_wait_group(), {wait_cnt});
    this->VisitExpr(wait_group, this->stream);
    auto inner = op->body.as<AttrStmtNode>();
    ICHECK(inner);
    this->VisitStmt(inner->body);
    return;
  } else if (op->attr_key == "threadblock_swizzle_pattern") {
    this->PrintIndent();
    std::string func_name;
    int panel_size = 0;
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(tirx::builtin::tvm_tuple()) &&
          call->args.size() >= 2) {
        const auto *name_node = call->args[0].as<StringImmNode>();
        const auto *size_node = call->args[1].as<IntImmNode>();
        ICHECK(name_node && size_node) << "threadblock_swizzle_pattern expects "
                                          "tvm_tuple(device_func, panel_size)";
        func_name = name_node->value;
        panel_size = static_cast<int>(size_node->value);
      }
    }
    ICHECK(!func_name.empty() && panel_size > 0)
        << "threadblock_swizzle_pattern: failed to extract func_name and "
           "panel_size";
    this->stream << "const dim3 blockIdx = tl::" << func_name << "<"
                 << panel_size << ">();\n";
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == tirx::attr::storage_alignment) {
    const VarNode *v = op->node.as<VarNode>();
    if (v && op->value.as<IntImmNode>()) {
      int alignment = static_cast<int>(op->value.as<IntImmNode>()->value);
      if (is_no_op(op->body)) {
        // Kernel parameter: emit __builtin_assume_aligned in function body.
        // The parameter var is already registered by AllocVarID in
        // AddFunction/PrintFunctionSignature, so look it up with GetVarID —
        // calling AllocVarID again trips its ICHECK(!var_idmap_.count(v))
        // "SSA form dup" and aborts codegen.
        this->PrintIndent();
        std::string vid = GetVarID(v);
        this->stream << vid << " = (decltype(" << vid
                     << "))__builtin_assume_aligned(" << vid << ", "
                     << alignment << ");\n";
      } else {
        // Shared/local buffer: store alignment to emit __align__ on the alloc
        alloc_storage_alignment_[v] = alignment;
        this->VisitStmt(op->body);
      }
    } else if (v) {
      this->VisitStmt(op->body);
    }
    return;
  } else if (op->attr_key == "pragma_unroll_factor") {
    const IntImmNode *factor = op->value.as<IntImmNode>();
    ICHECK(factor);
    unroll_factor[op->node.as<VarNode>()] = Downcast<IntImm>(factor);
  }

  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangTANG::VisitStmt_(const AllocBufferNode *op) {
  std::string vid = AllocVarID(op->buffer->data.get());
  this->PrintIndent();
  std::string scope = GetPtrStorageScope(op->buffer->data);
  const VarNode *buffer = op->buffer->data.as<VarNode>();
  DataType alloc_dtype = op->buffer->dtype;
  if (scope.find("wmma.") == 0) {
    if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b") {
      ICHECK(
          alloc_dtype == DataType::Float(16) ||
          alloc_dtype == DataType::Int(8) || alloc_dtype == DataType::UInt(8) ||
          alloc_dtype == DataType::Int(4) || alloc_dtype == DataType::UInt(4) ||
          alloc_dtype == DataType::Int(1) ||
          alloc_dtype == DataType::BFloat(16))
          << "Matrix_a and matrix_b only support half or char or unsigned char "
          << "or uint4 or int4 or int1 type for now";
    } else {
      ICHECK(alloc_dtype == DataType::Float(16) ||
             alloc_dtype == DataType::Float(32) ||
             alloc_dtype == DataType::Int(32))
          << "Accumulator only support half, float and int type for now";
    }
    PrintWmmaScope(scope, alloc_dtype, buffer, stream);
  } else {
    PrintStorageScope(scope, stream);
    // Emit alignment for shared buffers when specified via
    // T.alloc_buffer(align=N). shared.dyn already has hardcoded alignment
    // (512/1024) via PrintStorageScope; only override scope=="shared". Check
    // the AttrStmt-derived map first, then fall back to the AllocBuffer node
    // annotations (e.g. merged shmem).
    if (buffer && scope == "shared") {
      int alignment = 0;
      auto align_it = alloc_storage_alignment_.find(buffer);
      if (align_it != alloc_storage_alignment_.end()) {
        alignment = align_it->second;
      } else {
        auto annot_it = op->annotations.find(tirx::attr::storage_alignment);
        if (annot_it != op->annotations.end()) {
          alignment =
              static_cast<int>(Downcast<IntImm>((*annot_it).second)->value);
        }
      }
      if (alignment > 0) {
        stream << "__align__(" << alignment << ") ";
      }
    }
    PrintType(alloc_dtype, stream);
  }

  if (scope == "shared.dyn") {
    Target cur_target = Target::Current(/*allow_not_defined=*/true);
    bool is_stcuv2 = cur_target.defined() && tl::TargetTangIsSTCUV2(cur_target);
    if (is_stcuv2) {
      // S3 (stcuv2): reserve the buffer statically with a fixed size so the
      // TensorCore gets a 512-byte-aligned base (see PrintStorageScope above).
      auto opt_size = GetRef<AllocBuffer>(op).ConstantAllocationSize();
      ICHECK(opt_size.has_value())
          << "TANG requires constant allocation size for "
          << op->buffer->data->name_hint;
      size_t constant_size = static_cast<size_t>(opt_size.value());
      stream << ' ' << vid << '[' << constant_size << "];\n";
    } else {
      // S2 (stcu): `extern __shared__` arrays are unsized; their byte count
      // comes from the launch's dynamic shared memory parameter.
      stream << ' ' << vid << "[];\n";
    }
  } else {
    auto opt_size = GetRef<AllocBuffer>(op).ConstantAllocationSize();
    ICHECK(opt_size.has_value())
        << "TANG requires constant allocation size for "
        << op->buffer->data->name_hint;
    size_t constant_size = static_cast<size_t>(opt_size.value());
    if (scope.find("wmma.") == 0) {
      constant_size = GetWmmaFragmentSize(scope, buffer, constant_size);
    }
    if ((alloc_dtype == DataType::Int(4) || alloc_dtype == DataType::UInt(4) ||
         alloc_dtype == DataType::Int(1)) &&
        scope == "shared") {
      constant_size = constant_size / (32 / alloc_dtype.bits());
    }
    if (scope == "shared" || scope == "shared.tmem" ||
        scope == "shared.tmem_addr") {
      stream << ' ' << vid << '[' << constant_size << "];\n";
    } else if (scope == "shared.barrier") {
      auto v_id_mem = vid + "_mem";
      stream << ' ' << v_id_mem << "[" << constant_size << "];\n";
      PrintIndent();
      stream << "auto " << vid << " = reinterpret_cast<" << mbarrier_dtype_
             << "*>(" << v_id_mem << ");\n";
    } else if (scope == "local") {
      stream << ' ' << vid << '[' << constant_size << "];\n";
    } else if (scope == "local.var") {
      PrimExpr init = tirx::make_const(alloc_dtype, 0);
      auto init_it = op->annotations.find(tl::attr::kLocalVarInit);
      if (init_it != op->annotations.end()) {
        PrimExpr user_init = Downcast<PrimExpr>((*init_it).second);
        if (!user_init.dtype().is_void() && user_init.dtype() != alloc_dtype) {
          user_init = tirx::Cast(alloc_dtype, user_init);
        }
        init = user_init;
      }
      stream << ' ' << vid << " = " << PrintExpr(init) << ";\n";
    } else {
      ICHECK(false) << "Unsupported scope: " << scope;
    }
  }

  RegisterHandleType(op->buffer->data.get(), alloc_dtype);
}

void CodeGenTileLangTANG::VisitStmt_(const EvaluateNode *op) {
  if (is_const_int(op->value))
    return;
  const CallNode *call = op->value.as<CallNode>();
  if (call && call->op.same_as(builtin::tvm_global_barrier_kinit())) {
    PrintIndent();
    stream << "__shared__ unsigned " << vid_global_barrier_expect_ << ";\n";
    PrintIndent();
    stream << "if (threadIdx.x == 0) {\n";
    PrintIndent();
    stream << "  " << vid_global_barrier_expect_ << " = 0;\n";
    PrintIndent();
    stream << "}\n";
  }
  if (call && (call->op.same_as(tvm::tl::device_assert()))) {
    std::string cond = PrintExpr(call->args[0]);
    this->PrintIndent();
    stream << "device_assert(" << cond << ");\n";
  } else if (call && call->op.same_as(tvm::tl::device_assert_with_msg())) {
    std::string cond = PrintExpr(call->args[0]);
    std::string msg_expr = PrintExpr(call->args[1]);
    this->PrintIndent();
    stream << "device_assert_with_msg(" << cond << ", " << msg_expr << ");\n";
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenTileLangTANG::VisitExpr_(const RampNode *op, std::ostream &os) {
  int lanes = static_cast<int>(Downcast<IntImm>(op->lanes)->value);
  ICHECK_LE(lanes, 4) << "Translate Ramp Node " << tvm::ffi::GetRef<Ramp>(op)
                      << " with " << lanes << " lanes is not allowed.";
  os << "(make_";
  PrintType(op->dtype, os);
  os << "(";
  for (int i = 0; i < lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    if (i != lanes - 1)
      os << ", ";
  }
  os << "))";
}

void CodeGenTileLangTANG::VisitExpr_(const BufferLoadNode *op,
                                     std::ostream &os) { // NOLINT(*)
  ICHECK_EQ(op->indices.size(), 1)
      << "Load from non-flat memory not supported.";
  ICHECK(!op->predicate.defined())
      << "Predicated buffer load is not supported.";

  DataType value_dtype = op->dtype;
  PrimExpr index = op->indices[0];
  Var buffer_var = op->buffer->data;
  DataType element_dtype = op->buffer->dtype;

  int lanes = op->dtype.lanes();
  // declare type.
  if (value_dtype.lanes() == element_dtype.lanes()) {
    std::string ref = GetBufferRef(op->dtype, op->buffer.get(), index);
    HandleVolatileLoads(ref, op, os);
  } else {
    bool can_vector_load = false;
    arith::PVar<PrimExpr> base;
    if (arith::ramp(base, 1, op->dtype.lanes()).Match(index)) {
      const RampNode *ramp = index.as<RampNode>();
      ICHECK(ramp);
      can_vector_load = true;
      // arith::ModularSet me = arith::Analyzer().modular_set(ramp->base);
      // The condition: {k * coeff + base} divisible by the alignment for any k
      // if (me->coeff % op->dtype.lanes() == 0 && me->base % op->dtype.lanes()
      // == 0) {
      //   can_vector_load = true;
      // }
    }

    if (value_dtype.is_float4_e2m1fn() && lanes != 1) {
      // A float4_e2m1fn element has 4 bits, which is an incomplete byte.
      // So we cannot vector load it.
      can_vector_load = false;
    }
    if (can_vector_load) {
      std::string ref = GetVecLoad(op->dtype, op->buffer.get(), base.Eval());
      HandleVolatileLoads(ref, op, os);
    } else {
      std::ostringstream svalue_expr;
      std::string sindex = SSAGetID(PrintExpr(index), index.dtype());
      std::string vid = GetVarID(buffer_var.get());
      DataType elem_type = op->dtype.element_of();
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value_temp;
        if (!HandleTypeMatch(buffer_var.get(), elem_type)) {
          value_temp << "((";
          if (buffer_var.get()->dtype.is_handle()) {
            auto it = alloc_storage_scope_.find(buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, value_temp);
            }
          }
          PrintType(elem_type, value_temp);
          value_temp << "*)" << vid << ')';
        } else {
          value_temp << vid;
        }
        value_temp << '[';
        PrintVecElemLoad(sindex, index.dtype(), i, value_temp);
        value_temp << ']';
        PrintVecElemLoadExpr(op->dtype, i, value_temp.str(), svalue_expr);
      }
      os << svalue_expr.str();
    }
  }
}

void CodeGenTileLangTANG::VisitExpr_(const BroadcastNode *op,
                                     std::ostream &os) { // NOLINT(*)
  int lanes = static_cast<int>(Downcast<IntImm>(op->lanes)->value);
  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 8) {
    if (lanes == 32) {
      std::string v = PrintExpr(op->value);
      std::string packed_byte = "((unsigned long long)((unsigned char)(" + v +
                                ")) * 0x0101010101010101ULL)";
      if (op->dtype.is_uint()) {
        os << "make_ulonglong4(";
      } else {
        os << "make_longlong4(";
      }
      for (int i = 0; i < 4; ++i) {
        if (i != 0)
          os << ", ";
        if (op->dtype.is_uint()) {
          os << packed_byte;
        } else {
          os << "((long long)" << packed_byte << ")";
        }
      }
      os << ")";
      return;
    }

    const int64_t *p = as_const_int(op->value);
    if (p) {
      if (lanes == 4) {
        // make_int8x4
        ICHECK(p);
        int64_t v = *p & 0xFF;
        v = (v << 24) | (v << 16) | (v << 8) | v;
        if (op->dtype.is_uint()) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
        return;
      }
    }
  }

  if (op->dtype.is_float16()) {
    std::string v = PrintExpr(op->value);
    os << "make_";
    PrintType(op->dtype, os);
    os << '(';
    if (lanes <= 8) {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0)
          os << ", ";
        os << "__pack_half2(" << v << ", " << v << ")";
      }
    } else {
      for (int i = 0; i < lanes / 4; ++i) {
        if (i != 0)
          os << ", ";
        os << "tl::pack_float16x4(" << v << ", " << v << ", " << v << ", " << v
           << ")";
      }
    }
    os << ')';
    return;
  }

  if (op->dtype.is_bfloat16()) {
    std::string v = PrintExpr(op->value);
    os << "make_";
    PrintType(op->dtype, os);
    os << '(';
    if (lanes <= 8) {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0)
          os << ", ";
        os << "__pack_bfloat162(" << v << ", " << v << ")";
      }
    } else {
      for (int i = 0; i < lanes / 4; ++i) {
        if (i != 0)
          os << ", ";
        os << "tl::pack_bfloat16x4(" << v << ", " << v << ", " << v << ", " << v
           << ")";
      }
    }
    os << ')';
    return;
  }

  if (op->dtype.is_float() && op->dtype.bits() == 32 &&
      op->dtype.lanes() == 8) {
    std::string v = PrintExpr(op->value);
    os << "make_ulonglong4(";
    for (int i = 0; i < 4; ++i) {
      if (i != 0)
        os << ", ";
      os << "*(unsigned long long*)&make_float2(" << v << ", " << v << ")";
    }
    os << ')';
    return;
  }

  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 4) {
    bool fail = false;
    const int64_t *p = as_const_int(op->value);
    ICHECK(p);
    int64_t v = *p & 0xF;

    if (lanes == 4) {
      v = (v << 12) | (v << 8) | (v << 4) | v;
      if (op->dtype.is_uint()) {
        os << "(uint16_t)" << v;
      } else {
        os << "(int16_t)" << v;
      }
    } else {
      v = (v << 28) | (v << 24) | (v << 20) | (v << 16) | (v << 12) | (v << 8) |
          (v << 4) | v;
      if (lanes == 8) {
        if (op->dtype.is_uint()) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
      } else if (lanes == 16 || lanes == 32) {
        os << "make_";
        PrintType(op->dtype, os);
        os << '(';
        for (int i = 0; i < lanes / 8; ++i) {
          if (i != 0)
            os << ", ";
          if (op->dtype.is_uint()) {
            os << "(uint)" << v;
          } else {
            os << "(int)" << v;
          }
        }
        os << ')';
      } else {
        fail = true;
      }
    }

    if (!fail) {
      return;
    }
  }

  std::string v = PrintExpr(op->value);
  os << "make_";
  PrintType(op->dtype, os);
  os << '(';
  for (int i = 0; i < lanes; ++i) {
    if (i != 0)
      os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenTileLangTANG::VisitExpr_(const ShuffleNode *op,
                                     std::ostream &os) { // NOLINT(*)
  if ((op->dtype.is_float16() || op->dtype.is_bfloat16()) &&
      op->dtype.lanes() == 2 && op->indices.size() == 2) {
    std::vector<std::string> concat_vec;
    for (const PrimExpr &vec : op->vectors) {
      std::string vec_value = PrintExpr(vec);
      if (vec.dtype().is_scalar()) {
        concat_vec.push_back(vec_value);
      } else {
        for (int i = 0; i < vec.dtype().lanes(); ++i) {
          std::ostringstream elem;
          PrintVecElemLoad(vec_value, vec.dtype(), i, elem);
          concat_vec.push_back(elem.str());
        }
      }
    }

    os << "make_uint1(";
    os << (op->dtype.is_bfloat16() ? "__pack_bfloat162(" : "__pack_half2(");
    for (size_t i = 0; i < op->indices.size(); ++i) {
      ICHECK(op->indices[i]->IsInstance<IntImmNode>())
          << "TANG vector construction requires constant Shuffle indices";
      int64_t index = Downcast<IntImm>(op->indices[i])->value;
      ICHECK_LT(index, concat_vec.size());
      if (i != 0)
        os << ", ";
      os << concat_vec[index];
    }
    os << "))";
    return;
  }

  CodeGenC::VisitExpr_(op, os);
}

inline void PrintConst(const FloatImmNode *op, std::ostream &os,
                       CodeGenTileLangTANG *p) { // NOLINT(*)
  // Type code is kBFloat/kFloat16
  // which is indeed CUTLASS supported types currently
  if (op->dtype.is_float16()) {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      if (op->value < 0) {
        temp << "-";
      }
      temp << "(";
      p->PrintType(op->dtype, temp);
      temp << ")";
      temp << "TANGRT_INF_FP16";
      p->need___clang_tang_fp16_h = true;
    } else if (std::isnan(op->value)) {
      temp << "TANGRT_NAN_FP16";
      p->need___clang_tang_fp16_h = true;
    } else {
      p->PrintType(op->dtype, temp);
      temp << '(' << std::hexfloat << op->value << 'f';
      temp << "/*" << std::scientific << op->value << "*/";
      temp << ')';
    }
    p->MarkConst(temp.str());
    os << temp.str();
    return;
  }
  if (op->dtype.is_bfloat16()) {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      if (op->value < 0) {
        temp << "-";
      }
      temp << "(";
      p->PrintType(op->dtype, temp);
      temp << ")";
      temp << "TANGRT_INF_BF16";
      p->need___clang_tang_bf16_h = true;
    } else if (std::isnan(op->value)) {
      temp << "TANGRT_NAN_BF16";
      p->need___clang_tang_bf16_h = true;
    } else {
      p->PrintType(op->dtype, temp);
      temp << '(' << std::hexfloat << op->value << 'f';
      temp << "/*" << std::scientific << op->value << "*/";
      temp << ')';
    }
    p->MarkConst(temp.str());
    os << temp.str();
    return;
  }
  // Type code is kFloat8_e5m2 or kE4M4Float
  if (op->dtype.is_float8() || op->dtype.is_float4()) {
    p->PrintType(op->dtype, os);
    os << '(' << std::hexfloat << op->value << 'f';
    os << "/*" << std::scientific << op->value << "*/";
    os << ')';
    return;
  }
  // Type code is kFloat64/kFloat32 (kFloat16 is handled above)
  switch (op->dtype.bits()) {
  case 64:
  case 32: {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      if (op->value < 0) {
        temp << "-";
      }
      temp << ((op->dtype.bits() == 32) ? "TANG_RT_INF_F" : "TANG_RT_INF");
      p->need___clang_tang_builtin_vars_h = true;
    } else if (std::isnan(op->value)) {
      temp << ((op->dtype.bits() == 32) ? "TANG_RT_NAN_F" : "TANG_RT_NAN");
      p->need___clang_tang_builtin_vars_h = true;
    } else {
      temp << std::hexfloat << op->value;
      if (op->dtype.bits() == 32)
        temp << 'f';
      temp << "/*" << std::scientific << op->value << "*/";
    }
    p->MarkConst(temp.str());
    os << temp.str();
    break;
  }
  default:
    LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenTileLangTANG::VisitExpr_(const FloatImmNode *op,
                                     std::ostream &os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenTileLangTANG::PrintWmmaScope(const std::string &scope, DataType t,
                                         const VarNode *variable,
                                         std::ostream &os) {
  std::stringstream type;
  PrintType(t, type);
  ICHECK(fragment_shapes.count(variable))
      << "Cannot find shape of the wmma fragment " << variable->name_hint;
  std::string shape_str = fragment_shapes.at(variable);
  if ((t.is_int() || t.is_uint()) && t.bits() < 8 && t.lanes() == 1) {
    type.str(std::string());
    if (t.is_int()) {
      if (t.bits() == 4) {
        type << "nvcuda::wmma::experimental::precision::s4";
      } else if (t.bits() == 1) {
        type << "nvcuda::wmma::experimental::precision::b1";
      } else {
        LOG(FATAL) << "Unhandled integer type for wmma fragment!";
      }
    } else if (t.is_uint()) {
      if (t.bits() == 4) {
        type << "nvcuda::wmma::experimental::precision::u4";
      } else {
        LOG(FATAL) << "Unhandled integer type for wmma fragment!";
      }
    }
  }
  if (scope == "wmma.matrix_a") {
    std::string layout_str = fragment_layouts[variable];
    ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_a";
    os << "nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, " << shape_str << ", "
       << type.str() << ", nvcuda::wmma::" << layout_str << ">";
  } else if (scope == "wmma.matrix_b") {
    std::string layout_str = fragment_layouts[variable];
    ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_b";
    os << "nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, " << shape_str << ", "
       << type.str() << ", nvcuda::wmma::" << layout_str << ">";
  } else if (scope == "wmma.accumulator") {
    os << "nvcuda::wmma::fragment<nvcuda::wmma::accumulator, " << shape_str
       << ", " << type.str() << ">";
  }
}

int32_t CodeGenTileLangTANG::GetWmmaFragmentSize(const std::string &scope,
                                                 const VarNode *variable,
                                                 int32_t size) {
  ICHECK(fragment_shapes.count(variable))
      << "Cannot find shape of the wmma fragment " << variable->name_hint;
  std::string shape_str = fragment_shapes.at(variable);
  std::pair<int32_t, int32_t> dim = GetWmmaFragmentDimSize(shape_str, scope);
  if (dim.first * dim.second != 0)
    return size / dim.first / dim.second;
  else
    return 0;
}

void CodeGenTileLangTANG::HandleVolatileLoads(const std::string &value,
                                              const BufferLoadNode *op,
                                              std::ostream &os) {
  // Cast away volatile qualifier for fp16 types. That is, only loads and
  // stores are volatile. The loaded objects are not marked as volatile.
  //
  if ((op->dtype.is_float16() || op->dtype.is_bfloat16()) &&
      IsVolatile(op->buffer->data.get())) {
    os << "(";
    PrintType(op->dtype, os);
    os << ")(" << value << ")";
  } else {
    os << value;
  }
}

void CodeGenTileLangTANG::PrintVecElemLoadExpr(DataType t, int i,
                                               const std::string &value,
                                               std::ostream &os) {
  ICHECK_GT(t.lanes(), 1);
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (!(t.lanes() == 2 || t.lanes() == 3)) {
      if (i != 0) {
        os << "|";
      }
      os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8
         << "))";
      return;
    }
  }

  if (t.is_float16()) {
    if (i == 0) {
      os << "make_";
      PrintType(t, os);
      os << '(';
    }
    if (i % 2 == 0) {
      os << "__pack_half2(" << value;
    } else {
      os << "," << value << ")";
      if (i != t.lanes() - 1) {
        os << ",";
      } else {
        os << ")";
      }
    }
    return;
  }

  if (t.is_bfloat16()) {
    if (i == 0) {
      os << "make_";
      PrintType(t, os);
      os << '(';
    }
    if (i % 2 == 0) {
      os << "__pack_bfloat162(" << value;
    } else {
      os << "," << value << ")";
      if (i != t.lanes() - 1) {
        os << ",";
      } else {
        os << ")";
      }
    }
    return;
  }

  if (i == 0) {
    os << "make_";
    PrintType(t, os);
    os << "(";
  }
  os << value;
  if (i != t.lanes() - 1) {
    os << ",";
  } else {
    os << ")";
  }
  return;
}

/*!
 * \brief Detect parameters that should NOT be marked as const.
 *
 * When generating code for Tang, we add 'const' qualifier to read-only
 * parameters. However, parameters used in the following operations must NOT
 * have const:
 * - pts_load_async: Tang's async load builtin expects void* (not const)
 * - atomicAdd: atomic operation requires non-const pointer
 *
 * \param func The PrimFunc to analyze
 * \return Set of parameter VarNodes that need non-const pointers
 * \note Uses caching to avoid redundant analysis
 */
std::unordered_set<const tirx::VarNode *>
CodeGenTileLangTANG::DetectParamsNeedingNonConst(const PrimFunc &func) {
  // Check cache first
  auto it = non_const_param_cache_.find(func.get());
  if (it != non_const_param_cache_.end()) {
    return it->second;
  }

  // Collect parameter pointers and their buffer data
  std::unordered_set<const tirx::VarNode *> param_ptrs;
  std::unordered_set<const tirx::VarNode *> buffer_data_ptrs;

  for (const auto &param : func->params) {
    if (param->dtype.is_handle()) {
      param_ptrs.insert(param.get());
      if (auto opt = func->buffer_map.Get(param)) {
        if (opt.value()->data.get()) {
          buffer_data_ptrs.insert(opt.value()->data.get());
        }
      }
    }
  }

  std::unordered_set<const tirx::VarNode *> result;

  if (!param_ptrs.empty() || !buffer_data_ptrs.empty()) {
    AsyncLoadParamCollector collector(param_ptrs, buffer_data_ptrs);
    collector.Collect(func->body);
    // Merge both sets into result
    result = std::move(collector.used_in_async_load_);
    result.insert(collector.used_in_atomic_add_.begin(),
                  collector.used_in_atomic_add_.end());
  }

  // Cache the result
  non_const_param_cache_[func.get()] = result;
  return result;
}

void CodeGenTileLangTANG::PrintFunctionSignature(const String &function_name,
                                                 const PrimFunc &func,
                                                 std::ostream &os) {
  PrintFuncPrefix(os);
  CodeGenC::PrintType(func->ret_type, os);
  CodeGenC::PrintExtraAttrs(func, os);
  bool no_alias = func->HasNonzeroAttr(tirx::attr::kNoAlias);
  bool has_tang_pdl_sync = func->HasNonzeroAttr(tl::attr::kHasGridSync);
  std::unordered_set<const VarNode *> non_restrict;
  if (auto opt =
          func->GetAttr<ffi::Array<tirx::Var>>(tl::attr::kNonRestrictParams)) {
    for (const tirx::Var &v : opt.value())
      non_restrict.insert(v.get());
  }
  // Read-only param indices attribute, if present.
  // Only trust the indices computed by the AnnotateReadOnlyParams pass; do not
  // apply any positional fallback heuristic. A fallback that assumes params
  // 0/1 are read-only (based solely on no_alias + arity) mis-marks written
  // handle params (e.g. `A[i] = 0` with signature `(A, l, r)`) as const,
  // producing "read-only variable is not assignable" at compile time.
  std::unordered_set<int> ro_param_indices;
  if (auto opt =
          func->GetAttr<ffi::Array<Integer>>("tl.readonly_param_indices")) {
    for (const auto &idx : opt.value()) {
      ro_param_indices.insert(static_cast<int>(Downcast<Integer>(idx)->value));
    }
  }

  // Detect params that must NOT have const qualifier (modified via
  // async_load or atomic_add), so we don't incorrectly mark in-place
  // buffers as read-only.
  auto non_const_params = DetectParamsNeedingNonConst(func);

  os << " " << function_name << "(";
  for (size_t i = 0; i < func->params.size(); ++i) {
    tirx::Var v = func->params[i];
    std::string vid = AllocVarID(v.get());

    if (i > 0) {
      os << ", ";
    }

    if (v.dtype().is_handle()) {
      // Handle grid_constant parameters
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (ptr->storage_scope == "grid_constant") {
          os << "__grid_constant__ const ";
          CodeGenC::PrintType(ptr->element_type, os);
          os << ' ' << vid;
          continue;
        }
      }

      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, os);
      }

      // Emit const qualifier if param is explicitly marked read-only
      // and not detected as modified by async_load or atomic_add.
      if (ro_param_indices.count(static_cast<int>(i)) &&
          !non_const_params.count(v.get())) {
        os << "const ";
      }

      CodeGenC::PrintType(GetType(v), os);
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }

      if (!has_tang_pdl_sync && no_alias && !non_restrict.count(v.get())) {
        PrintRestrict(v, os);
      }
    } else {
      CodeGenC::PrintType(GetType(v), os);
    }
    os << ' ' << vid;
  }
  os << ")";

  // Register handle data type
  // TODO(tvm-team): consider simply keep type info in the
  // type annotation(via a normalizing rewriting).
  for (const auto &param : func->params) {
    if (auto *ptr = param->type_annotation.as<PointerTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType(param.get(), prim->dtype);
      }
    }
  }
}

void CodeGenTileLangTANG::InitFuncState(const PrimFunc &f) {
  CodeGenC::InitFuncState(f);
  rng_state_declared_ = false;
  rng_initialized_ = false;
  tang_random_generator_state_.clear();
  tang_random_generator_state_type_.clear();
  alloc_storage_alignment_.clear();
}

void CodeGenTileLangTANG::AddFunction(const GlobalVar &gvar,
                                      const PrimFunc &f) {
  // If the function has already been forward-declared, this is a
  // no-op.
  CodeGenC::DeclareFunction(gvar, f);
  // clear previous generated state.
  this->InitFuncState(f);
  // Build VarNode → param index map for global buffer → PTX base reg lookup.
  func_param_index_.clear();
  for (size_t i = 0; i < f->params.size(); ++i) {
    func_param_index_[f->params[i].get()] = static_cast<int>(i);
  }
  // reserve keywords
  ReserveKeywordsAsUnique();

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol)
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tirx::attr::kNoAlias);
  bool has_tang_pdl_sync = f->HasNonzeroAttr(tl::attr::kHasGridSync);
  std::unordered_set<const VarNode *> non_restrict;
  if (auto opt =
          f->GetAttr<ffi::Array<tirx::Var>>(tl::attr::kNonRestrictParams)) {
    for (const tirx::Var &v : opt.value())
      non_restrict.insert(v.get());
  }
  // Read-only param indices attribute, if present.
  // Only trust the indices computed by the AnnotateReadOnlyParams pass; do not
  // apply any positional fallback heuristic. A fallback that assumes params
  // 0/1 are read-only (based solely on no_alias + arity) mis-marks written
  // handle params (e.g. `A[i] = 0` with signature `(A, l, r)`) as const,
  // producing "read-only variable is not assignable" at compile time.
  std::unordered_set<int> ro_param_indices;
  if (auto opt = f->GetAttr<ffi::Array<Integer>>("tl.readonly_param_indices")) {
    for (const auto &idx : opt.value()) {
      ro_param_indices.insert(static_cast<int>(Downcast<Integer>(idx)->value));
    }
  }

  // Detect params that must NOT have const qualifier (modified via
  // async_load or atomic_add), so we don't incorrectly mark in-place
  // buffers as read-only.
  auto non_const_params = DetectParamsNeedingNonConst(f);

  this->PrintFuncPrefix(stream);
  CodeGenC::PrintType(f->ret_type, stream);
  this->PrintExtraAttrs(f);

  this->stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    tirx::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0)
      stream << ", ";
    if (v.dtype().is_handle()) {
      // Handle grid_constant parameters
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (ptr->storage_scope == "grid_constant") {
          stream << "__grid_constant__ const ";
          CodeGenC::PrintType(ptr->element_type, stream);
          stream << ' ' << vid;
          continue;
        }
      }

      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
      }

      // Emit const qualifier if param is explicitly marked read-only
      // and not detected as modified by async_load or atomic_add.
      if (ro_param_indices.count(static_cast<int>(i)) &&
          !non_const_params.count(v.get())) {
        stream << "const ";
      }

      CodeGenC::PrintType(GetType(v), stream);
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }

      if (!has_tang_pdl_sync && no_alias && !non_restrict.count(v.get())) {
        PrintRestrict(v, stream);
      }
    } else {
      CodeGenC::PrintType(GetType(v), stream);
    }
    stream << ' ' << vid;
  }
  stream << ") {\n";
  // Declare the Tang RNG state at function scope. Sync-insertion
  // passes may split the block holding rng_init across __syncthreads(),
  // so a call-site declaration can go out of scope before later
  // rng_rand / rng_rand_float uses. One state per PrimFunc; a second
  // rng_init is rejected here.
  tirx::PostOrderVisit(f->body, [this](const ObjectRef &n) {
    const auto *call = n.as<CallNode>();
    if (call == nullptr || !call->op.same_as(tl::rng_init())) {
      return;
    }
    ICHECK(!rng_state_declared_)
        << "Tang RNG: only one rng_init is supported per PrimFunc; found a "
           "second rng_init call.";
    ICHECK_EQ(call->args.size(), 4U);
    std::string gen = Downcast<StringImm>(call->args[3])->value;
    std::string type_name;
    if (gen == "curandStatePhilox4_32_10_t") {
      tang_random_generator_state_type_ = "philox";
      type_name = "TangRNGStatePhilox";
    } else if (gen == "curandStateMRG32k3a_t") {
      tang_random_generator_state_type_ = "mrg32k3a";
      type_name = "TangRNGStateMRG32k3a";
    } else if (gen == "curandStateXORWOW_t") {
      tang_random_generator_state_type_ = "xorwow";
      type_name = "TangRNGStateXORWOW";
    } else {
      ICHECK(false) << "Tang rng_init: unknown generator " << gen;
    }
    tang_random_generator_state_ =
        name_supply_->FreshName("__tang_random_generator_state");
    this->PrintIndent();
    this->stream << "tl::" << type_name << " " << tang_random_generator_state_
                 << ";\n";
    rng_state_declared_ = true;
  });
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

} // namespace codegen
} // namespace tvm
