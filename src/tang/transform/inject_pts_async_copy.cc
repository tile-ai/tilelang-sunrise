/*!
 * \brief Replace copy from global to shared with async copy
 * \file inject_pts_async_copy.cc
 */
#include "support/check.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "op/builtin.h"
#include "tir/ir/buffer_common.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/tirx/stmt.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class PTSAsyncCopyInjector : public StmtMutator {
public:
  Stmt VisitStmt_(const AttrStmtNode *attr) {
    if (attr->attr_key == s_tir::attr::async_scope) {
      ICHECK(in_async == false) << "Nested async scopes not supported";
      in_async = true;
      auto body = this->VisitStmt(attr->body);
      in_async = false;
      return body;
    }
    return StmtMutator::VisitStmt_(attr);
  }

  Stmt InjectPTS(const BufferLoadNode *load, const BufferStoreNode *store,
                 bool predicated = false,
                 const PrimExpr &predicate_value = PrimExpr()) {
    bool is_shared = (load->buffer.scope() == "shared" ||
                      load->buffer.scope() == "shared.dyn");
    if (store->buffer.scope() == "global" && is_shared) {
      ICHECK(load->indices.size() == 1 && store->indices.size() == 1);
      ICHECK(load->indices[0]->dtype.lanes() ==
             store->indices[0]->dtype.lanes())
          << load->indices[0] << " vs. " << store->indices[0] << " with lanes "
          << load->indices[0]->dtype.lanes() << " vs. "
          << store->indices[0]->dtype.lanes();

      const int indices_lanes = load->indices[0]->dtype.lanes();
      const int bytes = indices_lanes * load->buffer->dtype.bytes();

      // Only 4/8/16-byte transfers are emitted as async DMA. Sub-4-byte
      // scalar copies (int8 -> 1 byte, fp16 -> 2 bytes) must NOT become
      // pts_store_async: codegen widens every async store to a 4-byte
      // cop4 transaction, so a 1/2-byte store would overrun global memory
      // (see gemm-int8-epilogue-cop4-race). Mirror the load path below,
      // which gates its whole injection on this same check; anything that
      // doesn't qualify falls through to the plain synchronous store at the
      // end of this function.
      if (bytes == 4 || bytes == 8 || bytes == 16) {
        auto dst_elem_type =
            GetPointerType(store->buffer->data->type_annotation);
        auto src_elem_type =
            GetPointerType(load->buffer->data->type_annotation);
        ICHECK(dst_elem_type.has_value() && src_elem_type.has_value())
            << "Both store and load buffer should have a pointer type "
               "annotation.";

        int index_factor = 1;
        // FIXME: require matching source and destination element types.

        if (indices_lanes == 1) {
          auto src_offset = load->indices[0];
          auto dst_offset = store->indices[0];
          // args order: {src_shared, dst_global, bytes}.
          // This order is consumed by AsyncLoadParamCollector in
          // codegen_tang.cc, which reads args[1] as the global destination
          // that must not be const-qualified. Keep the two in sync.
          Array<PrimExpr> args = {
              AddressOffset(load->buffer->data, load->buffer->dtype,
                            src_offset),
              AddressOffset(store->buffer->data, store->buffer->dtype,
                            mul(dst_offset, PrimExpr(index_factor))),
              PrimExpr(bytes)};

          // use arguments size to indicate whether or not to use predicated
          // cp.async
          if (predicated) {
            args.push_back(predicate_value);
          }
          return Evaluate(
              Call(store->buffer->dtype, tvm::tl::pts_store_async(), args));
        }

        // Predicated load don't support vectorized indexing.
        if (!predicated) {
          // Only some vectorized indexing patterns are supported for now.
          // Only some vectorized indexing patterns are supported for now.
          auto src_offset = [=]() -> PrimExpr {
            if (auto *add = load->indices[0].as<AddNode>()) {
              if (!add->a->IsInstance<RampNode>())
                return PrimExpr();
              if (!add->b->IsInstance<BroadcastNode>())
                return PrimExpr();
              return tirx::Add(add->a.as<RampNode>()->base,
                               add->b.as<BroadcastNode>()->value);
            }
            if (load->indices[0].as<RampNode>()) {
              return load->indices[0].as<RampNode>()->base;
            }
            return PrimExpr();
          }();

          auto dst_offset = [=]() -> PrimExpr {
            if (store->indices[0].as<RampNode>()) {
              return store->indices[0].as<RampNode>()->base;
            } else if (store->indices[0].as<AddNode>()) {
              // The case where the dst buffer is a byte buffer generated by
              // merging dynamic shared memory. A_shared.dyn[(ramp(...), 1, 8) +
              // x8(17408))] = A_global[ramp(...),1, 8)]
              auto *add = store->indices[0].as<AddNode>();
              if (!add->a->IsInstance<RampNode>())
                return PrimExpr();
              if (!add->b->IsInstance<BroadcastNode>())
                return PrimExpr();
              return tirx::Add(add->a.as<RampNode>()->base,
                               add->b.as<BroadcastNode>()->value);
            }
            return PrimExpr();
          }();

          if (src_offset.defined() && dst_offset.defined()) {
            // args order: {src_shared, dst_global, bytes} -- consumed by
            // AsyncLoadParamCollector in codegen_tang.cc (args[1] = dst).
            return Evaluate(
                Call(store->buffer->dtype, tvm::tl::pts_store_async(),
                     {AddressOffset(load->buffer->data, load->buffer->dtype,
                                    src_offset),
                      AddressOffset(store->buffer->data, store->buffer->dtype,
                                    mul(dst_offset, PrimExpr(index_factor))),
                      PrimExpr(bytes)}));
          }
        }
      }
    }
    if (load->buffer.scope() == "global") {
      ICHECK(load->indices.size() == 1 && store->indices.size() == 1);
      ICHECK(load->indices[0]->dtype.lanes() ==
             store->indices[0]->dtype.lanes())
          << load->indices[0] << " vs. " << store->indices[0] << " with lanes "
          << load->indices[0]->dtype.lanes() << " vs. "
          << store->indices[0]->dtype.lanes();

      const int indices_lanes = load->indices[0]->dtype.lanes();
      const int bytes = indices_lanes * load->buffer->dtype.bytes();

      if (bytes == 4 || bytes == 8 || bytes == 16) {
        auto dst_elem_type =
            GetPointerType(store->buffer->data->type_annotation);
        auto src_elem_type =
            GetPointerType(load->buffer->data->type_annotation);
        ICHECK(dst_elem_type.has_value() && src_elem_type.has_value())
            << "Both store and load buffer should have a pointer type "
               "annotation.";

        int index_factor = 1;
        // FIXME: require matching source and destination element types.

        if (indices_lanes == 1) {
          auto src_offset = load->indices[0];
          auto dst_offset = store->indices[0];
          // args order: {dst_shared, src_global, bytes} -- consumed by
          // AsyncLoadParamCollector in codegen_tang.cc (args[0] = dst).
          Array<PrimExpr> args = {
              AddressOffset(store->buffer->data, store->buffer->dtype,
                            mul(dst_offset, PrimExpr(index_factor))),
              AddressOffset(load->buffer->data, load->buffer->dtype,
                            src_offset),
              PrimExpr(bytes)};

          // use arguments size to indicate whether or not to use predicated
          // cp.async
          if (predicated) {
            args.push_back(predicate_value);
          }
          return Evaluate(
              Call(store->buffer->dtype, tvm::tl::pts_load_async(), args));
        }

        // Predicated load don't support vectorized indexing.
        if (!predicated) {
          // Only some vectorized indexing patterns are supported for now.
          // Only some vectorized indexing patterns are supported for now.
          auto src_offset = [=]() -> PrimExpr {
            if (load->indices[0]->IsInstance<RampNode>()) {
              return load->indices[0].as<RampNode>()->base;
            }
            return PrimExpr();
          }();

          auto dst_offset = [=]() -> PrimExpr {
            if (store->indices[0].as<RampNode>()) {
              return store->indices[0].as<RampNode>()->base;
            } else if (store->indices[0].as<AddNode>()) {
              // The case where the dst buffer is a byte buffer generated by
              // merging dynamic shared memory. A_shared.dyn[(ramp(...), 1, 8) +
              // x8(17408))] = A_global[ramp(...),1, 8)]
              auto *add = store->indices[0].as<AddNode>();
              if (!add->a->IsInstance<RampNode>())
                return PrimExpr();
              if (!add->b->IsInstance<BroadcastNode>())
                return PrimExpr();
              return tirx::Add(add->a.as<RampNode>()->base,
                               add->b.as<BroadcastNode>()->value);
            }
            return PrimExpr();
          }();
          if (src_offset.defined() && dst_offset.defined()) {
            // args order: {dst_shared, src_global, bytes} -- consumed by
            // AsyncLoadParamCollector in codegen_tang.cc (args[0] = dst).
            return Evaluate(
                Call(store->buffer->dtype, tvm::tl::pts_load_async(),
                     {AddressOffset(store->buffer->data, store->buffer->dtype,
                                    mul(dst_offset, PrimExpr(index_factor))),
                      AddressOffset(load->buffer->data, load->buffer->dtype,
                                    src_offset),
                      PrimExpr(bytes)}));
          }
        }
      }
    }
    return StmtMutator::VisitStmt_(store);
  }

  Stmt VisitStmt_(const BufferStoreNode *store) {
    bool is_shared = (store->buffer.scope() == "shared" ||
                      store->buffer.scope() == "shared.dyn");
    if (in_async && store->buffer.scope() == "global") {
      if (auto *load = store->value.as<BufferLoadNode>()) {
        return InjectPTS(load, store);
      }
    } else if (in_async && is_shared) {
      if (auto *load = store->value.as<BufferLoadNode>()) {
        return InjectPTS(load, store);
      } else if (auto *call = store->value.as<CallNode>()) {
        // tir.if_then_else is a call to builtin::if_then_else()
        if (call->op.same_as(builtin::if_then_else()) &&
            call->args.size() == 3) {
          if (auto *load = call->args[1].as<BufferLoadNode>()) {
            // Only default value of 0 is supported since 0 is the default value
            // used by cp.async ptx. @see section 9.7.8.22.3. of
            // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-memory-operations
            bool else_value_is_zero = false;
            if (auto *b = call->args[2].as<BroadcastNode>()) {
              if (auto *f = b->value.as<FloatImmNode>()) {
                else_value_is_zero = f->value == 0.0f;
              } else if (auto *i = b->value.as<IntImmNode>()) {
                else_value_is_zero = i->value == 0;
              }
            }
            if (auto *f = call->args[2].as<FloatImmNode>()) {
              else_value_is_zero = f->value == 0.0f;
            } else if (auto *i = call->args[2].as<IntImmNode>()) {
              else_value_is_zero = i->value == 0;
            }
            if (else_value_is_zero) {
              return InjectPTS(load, store, true, call->args[0]);
            }
          }
        }
      }
    }
    return StmtMutator::VisitStmt_(store);
  }

private:
  bool in_async{false};
};

using namespace tirx::transform;

tvm::transform::Pass InjectPTSAsyncCopy() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto *n = f.CopyOnWrite();
    n->body = PTSAsyncCopyInjector()(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectPTSAsyncCopy", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.tang.transform.InjectPTSAsyncCopy",
                        InjectPTSAsyncCopy);
}

} // namespace tl
} // namespace tvm
