/*!
 * \file src/tang/transform/s3_pipeline_injection.cc
 * \brief Prepare and finalize S3 metadata around generic software pipelining.
 */
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <map>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../backend/common/target_utils.h"
#include "../../op/builtin.h"
#include "../../transform/common/pipeline_utils.h"
#include "s3_async_pipeline_sync_schema.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {
namespace tang {
namespace transform {

using namespace tirx;
using namespace ffi;

namespace {

constexpr const char *kPipelineRegionWarpsAttr = "tl.pipeline_region_warps";

bool IsPipelineLoop(const ForNode *loop) {
  return loop->annotations.count(s_tir::attr::software_pipeline_stage) != 0 ||
         loop->annotations.count(s_tir::attr::software_pipeline_order) != 0;
}

std::string BufferKey(const Buffer &buffer) {
  std::ostringstream os;
  os << buffer->data;
  return os.str();
}

struct PipelineRegionAccess {
  std::string key;
  int64_t access{0};
};

class PipelineRegionFinder : public StmtExprVisitor {
public:
  void VisitExpr_(const CallNode *op) final {
    static const Op &gemm = Op::Get("tl.tileop.gemm");
    if (op->op.same_as(gemm)) {
      has_mma_consumer = true;
    }
    if (op->op.same_as(tl::region()) && op->args.size() >= 2) {
      const auto *load = op->args[0].as<BufferLoadNode>();
      const auto *mask = op->args[1].as<IntImmNode>();
      if (load != nullptr && mask != nullptr) {
        accesses.push_back({BufferKey(load->buffer), mask->value});
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  std::vector<PipelineRegionAccess> accesses;
  bool has_mma_consumer{false};
};

Map<String, Any> BuildPipelinePlans(const Array<Stmt> &statements,
                                    const Array<Integer> &stages,
                                    int64_t versions, int64_t loop_uid) {
  Map<String, Any> facts;
  if (versions <= 1 || loop_uid <= 0 || statements.size() != stages.size()) {
    return facts;
  }

  std::map<std::string, int64_t> producers;
  std::map<std::string, int64_t> consumers;
  std::unordered_set<std::string> ambiguous;
  for (size_t i = 0; i < statements.size(); ++i) {
    PipelineRegionFinder finder;
    finder(statements[i]);
    for (const PipelineRegionAccess &access : finder.accesses) {
      if ((access.access & 2) != 0) {
        if (!producers.emplace(access.key, stages[i]->value).second) {
          ambiguous.insert(access.key);
        }
      }
      if (finder.has_mma_consumer && (access.access & 1) != 0) {
        if (!consumers.emplace(access.key, stages[i]->value).second) {
          ambiguous.insert(access.key);
        }
      }
    }
  }

  int64_t buffer_uid = 0;
  for (const auto &[key, producer_stage] : producers) {
    auto consumer = consumers.find(key);
    if (ambiguous.count(key) || consumer == consumers.end()) {
      continue;
    }
    const int64_t distance = consumer->second - producer_stage;
    if (distance <= 0 || distance >= versions) {
      continue;
    }
    PipelinePlan plan;
    plan.loop_uid = loop_uid;
    plan.buffer_uid = ++buffer_uid;
    plan.versions = versions;
    plan.distance = distance;
    plan.producer_stage = producer_stage;
    plan.consumer_stage = consumer->second;
    facts.Set(String(key), EncodePipelinePlan(plan));
  }
  return facts;
}

class CopyPlanAnnotator : public StmtExprMutator {
public:
  explicit CopyPlanAnnotator(PipelinePlan plan) : plan_(plan) {}

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    const auto *call = expr.as<CallNode>();
    const auto *callee = call == nullptr ? nullptr : call->op.as<OpNode>();
    if (callee == nullptr || (callee->name != "tl.tileop.copy" &&
                              callee->name != "tl.tileop.async_copy" &&
                              callee->name != "tl.tileop.tma_copy")) {
      return expr;
    }
    Map<String, ObjectRef> annotations = call->annotations;
    annotations.Set(kPipelineLoopUidAttr, Integer(plan_.loop_uid));
    annotations.Set(kPipelineBufferUidAttr, Integer(plan_.buffer_uid));
    annotations.Set(kPipelineVersionsAttr, Integer(plan_.versions));
    annotations.Set(kPipelineDistanceAttr, Integer(plan_.distance));
    annotations.Set(kPipelineProducerStageAttr, Integer(plan_.producer_stage));
    annotations.Set(kPipelineConsumerStageAttr, Integer(plan_.consumer_stage));
    annotations.Set(
        kPipelineReadyOwnerAttr,
        Integer(static_cast<int64_t>(PipelineReadyOwner::kS3AsyncCopy)));
    return Call(call->dtype, call->op, call->args, annotations, call->span);
  }

  PipelinePlan plan_;
};

Stmt AnnotateCopyPlan(const Stmt &stmt, const Map<String, Any> &facts) {
  PipelineRegionFinder finder;
  finder(stmt);
  std::optional<PipelinePlan> selected;
  for (const PipelineRegionAccess &access : finder.accesses) {
    if ((access.access & 2) == 0) {
      continue;
    }
    auto encoded = facts.Get(String(access.key));
    if (!encoded.has_value()) {
      continue;
    }
    PipelinePlan plan;
    if (!ParsePipelinePlan(Downcast<Array<Integer>>(encoded.value()), &plan)) {
      continue;
    }
    if (selected.has_value() && selected->buffer_uid != plan.buffer_uid) {
      return stmt;
    }
    selected = plan;
  }
  return selected.has_value() ? CopyPlanAnnotator(*selected)(stmt) : stmt;
}

bool HasCompleteCopyPlan(const Stmt &stmt) {
  class Checker : public StmtExprVisitor {
  public:
    void VisitExpr_(const CallNode *op) final {
      const auto *callee = op->op.as<OpNode>();
      if (callee != nullptr && (callee->name == "tl.tileop.copy" ||
                                callee->name == "tl.tileop.async_copy" ||
                                callee->name == "tl.tileop.tma_copy")) {
        saw_copy = true;
        Array<Integer> encoded{Integer(kPipelinePlanSchemaVersion)};
        for (const char *key :
             {kPipelineLoopUidAttr, kPipelineBufferUidAttr,
              kPipelineVersionsAttr, kPipelineDistanceAttr,
              kPipelineProducerStageAttr, kPipelineConsumerStageAttr,
              kPipelineReadyOwnerAttr}) {
          auto value = op->annotations.Get(key);
          const auto *imm =
              value.has_value() ? value.value().as<IntImmNode>() : nullptr;
          if (imm == nullptr) {
            valid = false;
            return;
          }
          encoded.push_back(Integer(imm->value));
        }
        PipelinePlan plan;
        valid &= ParsePipelinePlan(encoded, &plan);
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    bool saw_copy{false};
    bool valid{true};
  } checker;
  checker(stmt);
  return checker.saw_copy && checker.valid;
}

Array<Stmt> PipelineStatements(const Stmt &body) {
  Array<Stmt> result;
  for (const Stmt &stmt : tl::NormalizePipelineBody(body)) {
    if (!tl::IsPipelineDeclarationStmt(stmt)) {
      result.push_back(stmt);
    }
  }
  return result;
}

class S3PipelinePlanAnnotator : public StmtExprMutator {
public:
  S3PipelinePlanAnnotator(int64_t block_warps,
                          std::unordered_set<const VarNode *> thread_vars)
      : block_warps_(block_warps), thread_vars_(std::move(thread_vars)) {}

private:
  Stmt VisitStmt_(const IfThenElseNode *op) final {
    const bool thread_guard =
        tirx::UsesVar(op->condition, [&](const VarNode *var) {
          return thread_vars_.count(var) != 0;
        });
    ++conditional_depth_;
    thread_guard_depth_ += thread_guard ? 1 : 0;
    Stmt result = StmtExprMutator::VisitStmt_(op);
    thread_guard_depth_ -= thread_guard ? 1 : 0;
    --conditional_depth_;
    return result;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (!IsPipelineLoop(op)) {
      return loop;
    }

    auto stage_annotation =
        op->annotations.Get(s_tir::attr::software_pipeline_stage);
    if (!stage_annotation.has_value()) {
      return loop;
    }
    Array<Integer> stages = Downcast<Array<Integer>>(stage_annotation.value());
    Array<Stmt> statements = PipelineStatements(loop->body);

    int64_t configured_stages = 0;
    if (Optional<Integer> num_stages = tl::GetPipelineNumStages(op)) {
      configured_stages = num_stages.value()->value;
    }
    int64_t versions = configured_stages;
    for (const Integer &stage : stages) {
      versions = std::max(versions, stage->value + int64_t{1});
    }

    const int64_t loop_uid = next_loop_uid_++;
    const bool has_engine_copy_annotation =
        loop->annotations.count(tl::kPipelineTmaCopies) != 0;
    Map<String, Any> facts;
    if (!has_engine_copy_annotation || configured_stages > 1) {
      facts = BuildPipelinePlans(statements, stages, versions, loop_uid);
    }
    if (!facts.empty()) {
      for (size_t i = 0; i < statements.size(); ++i) {
        statements.Set(i, AnnotateCopyPlan(statements[i], facts));
      }
      loop.CopyOnWrite()->body =
          statements.size() == 1 ? statements[0] : SeqStmt(statements);
    }

    Map<String, Any> annotations = loop->annotations;
    annotations.Set(kPipelineLoopUidAttr, Integer(loop_uid));
    annotations.Set(kPipelineRegionWarpsAttr,
                    Integer(thread_guard_depth_ > 0 ? -1 : block_warps_));
    if (!facts.empty()) {
      annotations.Set(kPipelinePlanAttr, facts);
    }

    if (auto tma = annotations.Get(tl::kPipelineTmaCopies)) {
      Array<Integer> flags = Downcast<Array<Integer>>(tma.value());
      bool valid = conditional_depth_ == 0 && configured_stages > 1 &&
                   flags.size() == statements.size() && !facts.empty();
      if (valid) {
        for (size_t i = 0; i < flags.size(); ++i) {
          if (!is_zero(flags[i]) && !HasCompleteCopyPlan(statements[i])) {
            valid = false;
            break;
          }
        }
      }
      if (!valid) {
        Array<Integer> blocking(flags.size(), Integer(0));
        annotations.Set(tl::kPipelineTmaCopies, blocking);
      }
    }
    loop.CopyOnWrite()->annotations = std::move(annotations);
    return loop;
  }

  int64_t block_warps_{0};
  std::unordered_set<const VarNode *> thread_vars_;
  int conditional_depth_{0};
  int thread_guard_depth_{0};
  int64_t next_loop_uid_{1};
};

struct ThreadInfo {
  int64_t block_warps{0};
  std::unordered_set<const VarNode *> thread_vars;
};

ThreadInfo CollectThreadInfo(const Stmt &body) {
  class Finder : public StmtVisitor {
  public:
    void VisitStmt_(const AttrStmtNode *op) final {
      if (op->attr_key == tirx::attr::thread_extent) {
        const auto *iv = op->node.as<IterVarNode>();
        if (iv != nullptr &&
            std::string(iv->thread_tag).compare(0, 9, "threadIdx") == 0) {
          info.thread_vars.insert(iv->var.get());
        }
        if (iv != nullptr && iv->thread_tag == "threadIdx.x") {
          const auto *extent = op->value.as<IntImmNode>();
          if (extent != nullptr && extent->value > 0 &&
              extent->value % 32 == 0) {
            const int64_t warps = extent->value / 32;
            ICHECK(info.block_warps == 0 || info.block_warps == warps);
            info.block_warps = warps;
          }
        }
      }
      StmtVisitor::VisitStmt_(op);
    }

    ThreadInfo info;
  } finder;
  finder(body);
  return std::move(finder.info);
}

const VarNode *BarrierVar(const PrimExpr &barrier) {
  if (const auto *load = barrier.as<BufferLoadNode>()) {
    return load->buffer->data.get();
  }
  return nullptr;
}

class S3PipelineFinalizer : public StmtExprMutator {
public:
  explicit S3PipelineFinalizer(int64_t block_warps)
      : block_warps_(block_warps) {}

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    const auto *call = expr.as<CallNode>();
    const auto *callee = call == nullptr ? nullptr : call->op.as<OpNode>();
    if (callee == nullptr || callee->name != "tl.tileop.tma_copy") {
      return expr;
    }
    auto barrier = call->annotations.Get("barrier");
    std::optional<PipelinePlan> plan = ReadPlan(call->annotations);
    if (!barrier.has_value() || !plan.has_value()) {
      return expr;
    }
    const auto *barrier_var = BarrierVar(Downcast<PrimExpr>(barrier.value()));
    if (barrier_var == nullptr || block_warps_ <= 0) {
      return expr;
    }
    copy_buffer_uids_[barrier_var].insert(plan->buffer_uid);
    Map<String, ObjectRef> annotations = call->annotations;
    annotations.Set("tang_pipeline_arrive_count", Integer(block_warps_));
    annotations.Set(kPipelineReadyOwnerAttr,
                    Integer(static_cast<int64_t>(
                        PipelineReadyOwner::kInjectSoftwarePipeline)));
    return Call(call->dtype, call->op, call->args, annotations, call->span);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    auto init = block->annotations.Get("barrier_init");
    if (!init.has_value()) {
      return block;
    }
    auto init_map = Downcast<Map<Var, Array<PrimExpr>>>(init.value());
    bool changed = false;
    for (const auto &[barrier, counts] : init_map) {
      auto copies = copy_buffer_uids_.find(barrier.get());
      if (copies == copy_buffer_uids_.end()) {
        continue;
      }
      Array<PrimExpr> updated;
      for (size_t i = 0; i < counts.size(); ++i) {
        updated.push_back(
            IntImm(DataType::Int(32),
                   block_warps_ * static_cast<int64_t>(copies->second.size())));
      }
      init_map.Set(barrier, updated);
      changed = true;
    }
    if (!changed) {
      return block;
    }
    Map<String, Any> annotations = block->annotations;
    annotations.Set("barrier_init", init_map);
    block.CopyOnWrite()->annotations = std::move(annotations);
    return block;
  }

  static std::optional<PipelinePlan>
  ReadPlan(const Map<String, ObjectRef> &annotations) {
    Array<Integer> encoded{Integer(kPipelinePlanSchemaVersion)};
    for (const char *key :
         {kPipelineLoopUidAttr, kPipelineBufferUidAttr, kPipelineVersionsAttr,
          kPipelineDistanceAttr, kPipelineProducerStageAttr,
          kPipelineConsumerStageAttr, kPipelineReadyOwnerAttr}) {
      auto value = annotations.Get(key);
      const auto *imm =
          value.has_value() ? value.value().as<IntImmNode>() : nullptr;
      if (imm == nullptr) {
        return std::nullopt;
      }
      encoded.push_back(Integer(imm->value));
    }
    PipelinePlan plan;
    return ParsePipelinePlan(encoded, &plan) ? std::optional<PipelinePlan>(plan)
                                             : std::nullopt;
  }

  int64_t block_warps_{0};
  std::unordered_map<const VarNode *, std::unordered_set<int64_t>>
      copy_buffer_uids_;
};

bool IsS3(const PrimFunc &func) {
  Optional<Target> target = func->GetAttr<Target>(tvm::attr::kTarget);
  return target.defined() && TargetTangIsSTCUV2(target.value());
}

} // namespace

tvm::transform::Pass AnnotateS3PipelinePlan() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    if (!IsS3(func)) {
      return func;
    }
    ThreadInfo info = CollectThreadInfo(func->body);
    Stmt body = S3PipelinePlanAnnotator(
        info.block_warps, std::move(info.thread_vars))(func->body);
    if (body.same_as(func->body)) {
      return func;
    }
    func.CopyOnWrite()->body = std::move(body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tang.AnnotateS3PipelinePlan", {});
}

tvm::transform::Pass FinalizeS3PipelineInjection() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    if (!IsS3(func)) {
      return func;
    }
    ThreadInfo info = CollectThreadInfo(func->body);
    Stmt body = S3PipelineFinalizer(info.block_warps)(func->body);
    if (body.same_as(func->body)) {
      return func;
    }
    func.CopyOnWrite()->body = std::move(body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tang.FinalizeS3PipelineInjection", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.tang.transform.AnnotateS3PipelinePlan", AnnotateS3PipelinePlan)
      .def("tl.tang.transform.FinalizeS3PipelineInjection",
           FinalizeS3PipelineInjection);
}

} // namespace transform
} // namespace tang
} // namespace tl
} // namespace tvm
