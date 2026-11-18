/*!
 * \file inject_s3_async_copy.cc
 * \brief Materialize stcuv2 pipeline handshakes after bulk-copy lowering.
 *
 * This pass consumes typed ``tl.tang_bulk_async_commit`` records and adds the
 * free/overwrite edge: the copy waits before reusing a shared-buffer version,
 * and the consuming MMA publishes retirement with a tensor-core fence. When
 * InjectSoftwarePipeline does not own the ready edge, the same plan also
 * materializes its producer arrival and consumer wait. Copies without a typed
 * plan remain blocking.
 *
 * Barrier counts and version identities must be statically exact. If they are
 * not, the pass leaves the function unchanged so the blocking copy remains
 * correct. Complete edges are certified for the later redundant-sync removal.
 */
#include "cuda/op/builtin.h"
#include "support/check.h"
#include "tang/op/builtin.h"
#include "tang/target_utils.h"
#include "tang/transform/s3_async_pipeline_sync_schema.h"
#include "tang/transform/s3_mbarrier_utils.h"
#include <algorithm>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tvm/ir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
using tvm::transform::PassContext;

// The mbarrier protocol is shared with the late TANG materializer: one home
// for the prologue, so a fix lands once (tang/transform/s3_mbarrier_utils.h).
using tang::transform::AnchorAsyncEdge;
using tang::transform::AsyncEdgeAnchorRole;
using tang::transform::BlockThreads;
using tang::transform::FindBlockThreads;
using tang::transform::kAsyncEdgeFactsAttr;
using tang::transform::kPipelinePlanSchemaVersion;
using tang::transform::ParsePipelinePlan;
using tang::transform::PipelinePlan;
using tang::transform::ThreadExtentPrologue;

namespace {

// Each pipelined buffer owns 2*versions contiguous slots: ready[stage] followed
// by free[stage]. Operands cannot share slots because their arrival counts and
// parity generations are independent.
constexpr int64_t kReadyBase = 0;

// ISP records each validated buffer rotation in the typed v2 plan.
constexpr const char *kPipelineFacts = tang::transform::kPipelinePlanAttr;

// ISP records the loop's participating warps, or -1 when a surrounding thread
// guard prevents an exact count.
constexpr const char *kLoopRegionWarps = "tl.pipeline_region_warps";

bool IsMmaOp(const CallNode *call) {
  return call != nullptr && (call->op.same_as(tl::tang_tcgen05_mma_ss()) ||
                             call->op.same_as(tl::tang_tcgen05_mma_ts()));
}

/*!
 * \brief One record the lowering leaves next to an IR-emitted copy.
 *
 * Exactly the facts a handshake would otherwise have to reverse-engineer out of
 * the copy op: the buffer, the version written, how many versions it has, the
 * arrive count, and the copy's region warp count. The
 * lowering's own names for the last two are ``active_warps`` and
 * ``region_warps``; ``arrive``/``num_warps`` are what the two readers call them
 * here, because that is the role they play in the handshake.
 *
 * This pass is the record's only reader, so its private model and parser stay
 * next to the rewrite that consumes them.
 */
struct CopyRecord {
  std::string key;
  PrimExpr stage;
  int64_t versions{1};
  int64_t arrive{0};
  int64_t num_warps{1};
  int64_t fg{0};
  PipelinePlan plan;
  /*!
   * \brief The barrier the lowering already arrived on; undefined when the
   *        handshake pass still owns the copy's arrival.
   *
   * Set when the generic pipeline machinery owns the edge: it created the
   * barrier, annotated the copy with the reference and inserted the consumer's
   * phase wait, so the emitter's warps post the arrival themselves and the
   * pass must not post a second one.
   */
  Optional<PrimExpr> pipeline_barrier;
};

/*!
 * \brief Read a ``tl.tang_bulk_async_commit`` call; false when it is malformed.
 *
 * A malformed record is a lowering bug rather than a shape to tolerate, so the
 * caller refuses (or drops and counts) instead of guessing.
 */
inline bool ReadRecord(const CallNode *call, CopyRecord *out) {
  if (call == nullptr || (call->args.size() != 13 && call->args.size() != 14)) {
    return false;
  }
  const auto *schema = call->args[0].as<IntImmNode>();
  if (schema == nullptr || schema->value != kPipelinePlanSchemaVersion) {
    return false;
  }
  constexpr size_t base = 1;
  const auto *key = call->args[base].as<StringImmNode>();
  const auto *versions = call->args[base + 2].as<IntImmNode>();
  const auto *arrive = call->args[base + 3].as<IntImmNode>();
  const auto *region = call->args[base + 4].as<IntImmNode>();
  const auto *fg = call->args[base + 5].as<IntImmNode>();
  if (key == nullptr || versions == nullptr || arrive == nullptr ||
      region == nullptr || fg == nullptr) {
    return false;
  }
  out->key = key->value;
  out->stage = call->args[base + 1];
  out->versions = versions->value;
  out->arrive = arrive->value;
  out->num_warps = region->value;
  out->fg = fg->value;
  Array<Integer> encoded{Integer(kPipelinePlanSchemaVersion)};
  for (size_t i : {base + 6, base + 7}) {
    const auto *imm = call->args[i].as<IntImmNode>();
    if (imm == nullptr) {
      return false;
    }
    encoded.push_back(Integer(imm->value));
  }
  encoded.push_back(Integer(versions->value));
  for (size_t i = base + 8; i < base + 11; ++i) {
    const auto *imm = call->args[i].as<IntImmNode>();
    if (imm == nullptr) {
      return false;
    }
    encoded.push_back(Integer(imm->value));
  }
  const auto *owner = call->args[base + 11].as<IntImmNode>();
  if (owner == nullptr || (owner->value != 0 && owner->value != 1)) {
    return false;
  }
  encoded.push_back(Integer(owner->value));
  if (!ParsePipelinePlan(encoded, &out->plan)) {
    return false;
  }
  constexpr size_t barrier_index = base + 12;
  if (call->args.size() > barrier_index) {
    out->pipeline_barrier = call->args[barrier_index];
  }
  return true;
}

// S and D in ``floormod(loop_var + D, S)`` come from declarations: lowering
// records the version count and ISP records the producer-consumer distance.
struct BufferInfo {
  int64_t loop_uid{0};
  int64_t buffer_uid{0};
  int64_t distance{0};
  int64_t arrive{1};    // warps that publish the ready edge
  int64_t num_warps{1}; // warps in the copy's own region
  bool rotated{false};  // some copy writes it with a non-constant stage
  int64_t versions{0};  // pipeline stages, from the rotation's modulus
  int64_t slot{0};      // id slot, so operands do not share a barrier
  // The id this buffer's ready/free block starts at. Planned once by
  // ``PlanBarriers`` instead of recomputed from ``slot`` at every use: the id
  // space is this pass's own encoding of (owner buffer, role, version). Keep
  // that encoding stated in one place so the representation can change without
  // changing the protocol.
  int64_t ready_base{0};
  // Prologue producers: every *constant-stage* writer of this buffer, in the
  // order the walk met them. A pipeline peeled by more than one iteration
  // writes one version per peeled iteration. They seed generation zero of the
  // same ready slots the steady loop reuses.
  std::vector<int64_t> pro_stages;
  /*!
   * \brief Who posts this buffer's ready edge.
   *
   * 1 when every copy arrived on the barrier ``InjectSoftwarePipeline``
   * created (the generic pipeline owns the ready wait too). 0 when this pass
   * still owns it. -1 before any copy is registered. Mixed ownership is
   * refused: a phase the pipeline waits for and this pass also waits for
   * cannot be counted by either one.
   */
  int engine_ready{-1};
  bool PipelineOwnsReady() const { return engine_ready == 1; }
};

// ready[stage] of a buffer: one contiguous id block per buffer slot, with the
// odd id right after each ready id reserved for free[stage]. Where the block
// starts is *planned* (``BufferInfo::ready_base``), never recomputed here.
PrimExpr IdOf(const BufferInfo &info, const PrimExpr &stage) {
  return IntImm(stage.dtype(), info.ready_base) +
         stage * IntImm(stage.dtype(), 2);
}

// free[stage] of a buffer: posted by the consuming MMA's tensor-core fence once
// that version's reads have retired, waited on by the producer before it
// overwrites the version.
PrimExpr FreeIdOf(const BufferInfo &info, const PrimExpr &stage) {
  return IntImm(stage.dtype(), info.ready_base + 1) +
         stage * IntImm(stage.dtype(), 2);
}

int64_t LastIdOf(const BufferInfo &info) {
  const int64_t stages = info.versions > 0 ? info.versions : 1;
  return info.ready_base + 2 * (stages - 1) + 1;
}

/*!
 * \brief ``free[stage]`` of a buffer as a plain integer.
 *
 * The same number ``FreeIdOf`` builds as an expression: the id space and the
 * slot space are one space here (see the note above ``IdOf``), so the init
 * table and the emitted slot agree by construction.
 */
int64_t FreeSlotOf(const BufferInfo &info, int64_t stage) {
  return info.ready_base + 1 + 2 * stage;
}

// Assign one ready/free slot pair per version of each managed buffer.
bool PlanBarriers(std::unordered_map<std::string, BufferInfo> *buffers,
                  int64_t *last_id) {
  std::vector<BufferInfo *> managed;
  for (auto &entry : *buffers) {
    if (entry.second.rotated) {
      managed.push_back(&entry.second);
    }
  }
  std::sort(managed.begin(), managed.end(),
            [](const BufferInfo *a, const BufferInfo *b) {
              return a->slot < b->slot;
            });
  int64_t next = kReadyBase;
  for (BufferInfo *info : managed) {
    info->ready_base = next;
    next += 2 * info->versions;
  }
  *last_id = next - 1;
  return !managed.empty();
}

/*!
 * \brief Does the prologue seed the consumer's first versions in visit order?
 *
 * The first ``p`` visits consume versions ``0 .. p-1``. A different set/order
 * would leave a ready slot unseeded or seed one before the matching wait.
 */
bool PrologueCoversFirstVisits(const BufferInfo &info) {
  if (info.pro_stages.empty()) {
    return false;
  }
  if (info.pro_stages.size() > static_cast<size_t>(info.versions)) {
    return false;
  }
  for (size_t i = 0; i < info.pro_stages.size(); ++i) {
    if (info.pro_stages[i] != static_cast<int64_t>(i)) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Per-slot arrival counts.
//
// An mbarrier is initialized with the number of warps that publish that edge;
// the number of waiting warps is not part of its state:
//
//   ready   arrive = the warps the copy call itself splits the tile over
//   free    arrive = the warps where the arrive is appended (the loop body)
//
// ISP declares each loop's participating warps in ``kLoopRegionWarps``; this
// pass consumes that fact for the free-slot arrival count instead of
// interpreting thread guards again.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Phase 1: classify buffers and size their barriers.
// ---------------------------------------------------------------------------

/*!
 * \brief The record a lowering left next to one copy; false for any other
 *        statement (including a malformed record).
 */
bool RecordCall(const Stmt &stmt, CopyRecord *out) {
  const auto *eval = stmt.as<EvaluateNode>();
  const auto *call = eval == nullptr ? nullptr : eval->value.as<CallNode>();
  if (call == nullptr || !call->op.same_as(tl::tang_bulk_async_commit())) {
    return false;
  }
  return ReadRecord(call, out);
}

/*!
 * \brief A block-wide shared-memory rendezvous.
 *
 * The statement a copy's async form exists to remove: the lowering emits one on
 * each side of the transfer, and the handshake pass takes the load's trailing
 * one out together with the arrival it inserts.
 */
bool IsSharedSync(const Stmt &stmt) {
  const auto *eval = stmt.as<EvaluateNode>();
  const auto *call = eval == nullptr ? nullptr : eval->value.as<CallNode>();
  if (call == nullptr || !call->op.same_as(builtin::tvm_storage_sync()) ||
      call->args.size() != 1) {
    return false;
  }
  const auto *scope = call->args[0].as<StringImmNode>();
  return scope != nullptr && scope->value == "shared";
}

/*!
 * \brief The statement's type key, for diagnostics.
 */
std::string KindOf(const Stmt &stmt) {
  return stmt.defined() ? stmt->GetTypeKey() : "(none)";
}

/*!
 * \brief Whether ``call`` is an IR-emitted bulk g2s copy's leaf.
 *
 * Only the load side: this pass wires g2s, and lowering emits records only next
 * to loads.
 */
bool IsG2sCopyLeaf(const CallNode *call) {
  if (call == nullptr) {
    return false;
  }
  return call->op.same_as(tl::tang_fcpg2s_1d()) ||
         call->op.same_as(tl::tang_fcpg2s_3d());
}

/*!
 * \brief Whether ``stmt`` is the warp-group drain onto this fence group.
 *
 * The copy's own publication: the arrival replaces it, so that it waits for the
 * transfer *and* tells everyone else it is done, instead of only the former.
 */
bool IsFenceMem(const Stmt &stmt, int64_t fg) {
  const auto *eval = stmt.as<EvaluateNode>();
  const auto *call = eval == nullptr ? nullptr : eval->value.as<CallNode>();
  if (call == nullptr || !call->op.same_as(tl::tang_fence_mem()) ||
      call->args.size() != 1) {
    return false;
  }
  const auto *imm = call->args[0].as<IntImmNode>();
  return imm != nullptr && imm->value == fg;
}

using ReadyOwner = std::pair<int64_t, int64_t>;
using ReadyOwnersByBarrier =
    std::unordered_map<const VarNode *, std::vector<ReadyOwner>>;
using RecordsByLoop =
    std::unordered_map<const ForNode *, std::vector<CopyRecord>>;

class PipelineScanner : public StmtVisitor {
public:
  void Scan(const Stmt &body) {
    VisitStmt(body);
    // The v2 plan carried by each record is the synchronization authority.
    for (const CopyRecord &record : pending_records_) {
      if (record.fg != 0) {
        continue;
      }
      const bool pipeline_ready =
          record.plan.ready_owner ==
          tang::transform::PipelineReadyOwner::kInjectSoftwarePipeline;
      if (pipeline_ready != record.pipeline_barrier.defined()) {
        Refuse("a copy record disagrees with its ready-edge owner");
        continue;
      }
      Register(record.key, record.stage, record.arrive, record.num_warps,
               record.versions, pipeline_ready, record.plan);
      if (record.pipeline_barrier.defined()) {
        const auto *load = record.pipeline_barrier.value().as<BufferLoadNode>();
        if (load == nullptr) {
          Refuse("an ISP-owned ready edge has no barrier buffer");
          continue;
        }
        auto &owners = ready_owners_[load->buffer->data.get()];
        ReadyOwner owner{record.plan.loop_uid, record.plan.buffer_uid};
        if (std::find(owners.begin(), owners.end(), owner) == owners.end()) {
          owners.push_back(owner);
        }
      }
    }
  }

  const std::unordered_map<std::string, BufferInfo> &buffers() const {
    return buffers_;
  }
  const ReadyOwnersByBarrier &ready_owners() const { return ready_owners_; }
  const RecordsByLoop &loop_records() const { return loop_records_; }
  /*! \brief How many copy records the walk saw (registered or not). */
  size_t records_seen() const { return pending_records_.size(); }
  bool ok() const { return ok_; }
  // Why the scan refused, for the pass's diagnostics: only the first reason is
  // kept, and it is the one closest to the statement that forced the decision.
  const std::string &why() const { return why_; }

private:
  void Refuse(const std::string &reason) {
    ok_ = false;
    if (why_.empty()) {
      why_ = reason;
    }
  }
  void VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call != nullptr && call->op.same_as(tl::tang_bulk_async_commit())) {
      // The lowering's statement of what it just emitted. Collected here so
      // that the record's position in the statement list anchors it to its copy
      // -- "who issues it, in which loop, under which guard" is implied by
      // where it sits -- but registered only once the whole body is known (see
      // Scan): whether the copy may be wired at all depends on a consumer,
      // which may appear after it.
      CopyRecord record;
      if (!ReadRecord(call, &record)) {
        Refuse("a tl.tang_bulk_async_commit record is malformed");
        return;
      }
      pending_records_.push_back(record);
      for (const ForNode *loop : active_loops_) {
        loop_records_[loop].push_back(record);
      }
      return;
    }
    // Copies are classified only by their adjacent records.
    tirx::StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    active_loops_.push_back(op);
    tirx::StmtVisitor::VisitStmt_(op);
    active_loops_.pop_back();
  }

  /*!
   * \brief Register the copy of a (possibly rotated) buffer a record describes.
   */
  void Register(const std::string &key, const PrimExpr &stage, int64_t arrive,
                int64_t num_warps, int64_t versions_hint, bool pipeline_ready,
                const tang::transform::PipelinePlan &plan) {
    if (versions_hint != plan.versions) {
      Refuse("a copy record's version count disagrees with its v2 plan");
      return;
    }
    auto it = buffers_.find(key);
    const int ready_flag = pipeline_ready ? 1 : 0;
    if (it == buffers_.end()) {
      BufferInfo info;
      info.arrive = arrive;
      info.num_warps = num_warps;
      info.slot = static_cast<int64_t>(buffers_.size());
      info.engine_ready = ready_flag;
      info.loop_uid = plan.loop_uid;
      info.buffer_uid = plan.buffer_uid;
      info.distance = plan.distance;
      buffers_[key] = info;
      it = buffers_.find(key);
    } else if (it->second.loop_uid != plan.loop_uid ||
               it->second.buffer_uid != plan.buffer_uid ||
               it->second.distance != plan.distance) {
      Refuse("copies of one buffer disagree on their v2 pipeline identity");
      return;
    } else if (it->second.engine_ready != ready_flag) {
      Refuse("copies of one buffer disagree on their ready-edge owner");
      return;
    } else if (it->second.arrive != arrive ||
               it->second.num_warps != num_warps) {
      Refuse("two copies into the same buffer disagree on the barrier counts "
             "(one barrier cannot take two different arrive counts)");
      return;
    }
    if (stage.as<IntImmNode>() == nullptr) {
      int64_t versions = versions_hint;
      if (versions <= 1) {
        Refuse("a v2 pipeline copy record has no constant version count > 1");
        return;
      }
      if (it->second.versions != 0 && it->second.versions != versions) {
        Refuse("the same buffer is rotated by two different version counts");
        return;
      }
      it->second.versions = versions;
      it->second.rotated = true;
    } else {
      // A constant stage into a buffer whose copies rotate: that is the
      // prologue. A pipeline peeled by more than one iteration writes two of
      // them (a stage-3 pipeline peels 2), and the mbarrier form can only name
      // one with its single-use slot. The sync form can share the rotated slot
      // because its arrival is a count rather than a generation.
      const auto *imm = stage.as<IntImmNode>();
      if (std::find(it->second.pro_stages.begin(), it->second.pro_stages.end(),
                    imm->value) == it->second.pro_stages.end()) {
        it->second.pro_stages.push_back(imm->value);
      }
    }
  }

  std::unordered_map<std::string, BufferInfo> buffers_;
  ReadyOwnersByBarrier ready_owners_;
  RecordsByLoop loop_records_;
  std::vector<const ForNode *> active_loops_;
  /*! \brief Records seen during the walk; registered after it. */
  std::vector<CopyRecord> pending_records_;
  bool ok_{true};
  std::string why_;
};

// ---------------------------------------------------------------------------
// Phase 2: attach barrier ids and insert the consumer waits.
// ---------------------------------------------------------------------------
class AsyncCopyInjector : public StmtExprMutator {
public:
  using EdgeFact = tang::transform::AsyncEdgeFact;

  AsyncCopyInjector(const std::unordered_map<std::string, BufferInfo> &buffers,
                    std::unordered_map<std::string, int64_t> buffer_uids,
                    std::unordered_map<std::string, int64_t> buffer_loop_uids,
                    const RecordsByLoop &loop_records)
      : buffers_(buffers), buffer_uids_(std::move(buffer_uids)),
        buffer_loop_uids_(std::move(buffer_loop_uids)),
        loop_records_(loop_records) {}

  Stmt Inject(const Stmt &body) { return VisitStmt(body); }

  // False when a region the counts depend on could not be read. The caller then
  // keeps the function unchanged -- blocking copies -- instead of mis-sizing a
  // barrier, which would release a wait early or hang it.
  bool ok() const { return ok_; }
  const std::string &why() const { return why_; }

  /*! \brief Arrival count per free slot, for the mbarrier init table. */
  const std::unordered_map<int64_t, int64_t> &free_arrive_counts() const {
    return free_arrive_counts_;
  }

  const std::vector<EdgeFact> &edge_facts() const { return edge_facts_; }
  Stmt AnchorCertificates(Stmt body, const ReadyOwnersByBarrier &owners);

private:
  int64_t BufferUid(const std::string &key) const {
    auto it = buffer_uids_.find(key);
    return it == buffer_uids_.end() ? 0 : it->second;
  }

  int64_t LoopUidForBuffer(const std::string &key) const {
    auto it = buffer_loop_uids_.find(key);
    return it == buffer_loop_uids_.end() ? 0 : it->second;
  }

  Stmt Anchor(Stmt stmt, const std::string &key,
              AsyncEdgeAnchorRole role) const {
    const int64_t loop_uid =
        current_loop_uid_ > 0 ? current_loop_uid_ : LoopUidForBuffer(key);
    const int64_t buffer_uid = BufferUid(key);
    if (loop_uid <= 0 || buffer_uid <= 0) {
      return stmt;
    }
    return AnchorAsyncEdge(std::move(stmt), loop_uid, buffer_uid, role);
  }

  // Read the rotation from the typed plan. A copy without one stays blocking.
  bool PlanDistance(const std::string &key, int64_t *distance) {
    if (planned_.count(key) == 0)
      return false;
    auto info = buffers_.find(key);
    if (info == buffers_.end())
      return false;
    *distance = info->second.distance;
    return true;
  }

  void Refuse(const std::string &reason) {
    ok_ = false;
    if (why_.empty()) {
      why_ = reason;
    }
  }

  // mbarriers count arrivals only, so the wait may execute in a consumer
  // region whose warp count differs from the producer region.
  // get_mbarrier(i) prints as mbar[i] in codegen, so a runtime stage index
  // (``k % versions``) is fine.
  PrimExpr Mbar(const PrimExpr &slot) {
    return Call(DataType::Handle(), tl::get_mbarrier(), {slot});
  }

  Stmt MakeParityWait(const PrimExpr &slot, const PrimExpr &parity) {
    Array<PrimExpr> args{Mbar(slot), parity};
    return Evaluate(
        Call(DataType::Handle(), tl::tang_mbar_wait_parity(), args));
  }

  // Publish this warp's outstanding bulk copies into ready[stage]. Warp-level
  // and exactly one arrive (fence.arrive_mbar.noinc.mem): the whole point of
  // this edge is that the arrival count is the *copy's* warp count, whatever
  // the waiting side looks like. Never emulate it with a whole-warp
  // mbarrier_arrive, whose count is per call -- 32 lanes then post 32 arrivals
  // and desynchronize the phase.
  Stmt MakeMbarReadyArrive(const PrimExpr &slot) {
    return Evaluate(Call(DataType::Handle(),
                         tl::tang_fence_g2s_arrive_mbarrier(), {Mbar(slot)}));
  }

  // The n-th visit to a version waits for phase n, i.e. parity n & 1. Every
  // slot is visited once per ``versions`` iterations, so n = k / versions.
  // Starting at 0 for the first in-loop visit is right because the prologue's
  // copy already posted the arrival that flip belongs to.
  PrimExpr ReadyParity(const PrimExpr &k, int64_t versions) {
    const PrimExpr rounds = floordiv(k, IntImm(k.dtype(), versions));
    return floormod(rounds, IntImm(k.dtype(), 2));
  }

  // Publish "this version has been consumed" without stalling the issuing warp:
  // hardware posts the arrival when the calling warp's tensor-core group
  // retires, which is exactly what a buffer-reuse wait needs. The builtin takes
  // only the barrier; the fence group is the device default (fg_tc_default),
  // same as the blocking fence_tc<T> the MMA already sits behind.
  //
  Stmt MakeTcArrive(const PrimExpr &stage, const BufferInfo &info) {
    Array<PrimExpr> args{Mbar(FreeIdOf(info, stage))};
    return Evaluate(
        Call(DataType::Handle(), tl::tang_fence_tc_arrive_mbarrier(), args));
  }

  /*!
   * \brief The statement that publishes a copy, for the IR form.
   *
   * The IR copy has no helper argument that can carry the slot, so its
   * publication is emitted as a following statement.
   */
  Stmt MakeG2sArrival(const std::string &key, const BufferInfo &info,
                      const PrimExpr &stage) {
    (void)key;
    return MakeMbarReadyArrive(IdOf(info, stage));
  }

  class CertificateAnchorer : public StmtExprMutator {
  public:
    explicit CertificateAnchorer(const ReadyOwnersByBarrier &owners)
        : owners_(owners) {}

  private:
    Stmt VisitStmt_(const EvaluateNode *op) final {
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      const auto *eval = stmt.as<EvaluateNode>();
      const auto *call = eval == nullptr ? nullptr : eval->value.as<CallNode>();
      if (call == nullptr || call->args.empty()) {
        return stmt;
      }
      const auto *load = call->args[0].as<BufferLoadNode>();
      if (load == nullptr)
        return stmt;
      AsyncEdgeAnchorRole role;
      if (call->op.same_as(tl::tang_fence_g2s_arrive_mbarrier())) {
        role = AsyncEdgeAnchorRole::kReadyArrive;
      } else if (call->op.same_as(tl::mbarrier_wait_parity())) {
        role = AsyncEdgeAnchorRole::kReadyWait;
      } else {
        return stmt;
      }
      auto owners = owners_.find(load->buffer->data.get());
      if (owners == owners_.end())
        return stmt;
      for (auto it = owners->second.rbegin(); it != owners->second.rend();
           ++it) {
        stmt = AnchorAsyncEdge(std::move(stmt), it->first, it->second, role);
      }
      return stmt;
    }

    Stmt VisitStmt_(const SeqStmtNode *op) final {
      Stmt visited = StmtExprMutator::VisitStmt_(op);
      const auto *seq = visited.as<SeqStmtNode>();
      if (seq == nullptr)
        return visited;

      std::vector<Stmt> items(seq->seq.begin(), seq->seq.end());
      bool changed = false;
      for (size_t i = 0; i + 1 < items.size(); ++i) {
        const std::set<int64_t> waits =
            AnchorsWithRole(items[i], AsyncEdgeAnchorRole::kReadyWait);
        if (waits.empty() || !ContainsMma(items[i + 1]))
          continue;
        const std::set<int64_t> reads =
            AnchorsWithRole(items[i + 1], AsyncEdgeAnchorRole::kMmaRead);
        for (int64_t key : waits) {
          if (reads.count(key))
            continue;
          items[i + 1] = AttrStmt(
              Integer(0), tang::transform::kAsyncEdgeAnchorAttr,
              IntImm(DataType::Int(64),
                     key | static_cast<int64_t>(AsyncEdgeAnchorRole::kMmaRead)),
              items[i + 1]);
          changed = true;
        }
      }
      return changed ? Stmt(SeqStmt(items)) : visited;
    }

    static std::set<int64_t> AnchorsWithRole(const Stmt &stmt,
                                             AsyncEdgeAnchorRole role) {
      std::set<int64_t> keys;
      PostOrderVisit(stmt, [&](const ObjectRef &obj) {
        const auto *attr = obj.as<AttrStmtNode>();
        if (attr == nullptr ||
            attr->attr_key != tang::transform::kAsyncEdgeAnchorAttr)
          return;
        const auto *encoded = attr->value.as<IntImmNode>();
        if (encoded != nullptr &&
            tang::transform::AsyncEdgeRole(encoded->value) == role)
          keys.insert(tang::transform::AsyncEdgeKey(encoded->value));
      });
      return keys;
    }

    static bool ContainsMma(const Stmt &stmt) {
      bool found = false;
      PostOrderVisit(stmt, [&](const ObjectRef &obj) {
        if (const auto *call = obj.as<CallNode>())
          found = found || IsMmaOp(call);
      });
      return found;
    }

    const ReadyOwnersByBarrier &owners_;
  };

  // The consumer side follows from the producer's rotation: inside the loop
  // that stages version ``(k + D) % S``, the MMA at iteration k reads
  // ``k % S``, so one wait at the top of the body covers every consumer of that
  // buffer -- including a K-segmented MMA, which would otherwise reach the same
  // barrier several times per phase and over-count the consumers.
  Stmt VisitStmt_(const ForNode *op) final {
    // Plan the free edge on the *unrewritten* body first: the copy branch below
    // reads free_plan_ while it rewrites, and the prologue (constant stage, no
    // enclosing loop) then simply has no plan -- which is what keeps the
    // untouched versions out of the wait.
    const Var saved_var = current_loop_var_;
    const auto saved_plan = free_plan_;
    const auto saved_planned = planned_;
    const auto saved_mma_owners = current_mma_owners_;
    current_loop_var_ = op->loop_var;
    free_plan_.clear();
    planned_.clear();
    int64_t declared_loop_uid = 0;
    // Only buffers declared in this loop's typed plan can be wired.
    if (auto facts = op->annotations.Get(kPipelineFacts); facts.has_value()) {
      const Map<String, Any> map = Downcast<Map<String, Any>>(facts.value());
      for (const auto &kv : map) {
        const Array<Integer> value = Downcast<Array<Integer>>(kv.second);
        tang::transform::PipelinePlan plan;
        if (!tang::transform::ParsePipelinePlan(value, &plan)) {
          continue;
        }
        if (declared_loop_uid != 0 && declared_loop_uid != plan.loop_uid) {
          Refuse("one pipeline loop contains conflicting v2 loop identities");
          return GetRef<Stmt>(op);
        }
        declared_loop_uid = plan.loop_uid;
        planned_.insert(kv.first);
      }
    }
    // Both the waits and the free arrives are inserted at the body's level, so
    // this loop's region is what executes them, and its warp count is what the
    // barriers around them are sized with. ISP declares this count; -1 means a
    // surrounding thread guard prevented an exact count.
    int64_t loop_warps = -1;
    if (auto declared = op->annotations.Get(kLoopRegionWarps);
        declared.has_value()) {
      const auto *imm = declared.value().as<IntImmNode>();
      if (imm == nullptr) {
        Refuse("a loop's declared region warps is not an integer");
        return GetRef<Stmt>(op);
      }
      loop_warps = imm->value;
    }
    static const std::vector<CopyRecord> no_records;
    auto records = loop_records_.find(op);
    const std::vector<CopyRecord> &loop_records =
        records == loop_records_.end() ? no_records : records->second;
    {
      for (const CopyRecord &record : loop_records) {
        const std::string &key = record.key;
        const auto it = buffers_.find(key);
        if (it == buffers_.end() || !it->second.rotated) {
          continue;
        }
        if (free_plan_.count(key) != 0) {
          continue;
        }
        int64_t distance = 0;
        if (!PlanDistance(key, &distance)) {
          continue;
        }
        FreePlan plan;
        plan.arrive_warps = loop_warps;
        if (!it->second.PipelineOwnsReady()) {
          const auto *min = op->min.as<IntImmNode>();
          const auto *extent = op->extent.as<IntImmNode>();
          if (min == nullptr || extent == nullptr || extent->value < 1) {
            Refuse("the v2 plan cannot identify the peeled consumer because "
                   "the pipeline loop bounds are not constant");
            return GetRef<Stmt>(op);
          }
          const int64_t last_k = min->value + extent->value - 1;
          plan.epilogue_k = last_k + distance;
          plan.has_epilogue = true;
        }
        // The free edge is posted once per iteration by every warp of this
        // loop's region. Record its arrival count while the plan is live.
        for (int64_t s = 0; s < it->second.versions; ++s) {
          free_arrive_counts_[FreeSlotOf(it->second, s)] = loop_warps;
        }
        free_plan_[key] = plan;
      }
      // Do not partially wire a loop whose rotated copies disagree with its
      // typed plan.
      for (const CopyRecord &record : loop_records) {
        const auto it = buffers_.find(record.key);
        if (it == buffers_.end() || !it->second.rotated) {
          continue;
        }
        if (planned_.count(record.key) == 0) {
          Refuse("the loop stages " + record.key +
                 " without a v2 plan, so there is no certified distance for "
                 "its waits");
          return GetRef<Stmt>(op);
        }
      }
    }

    const bool pipelined = !free_plan_.empty();
    const int64_t saved_loop_uid = current_loop_uid_;
    if (pipelined) {
      if (declared_loop_uid <= 0) {
        Refuse("a pipelined loop has no stable async-edge uid");
        return GetRef<Stmt>(op);
      }
      current_loop_uid_ = declared_loop_uid;
      current_mma_owners_.clear();
      for (const auto &entry : free_plan_) {
        current_mma_owners_.push_back(entry.first);
      }
    }
    Stmt body = VisitStmt(op->body);
    std::vector<Stmt> waits;
    std::vector<Stmt> frees;
    std::vector<Stmt> epilogue_waits;
    std::set<std::string> seen;
    for (const CopyRecord &record : loop_records) {
      const std::string &key = record.key;
      if (!seen.insert(key).second) {
        continue;
      }
      const auto it = buffers_.find(key);
      if (it == buffers_.end() || !it->second.rotated) {
        continue;
      }
      const BufferInfo &info = it->second;
      if (planned_.count(key) == 0) {
        continue;
      }
      const int64_t versions = info.versions;
      ICHECK(versions > 1) << "a rotated buffer without a version count";
      const PrimExpr consumer_stage =
          floormod(op->loop_var, IntImm(op->loop_var.dtype(), versions));
      // The generic pipeline already waited this buffer's ready edge (it
      // created the barrier and inserted the phase wait). Waiting again here
      // would be on a slot this pass no longer arrives on.
      if (info.PipelineOwnsReady()) {
        // The free edge below is still this pass's.
      } else {
        if (!info.pro_stages.empty() && !PrologueCoversFirstVisits(info)) {
          Refuse("the prologue writes versions the consumer's first visits do "
                 "not read, so one ready generation sequence cannot order "
                 "them");
        }
        waits.push_back(
            Anchor(MakeParityWait(IdOf(info, consumer_stage),
                                  ReadyParity(op->loop_var, versions)),
                   key, AsyncEdgeAnchorRole::kReadyWait));
      }
      // Released at the *end* of the iteration so it covers every MMA that read
      // this version -- including the K-segmented ones -- and posts exactly
      // once per version per iteration.
      const auto plan = free_plan_.find(key);
      if (plan != free_plan_.end()) {
        frees.push_back(Anchor(MakeTcArrive(consumer_stage, info), key,
                               AsyncEdgeAnchorRole::kFreeArrive));
        edge_facts_.push_back(
            EdgeFact{current_loop_uid_, BufferUid(key), versions, info.distance,
                     /*ready_complete=*/1,
                     /*free_complete=*/1, info.num_warps, loop_warps});
        if (plan->second.has_epilogue) {
          const PrimExpr stage = IntImm(
              op->loop_var.dtype(), plan->second.epilogue_k % info.versions);
          const PrimExpr epilogue_k =
              IntImm(op->loop_var.dtype(), plan->second.epilogue_k);
          epilogue_waits.push_back(MakeParityWait(
              IdOf(info, stage), ReadyParity(epilogue_k, versions)));
        }
      }
    }
    current_loop_var_ = saved_var;
    current_loop_uid_ = saved_loop_uid;
    current_mma_owners_ = saved_mma_owners;
    free_plan_ = saved_plan;
    planned_ = saved_planned;
    Stmt loop = For(op->loop_var, op->min, op->extent, op->kind, body,
                    op->thread_binding, op->annotations, op->step, op->span);
    if (!waits.empty() || !frees.empty()) {
      std::vector<Stmt> body_stmts(waits);
      body_stmts.push_back(body);
      body_stmts.insert(body_stmts.end(), frees.begin(), frees.end());
      loop =
          For(op->loop_var, op->min, op->extent, op->kind, SeqStmt(body_stmts),
              op->thread_binding, op->annotations, op->step, op->span);
    }
    if (epilogue_waits.empty()) {
      return loop;
    }
    std::vector<Stmt> sequence{loop};
    sequence.insert(sequence.end(), epilogue_waits.begin(),
                    epilogue_waits.end());
    return SeqStmt(sequence);
  }

  /*!
   * \brief Wire one IR-emitted copy: the wait ahead of the leaf, the arrival
   *        where the copy's drain was.
   *
   * Both edits are anchored to the *leaf* rather than to the statement before
   * the record, because that statement is either the copy itself or the copy's
   * guard -- and when it is the guard, the wait has to go *inside* it, ahead of
   * the copy: the warps that pass that gate are the copy's warps, and a wait
   * placed after the transfer lets the DMA clobber a version the consumer is
   * still reading.
   */
  class WireCopy : public tirx::StmtExprMutator {
  public:
    WireCopy(Stmt wait, Stmt arrival, int64_t fg, int64_t loop_uid,
             int64_t buffer_uid)
        : wait_(std::move(wait)), arrival_(std::move(arrival)), fg_(fg),
          loop_uid_(loop_uid), buffer_uid_(buffer_uid) {}

    /*! \brief Whether the copy's leaf was found under this statement. */
    bool saw_leaf() const { return saw_leaf_; }
    /*! \brief Whether the drain was replaced by the arrival. */
    bool wired() const { return replaced_; }

    Stmt VisitStmt_(const EvaluateNode *op) final {
      const auto *call = op->value.as<CallNode>();
      if (IsG2sCopyLeaf(call)) {
        saw_leaf_ = true;
        Stmt leaf = tirx::StmtExprMutator::VisitStmt_(op);
        if (loop_uid_ > 0 && buffer_uid_ > 0) {
          leaf = AnchorAsyncEdge(std::move(leaf), loop_uid_, buffer_uid_,
                                 AsyncEdgeAnchorRole::kCopyWrite);
        }
        if (!waited_ && wait_.defined()) {
          waited_ = true;
          return SeqStmt({wait_, leaf});
        }
        return leaf;
      } else if (saw_leaf_ && !replaced_ && arrival_.defined() &&
                 IsFenceMem(GetRef<Stmt>(op), fg_)) {
        // The copy's own drain, which is what the arrival replaces. Only after
        // the leaf: a drain of some other statement is not this copy's.
        replaced_ = true;
        return arrival_;
      }
      return tirx::StmtExprMutator::VisitStmt_(op);
    }

  private:
    Stmt wait_;
    Stmt arrival_;
    int64_t fg_{0};
    int64_t loop_uid_{0};
    int64_t buffer_uid_{0};
    bool saw_leaf_{false};
    bool waited_{false};
    bool replaced_{false};
  };

  /*!
   * \brief Wire an IR-emitted copy from the record that sits next to it.
   *
   * The lowering states a load as ``[..., guarded, record, sync, ...]``: the
   * guard holds the leaf and the warp group's fence, the record says which
   * buffer version it filled, and the trailing sync is the block rendezvous the
   * async form exists to remove. The handshake is therefore a rewrite of three
   * *siblings*, and a single-pass mutator cannot see the statement before the
   * one it is visiting -- which is why this is done at the sequence level
   * rather than in VisitStmt_ of the guard or of the record.
   *
   * Whatever this cannot wire keeps its blocking form and synchronization.
   */
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tang::transform::kAsyncCopyRecordScopeAttr) {
      ++record_scope_depth_;
      Stmt body = VisitStmt(op->body);
      --record_scope_depth_;
      return body;
    }
    return tirx::StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    const Stmt visited = tirx::StmtExprMutator::VisitStmt_(op);
    const auto *seq = visited.as<SeqStmtNode>();
    if (seq == nullptr || record_scope_depth_ == 0) {
      return visited;
    }
    const size_t count = seq->seq.size();
    const bool explain = std::getenv("TL_S3_ASYNC_COPY_REPORT") != nullptr;
    const auto say = [&](const std::string &reason, const std::string &key) {
      if (explain) {
        LOG(WARNING) << "[InjectS3AsyncCopy] record for " << key
                     << " skipped: " << reason;
      }
    };
    std::vector<std::pair<size_t, Stmt>> rewritten;
    std::set<size_t> dropped;
    // ``i < count``, not ``i + 1 < count``: the rendezvous after the record is
    // optional (tl.tang_bulk_copy_emitter_syncs), so a record may sit last.
    for (size_t i = 1; i < count; ++i) {
      CopyRecord record;
      if (!RecordCall(seq->seq[i], &record)) {
        continue;
      }
      if (record.fg != 0) {
        // Only the g2s direction is wired, and the lowering only states a
        // record for a load; anything else is not this pass's business.
        say("not a g2s record", record.key);
        continue;
      }
      // Only a buffer an MMA reads can be wired: the wait this pass inserts
      // goes in front of that MMA, and without one there is nothing to wait --
      // which is exactly the gate the scanner registers records under.
      //
      // The rendezvous after the record is optional: the emitter states it
      // today (tl.tang_bulk_copy_emitter_syncs) and this pass removes it as
      // part of the handshake, but once the emitters stop placing barriers
      // there is simply nothing to remove and the copy has to wire the same
      // way.
      const auto it = buffers_.find(record.key);
      const bool has_sync = i + 1 < count && IsSharedSync(seq->seq[i + 1]);
      if (it == buffers_.end()) {
        say("no MMA consumer registered for this buffer", record.key);
        continue;
      }
      // The emitter already posted this copy's ready arrival on the pipeline
      // barrier, inside the same guard as the drain. Replacing the drain with
      // another arrival would both drop that drain and arrive twice.
      const bool pipeline_ready = record.pipeline_barrier.defined();
      const Stmt arrival =
          pipeline_ready
              ? Stmt()
              : Anchor(MakeG2sArrival(record.key, it->second, record.stage),
                       record.key, AsyncEdgeAnchorRole::kReadyArrive);
      // Free edge: the wait the op form puts in front of its copy, at the same
      // spot -- a version may only be reused once the MMA that read it has
      // retired; the first ``versions - distance`` a rotation produces are
      // produced for the first time, so waiting there could only deadlock.
      Stmt wait;
      const auto plan = free_plan_.find(record.key);
      if (plan != free_plan_.end()) {
        if (plan->second.arrive_warps <= 0) {
          Refuse("the warps that publish a buffer's free edge could not be "
                 "counted");
          continue;
        }
        // The producer waits for the consumption of the
        // version it is about to overwrite, i.e. the consumer's read at
        // iteration ``k' = k + distance - versions``, which is the
        // ``(k + distance) / versions - 1``-th arrival that free slot has seen.
        const PrimExpr k = current_loop_var_;
        PrimExpr rounds = floordiv(k + IntImm(k.dtype(), it->second.distance),
                                   IntImm(k.dtype(), it->second.versions));
        wait = MakeParityWait(
            FreeIdOf(it->second, record.stage),
            floormod(rounds - IntImm(k.dtype(), 1), IntImm(k.dtype(), 2)));
        // The first ``versions - distance`` iterations produce a version
        // nothing has consumed yet, so there is no arrival to wait for: waiting
        // there could only deadlock. That is exactly ``k' >= 0`` above.
        const int64_t first_consumed =
            it->second.versions - it->second.distance;
        if (first_consumed > 0) {
          wait =
              IfThenElse(GE(current_loop_var_,
                            IntImm(current_loop_var_.dtype(), first_consumed)),
                         wait);
        }
        wait =
            Anchor(std::move(wait), record.key, AsyncEdgeAnchorRole::kFreeWait);
      }
      // Two shapes carry the copy, depending on whether its warp gate survived
      // as IR: ``[guarded, record, sync]`` when it did, and
      // ``[copy, drain, record, sync]`` when it folded away. Wiring is anchored
      // to the leaf either way, so neither shape needs its own rule.
      bool wired = false;
      {
        WireCopy wire(wait, arrival, record.fg, LoopUidForBuffer(record.key),
                      BufferUid(record.key));
        Stmt head = wire(seq->seq[i - 1]);
        if (wire.saw_leaf()) {
          // An engine copy is wired once the leaf is found: its arrival is
          // already in the guard, so the drain must stay. The record and the
          // trailing block rendezvous still go.
          wired = pipeline_ready || wire.wired();
          rewritten.emplace_back(i - 1, head);
        } else if (!pipeline_ready && i >= 2 &&
                   IsFenceMem(seq->seq[i - 1], record.fg)) {
          // The drain is the sibling right after the copy: put the wait ahead
          // of the leaf, and let the drain itself become the arrival.
          WireCopy copy_only(wait, Stmt(), record.fg,
                             LoopUidForBuffer(record.key),
                             BufferUid(record.key));
          Stmt copy = copy_only(seq->seq[i - 2]);
          wired = copy_only.saw_leaf();
          rewritten.emplace_back(i - 2, copy);
          rewritten.emplace_back(i - 1, arrival);
        }
      }
      if (!wired) {
        // A record that stays unwired is a copy that keeps the blocking form:
        // worth saying, because it is silent otherwise (and the reason is
        // usually the shape of the statement list around it, not the copy).
        if (explain) {
          LOG(WARNING) << "[InjectS3AsyncCopy] record for " << record.key
                       << " not wired: position " << i << "/" << count
                       << ", neighbours " << KindOf(seq->seq[i - 1]) << " | "
                       << (i + 1 < count ? KindOf(seq->seq[i + 1]) : "(end)")
                       << " after";
        }
        continue;
      }
      dropped.insert(i);
      if (has_sync) {
        dropped.insert(i + 1);
      }
    }
    if (dropped.empty()) {
      return visited;
    }
    std::vector<Stmt> items(seq->seq.begin(), seq->seq.end());
    for (const auto &pair : rewritten) {
      items[pair.first] = pair.second;
    }
    std::vector<Stmt> out;
    for (size_t i = 0; i < count; ++i) {
      if (dropped.count(i) == 0) {
        out.push_back(items[i]);
      }
    }
    // Dropping the last two statements of a sequence can leave a single one
    // behind, and a length-1 SeqStmt is rejected outright -- the copy's own
    // guard is the usual survivor. An empty result cannot happen: every record
    // dropped takes a guard with it, and that guard stays.
    if (out.size() == 1) {
      return out[0];
    }
    return SeqStmt(out);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr) {
      return tirx::StmtExprMutator::VisitStmt_(op);
    }
    Stmt stmt = tirx::StmtExprMutator::VisitStmt_(op);
    if (IsMmaOp(call) && current_loop_uid_ > 0) {
      // free_plan_ contains exactly the rotated operands this loop paired with
      // an MMA consumer during the scanner/planner phase. Use that validated
      // pairing rather than re-keying the lowered access_ptr aliases here:
      // buffer renaming can change their printed Var while the copy record's
      // stable key deliberately does not.
      for (const std::string &key : current_mma_owners_) {
        stmt = Anchor(std::move(stmt), key, AsyncEdgeAnchorRole::kMmaRead);
      }
    }
    return stmt;
  }

  // State for the free (buffer-reuse) edge: the loop being visited and the
  // rotation of each pipelined buffer inside it.
  struct FreePlan {
    int64_t arrive_warps{0}; // warps that post the free arrive
    bool has_epilogue{false};
    int64_t epilogue_k{0};
  };

  const std::unordered_map<std::string, BufferInfo> &buffers_;
  const std::unordered_map<std::string, int64_t> buffer_uids_;
  const std::unordered_map<std::string, int64_t> buffer_loop_uids_;
  const RecordsByLoop &loop_records_;
  bool ok_{true};
  std::string why_;
  Var current_loop_var_;
  std::unordered_map<std::string, FreePlan> free_plan_;
  // Arrival counts of the free slots, keyed by slot: built while the plan is
  // live (VisitStmt_(ForNode)) because the init table is assembled afterwards,
  // where only the numbers survive.
  std::unordered_map<int64_t, int64_t> free_arrive_counts_;
  // A buffer absent from this loop's typed plan is not wired.
  std::set<std::string> planned_;
  std::vector<EdgeFact> edge_facts_;
  // The validated operands owned by the enclosing pipelined loop. Keep this
  // separate from free_plan_: nested scalar loops around the MMA save and
  // clear their local plan while this ownership must remain visible.
  std::vector<std::string> current_mma_owners_;
  int64_t current_loop_uid_{0};
  int record_scope_depth_{0};
};

Stmt AsyncCopyInjector::AnchorCertificates(Stmt body,
                                           const ReadyOwnersByBarrier &owners) {
  return CertificateAnchorer(owners)(std::move(body));
}

// The shared mbarrier prologue lives in s3_mbarrier_utils.h.

} // namespace

/*!
 * \brief Materialize S3 async-copy edges from typed pipeline plans.
 *
 * No-op on targets other than stcuv2, and for functions whose copies cannot be
 * classified exactly (see the failure policy in the file header).
 */
tvm::transform::Pass InjectS3AsyncCopy() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      const PassContext &ctx) -> PrimFunc {
    Optional<Target> opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!opt_target.defined() || !TargetTangIsSTCUV2(opt_target.value())) {
      return f;
    }
    // InjectSoftwarePipeline already owns the ready edge in full: barrier
    // allocation, arrive count, producer annotation and consumer wait. This
    // pass must not repair or reinterpret that count; it only wires the
    // independent free (overwrite) edge.
    // Diagnostics: with TL_S3_ASYNC_COPY_REPORT=1 the pass says why a pipelined
    // loop kept its blocking copies. Off by default, because a loop with
    // nothing to rotate refuses quite normally and a warning per kernel would
    // be CI noise -- but for anyone chasing a pipeline that never overlapped,
    // this is the first thing they want to know. Without it a refusal is
    // silent: the kernel stays correct, just serialized, so there is nothing to
    // notice.
    const bool report_on = [] {
      const char *env = std::getenv("TL_S3_ASYNC_COPY_REPORT");
      return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    std::string label = "kernel";
    if (auto sym = f->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
      label = sym.value();
    }
    auto report = [&](const std::string &reason) {
      if (report_on) {
        LOG(WARNING) << "[InjectS3AsyncCopy] " << label
                     << ": keeping the blocking copies -- " << reason;
      }
    };
    PipelineScanner scanner;
    scanner.Scan(f->body);
    if (!scanner.ok()) {
      report(scanner.why());
      return f;
    }
    std::unordered_map<std::string, BufferInfo> buffers = scanner.buffers();
    std::vector<std::string> buffer_keys;
    for (const auto &entry : buffers) {
      if (entry.second.rotated) {
        buffer_keys.push_back(entry.first);
      }
    }
    std::sort(buffer_keys.begin(), buffer_keys.end());
    std::unordered_map<std::string, int64_t> buffer_uids;
    std::unordered_map<std::string, int64_t> buffer_loop_uids;
    for (const std::string &key : buffer_keys) {
      const BufferInfo &info = buffers.at(key);
      buffer_uids.emplace(key, info.buffer_uid);
      buffer_loop_uids.emplace(key, info.loop_uid);
    }
    int64_t last_id = 0;
    if (!PlanBarriers(&buffers, &last_id)) {
      return f; // nothing pipelined: leave the blocking copies alone
    }
    const BlockThreads threads = FindBlockThreads(f->body);
    if (threads.warps <= 0) {
      // Without the block's warp count no barrier can be sized, and a barrier
      // with the wrong count is worse than a blocking copy.
      report("the block's threadIdx.x extent is not a constant, so no barrier "
             "can be sized");
      return f;
    }
    if (report_on) {
      // What the scan saw and what it could size: a copy that stays blocking
      // is usually a copy whose buffer never entered this map, and that is a
      // different fact from "the pass refused".
      LOG(WARNING) << "[InjectS3AsyncCopy] " << label << ": saw "
                   << scanner.records_seen() << " bulk-copy record(s), sized "
                   << buffers.size() << " pipelined buffer(s)";
    }
    AsyncCopyInjector injector(buffers, std::move(buffer_uids),
                               std::move(buffer_loop_uids),
                               scanner.loop_records());
    Stmt body = injector.Inject(f->body);
    if (!injector.ok()) {
      report(injector.why());
      return f;
    }
    if (body.same_as(f->body)) {
      return f;
    }
    body = injector.AnchorCertificates(std::move(body), scanner.ready_owners());
    // Every slot some warp arrives on is initialised with that warp count:
    // ready slots with the copy's warps, and free slots with the warps that
    // post the TC arrival.
    // The init table follows the plan: every slot that carries a count of its
    // own takes it from the table, and the free slots keep the default until
    // the injector states theirs below -- theirs is the count of the warps
    // that post the TC arrival, which is only known once the waits exist.
    std::vector<int64_t> arrive_counts(static_cast<size_t>(last_id) + 1, 1);
    for (const auto &entry : buffers) {
      const BufferInfo &info = entry.second;
      if (!info.rotated) {
        continue;
      }
      for (int64_t stage = 0; stage < info.versions; ++stage) {
        arrive_counts[static_cast<size_t>(info.ready_base + 2 * stage)] =
            info.arrive;
      }
      const int64_t base = info.ready_base;
      if (report_on) {
        LOG(WARNING) << "[InjectS3AsyncCopy] " << label
                     << ": barrier slots for " << entry.first
                     << " -- ready base " << base << " x " << info.versions
                     << " versions, arrive " << info.arrive << ", warps "
                     << info.num_warps;
      }
    }
    // The free edge's slots: initialised with the warps that post the TC
    // arrival, which the injector recorded while the free plan was live.
    for (const auto &entry : injector.free_arrive_counts()) {
      const auto slot = static_cast<size_t>(entry.first);
      if (slot < arrive_counts.size()) {
        arrive_counts[slot] = entry.second;
      }
    }
    // State the representation once, in full: who owns each slot, what it is
    // for and what it counts. Off by default (a per-kernel table is CI
    // noise); the pinning test uses it to verify the complete encoding.
    if (std::getenv("TL_DEBUG_BARRIER_PLAN") != nullptr) {
      size_t slot_count = 0;
      for (const auto &entry : buffers) {
        const BufferInfo &info = entry.second;
        if (!info.rotated) {
          continue;
        }
        for (int64_t stage = 0; stage < info.versions; ++stage) {
          for (int role = 0; role < 2; ++role) {
            const int64_t id = info.ready_base + 2 * stage + role;
            LOG(WARNING) << "[InjectS3AsyncCopy] " << label << ": slot " << id
                         << " " << (role == 0 ? "ready" : "free") << " "
                         << entry.first << " stage " << stage << " arrive "
                         << arrive_counts[static_cast<size_t>(id)];
            ++slot_count;
          }
        }
      }
      LOG(WARNING) << "[InjectS3AsyncCopy] " << label << ": " << slot_count
                   << " barrier slot(s), highest id " << last_id;
    }
    ThreadExtentPrologue inserter(
        arrive_counts, "the pipeline's mbarrier array", threads.thread_x,
        /*base=*/0,
        /*disable_shuffle_elect=*/
        ctx->GetConfig<Bool>(kDisableShuffleElect, Bool(false)).value());
    Stmt with_prologue = inserter.Insert(body);
    if (!inserter.inserted()) {
      report("the pipeline's mbarrier init has nowhere to go: no threadIdx.x "
             "extent wraps the kernel body");
      return f;
    }
    body = with_prologue;
    PrimFunc result(f->params, body, f->ret_type, f->buffer_map, f->attrs,
                    f->span);
    Map<String, Array<Integer>> facts;
    for (const AsyncCopyInjector::EdgeFact &fact : injector.edge_facts()) {
      if (fact.loop_uid <= 0 || fact.buffer_uid <= 0 || !fact.ready_complete ||
          !fact.free_complete) {
        continue;
      }
      const std::string key =
          std::to_string(fact.loop_uid) + ":" + std::to_string(fact.buffer_uid);
      facts.Set(String(key), tang::transform::EncodeAsyncEdgeFact(fact));
    }
    if (!facts.empty()) {
      result = WithAttr(std::move(result), kAsyncEdgeFactsAttr, facts);
    }
    return result;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.tang.InjectS3AsyncCopy", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.tang.transform.InjectS3AsyncCopy",
                        InjectS3AsyncCopy);
}

} // namespace tl
} // namespace tvm
