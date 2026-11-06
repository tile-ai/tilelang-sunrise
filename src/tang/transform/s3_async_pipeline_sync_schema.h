/*!
 * \file src/tang/transform/s3_async_pipeline_sync_schema.h
 * \brief Durable schema for S3 async-pipeline dependency certificates.
 */
#ifndef TVM_TL_TANG_TRANSFORM_S3_ASYNC_PIPELINE_SYNC_SCHEMA_H_
#define TVM_TL_TANG_TRANSFORM_S3_ASYNC_PIPELINE_SYNC_SCHEMA_H_

#include <tvm/tirx/stmt.h>

#include <string>
#include <utility>

namespace tvm {
namespace tl {
namespace tang {
namespace transform {

constexpr const char *kPipelinePlanAttr = "tl.pipeline_dependency_facts";
constexpr const char *kPipelineLoopUidAttr = "tl.pipeline_loop_uid";
constexpr const char *kPipelineBufferUidAttr = "tl.pipeline_buffer_uid";
constexpr const char *kPipelineVersionsAttr = "tl.pipeline_versions";
constexpr const char *kPipelineDistanceAttr = "tl.pipeline_distance";
constexpr const char *kPipelineProducerStageAttr = "tl.pipeline_producer_stage";
constexpr const char *kPipelineConsumerStageAttr = "tl.pipeline_consumer_stage";
constexpr const char *kPipelineReadyOwnerAttr = "tl.pipeline_ready_owner";
constexpr const char *kAsyncCopyRecordScopeAttr =
    "tl.tang_async_copy_record_scope";
constexpr int64_t kPipelinePlanSchemaVersion = 2;

enum class PipelineReadyOwner : int64_t {
  kS3AsyncCopy = 0,
  kInjectSoftwarePipeline = 1,
};

struct PipelinePlan {
  int64_t loop_uid{0};
  int64_t buffer_uid{0};
  int64_t versions{0};
  int64_t distance{0};
  int64_t producer_stage{-1};
  int64_t consumer_stage{-1};
  PipelineReadyOwner ready_owner{PipelineReadyOwner::kS3AsyncCopy};
};

inline ffi::Array<Integer> EncodePipelinePlan(const PipelinePlan &plan) {
  return {Integer(kPipelinePlanSchemaVersion),
          Integer(plan.loop_uid),
          Integer(plan.buffer_uid),
          Integer(plan.versions),
          Integer(plan.distance),
          Integer(plan.producer_stage),
          Integer(plan.consumer_stage),
          Integer(static_cast<int64_t>(plan.ready_owner))};
}

inline bool ParsePipelinePlan(const ffi::Array<Integer> &encoded,
                              PipelinePlan *plan) {
  if (encoded.size() != 8 || encoded[0]->value != kPipelinePlanSchemaVersion ||
      encoded[1]->value <= 0 || encoded[2]->value <= 0 ||
      encoded[3]->value <= 1 || encoded[4]->value <= 0 ||
      encoded[4]->value >= encoded[3]->value || encoded[5]->value < 0 ||
      encoded[6]->value <= encoded[5]->value ||
      encoded[6]->value - encoded[5]->value != encoded[4]->value ||
      (encoded[7]->value !=
           static_cast<int64_t>(PipelineReadyOwner::kS3AsyncCopy) &&
       encoded[7]->value !=
           static_cast<int64_t>(PipelineReadyOwner::kInjectSoftwarePipeline))) {
    return false;
  }
  plan->loop_uid = encoded[1]->value;
  plan->buffer_uid = encoded[2]->value;
  plan->versions = encoded[3]->value;
  plan->distance = encoded[4]->value;
  plan->producer_stage = encoded[5]->value;
  plan->consumer_stage = encoded[6]->value;
  plan->ready_owner = static_cast<PipelineReadyOwner>(encoded[7]->value);
  return true;
}

constexpr const char *kAsyncEdgeFactsAttr = "tl.tang_async_edge_facts";
constexpr const char *kAsyncEdgeAnchorAttr = "tl.tang_async_edge_anchor";
constexpr int64_t kAsyncEdgeSchemaVersion = 2;

struct AsyncEdgeFact {
  int64_t loop_uid{0};
  int64_t buffer_uid{0};
  int64_t versions{0};
  int64_t distance{0};
  int64_t ready_complete{0};
  int64_t free_complete{0};
  int64_t copy_warps{0};
  int64_t consumer_warps{0};
};

inline ffi::Array<Integer> EncodeAsyncEdgeFact(const AsyncEdgeFact &fact) {
  return {Integer(kAsyncEdgeSchemaVersion), Integer(fact.loop_uid),
          Integer(fact.buffer_uid),         Integer(fact.versions),
          Integer(fact.distance),           Integer(fact.ready_complete),
          Integer(fact.free_complete),      Integer(fact.copy_warps),
          Integer(fact.consumer_warps)};
}

inline bool ParseAsyncEdgeFact(const ffi::Array<Integer> &encoded,
                               AsyncEdgeFact *fact) {
  if (encoded.size() < 9 || encoded[0]->value != kAsyncEdgeSchemaVersion ||
      encoded[1]->value <= 0 || encoded[2]->value <= 0 ||
      encoded[3]->value <= 1 || encoded[4]->value <= 0 ||
      encoded[4]->value >= encoded[3]->value || encoded[5]->value != 1 ||
      encoded[6]->value != 1 || encoded[7]->value <= 0 ||
      encoded[8]->value <= 0) {
    return false;
  }
  fact->loop_uid = encoded[1]->value;
  fact->buffer_uid = encoded[2]->value;
  fact->versions = encoded[3]->value;
  fact->distance = encoded[4]->value;
  fact->ready_complete = encoded[5]->value;
  fact->free_complete = encoded[6]->value;
  fact->copy_warps = encoded[7]->value;
  fact->consumer_warps = encoded[8]->value;
  return true;
}

enum class AsyncEdgeAnchorRole : int64_t {
  kCopyWrite = 1,
  kMmaRead = 2,
  kReadyArrive = 3,
  kReadyWait = 4,
  kFreeArrive = 5,
  kFreeWait = 6,
};

inline int64_t EncodeAsyncEdgeAnchor(int64_t loop_uid, int64_t buffer_uid,
                                     AsyncEdgeAnchorRole role) {
  return (loop_uid << 32) | (buffer_uid << 8) | static_cast<int64_t>(role);
}

inline int64_t AsyncEdgeKey(int64_t encoded) {
  return encoded & ~int64_t{0xff};
}

inline AsyncEdgeAnchorRole AsyncEdgeRole(int64_t encoded) {
  return static_cast<AsyncEdgeAnchorRole>(encoded & int64_t{0xff});
}

inline tirx::Stmt AnchorAsyncEdge(tirx::Stmt body, int64_t loop_uid,
                                  int64_t buffer_uid,
                                  AsyncEdgeAnchorRole role) {
  return tirx::AttrStmt(
      Integer(0), kAsyncEdgeAnchorAttr,
      IntImm(DataType::Int(64),
             EncodeAsyncEdgeAnchor(loop_uid, buffer_uid, role)),
      std::move(body));
}

} // namespace transform
} // namespace tang
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TANG_TRANSFORM_S3_ASYNC_PIPELINE_SYNC_SCHEMA_H_
