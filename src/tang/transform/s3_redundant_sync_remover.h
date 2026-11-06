/*
 * S3-specific redundant synchronization removal for certified async edges.
 */
#ifndef TVM_TL_TANG_TRANSFORM_S3_REDUNDANT_SYNC_REMOVER_H_
#define TVM_TL_TANG_TRANSFORM_S3_REDUNDANT_SYNC_REMOVER_H_

#include <tvm/tirx/stmt.h>

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_set>

namespace tvm {
namespace tl {
namespace tang {
namespace transform {

struct S3RedundantSyncRemovalOptions {
  std::unordered_set<int64_t> certified_edges;
  bool enable_async_edge_certificates{false};
  bool preserve_uncertified_loop_barriers{false};
  std::string async_edge_anchor_attr;
  int64_t async_copy_write_role{0};
  int64_t async_consumer_read_role{0};
  std::function<bool(const tirx::CallNode *)> is_ordering_only_call;
  std::function<bool(const tirx::CallNode *)> requires_following_sync_call;
};

tirx::Stmt RemoveCertifiedS3Syncs(const tirx::Stmt &stmt,
                                  S3RedundantSyncRemovalOptions options);

} // namespace transform
} // namespace tang
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TANG_TRANSFORM_S3_REDUNDANT_SYNC_REMOVER_H_
