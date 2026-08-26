"""FFI APIs for tilelang"""

import tvm_ffi

# TVM_REGISTER_GLOBAL("tl.name").set_body_typed(func);
tvm_ffi.init_ffi_api("tl", __name__)

# Route TANG feature queries to backend-specific globals.
_target_has_bulk_copy = globals()["TargetHasBulkCopy"]
_target_tang_has_bulk_copy = globals()["TargetTangHasBulkCopy"]
_target_tang_has_tmem = globals()["TargetTangHasTmem"]


def TargetHasBulkCopy(target):  # noqa: N802
    if target.kind.name == "tang":
        return _target_tang_has_bulk_copy(target)
    return _target_has_bulk_copy(target)


def TargetHasTmem(target):  # noqa: N802
    if target.kind.name == "tang":
        return _target_tang_has_tmem(target)
    return False
