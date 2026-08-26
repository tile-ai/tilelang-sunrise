import tilelang
import tilelang.testing
import tilelang.language as T
import torch
from tilelang import tvm
from tvm import tirx
from tvm.tirx import op
from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue
from tilelang.transform import LowerAccessPtr


def issue_2123_atomic_load_repro(num_tiles, threads=32):
    @T.prim_func
    def kernel(status: T.Tensor((num_tiles,), T.int32), out: T.Tensor((1,), T.int32)):
        with T.Kernel(num_tiles, threads=threads) as tile:
            look = T.alloc_var(T.int32)
            state = T.alloc_var(T.int32)
            done = T.alloc_var(T.bool)
            tx = T.get_thread_binding()
            if tx == 0:
                look = tile - 1
                done = look < 0
                state = 0
                while not done:
                    state = T.atomic_load(status[look], memory_order="acquire")
                    if state != 0:
                        done = True
                    else:
                        look -= 1
                        done = look < 0
                if tile == num_tiles - 1:
                    out[0] = state

    return kernel


def _has_op_call(func, op_name):
    found = False
    target_op = op.Op.get(op_name)

    def _visit(node):
        nonlocal found
        if isinstance(node, tirx.Call) and node.op.same_as(target_op):
            found = True

    tirx.stmt_functor.post_order_visit(func.body, _visit)
    return found


def _assert_access_ptr_lowered(mod):
    assert _has_op_call(mod["main"], "tirx.tvm_access_ptr")
    assert not _has_op_call(mod["main"], "tl.access_ptr")


def test_issue_2123_atomic_load_lower_access_ptr_direct():
    func = issue_2123_atomic_load_repro(4).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)

    lowered = LowerAccessPtr()(mod)

    _assert_access_ptr_lowered(lowered)


def test_issue_2123_atomic_load_lower_access_ptr_pipeline():
    if torch.ptpu.is_available():
        target = tvm.target.Target("tang")
        func = issue_2123_atomic_load_repro(4).with_attr("global_symbol", "main")
        mod = tvm.IRModule.from_expr(func)
        with target:
            mod = tirx.transform.BindTarget(target)(mod)
            mod = tilelang.transform.MaterializeKernelLaunch()(mod)
            mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
            mod = tilelang.transform.LegalizeNegativeIndex()(mod)
            mod = tilelang.transform.InjectAssumes()(mod)
            mod = tilelang.transform.Simplify()(mod)
            mod = tilelang.transform.LayoutReducer()(mod)
            mod = tilelang.transform.IfStmtBinding()(mod)
            mod = tilelang.transform.PipelinePlanning()(mod)
            mod = tilelang.transform.InjectSoftwarePipeline()(mod)
            mod = tilelang.transform.Simplify()(mod)
            mod = tilelang.transform.LayoutInference()(mod)
            mod = tilelang.transform.LowerTileOp()(mod)
            mod = tilelang.transform.DecoupleTypeCast()(mod)
            mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
            mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
            mod = tilelang.transform.LowerAccessPtr()(mod)
            mod = tilelang.transform.Simplify()(mod)
        lowered = mod
    else:
        target = tvm.target.Target("cuda", host="llvm")
        func = issue_2123_atomic_load_repro(4).with_attr("global_symbol", "main")
        mod = tvm.IRModule.from_expr(func)
        lowered = CUDAPassPipelineBodyPrologue(mod, target)

    _assert_access_ptr_lowered(lowered)


if __name__ == "__main__":
    tilelang.testing.main()
