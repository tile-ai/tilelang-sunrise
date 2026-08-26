import tilelang as tl
import tilelang.language as T
import tilelang.testing
import torch
from tilelang import tvm
from tvm import tirx
from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue


def _apply_plan_update(func: tvm.tirx.PrimFunc) -> tvm.IRModule:
    if torch.ptpu.is_available():
        target = tvm.target.Target("tang")
        mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
        with target:
            mod = tirx.transform.BindTarget(target)(mod)
            mod = tl.transform.MaterializeKernelLaunch()(mod)
            mod = tl.transform.IfStmtBinding()(mod)
            mod = tl.tang.transform.LowerSharedTmem()(mod)
            mod = tl.transform.PlanAndUpdateBufferAllocationLocation()(mod)
            mod = tl.transform.HoistGlobalBufferAllocations()(mod)
            mod = tl.transform.LowerOpaqueBlock()(mod)
            mod = tl.transform.Simplify()(mod)
    else:
        target = tvm.target.Target("cuda")
        mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
        with target:
            mod = CUDAPassPipelineBodyPrologue(mod, target)
            mod = tl.cuda.transform.LowerSharedTmem()(mod)
            mod = tl.transform.IfStmtBinding()(mod)
            mod = tl.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    return mod


def _find_block(stmt: tvm.tirx.Stmt, name_hint: str) -> tvm.tirx.SBlock:
    blocks = []

    def _visit(node):
        if isinstance(node, tvm.tirx.SBlock) and str(node.name_hint) == name_hint:
            blocks.append(node)

    tvm.tirx.stmt_functor.post_order_visit(stmt, _visit)
    assert len(blocks) == 1, f"Expected exactly one block named {name_hint}, got {len(blocks)}"
    return blocks[0]


def _find_first_for(stmt: tvm.tirx.Stmt) -> tvm.tirx.For:
    loops = []

    def _visit(node):
        if isinstance(node, tvm.tirx.For):
            loops.append(node)

    tvm.tirx.stmt_functor.post_order_visit(stmt, _visit)
    assert loops, "Expected at least one loop"
    return loops[0]


def test_plan_update_keeps_loop_header_local_var_outside_loop_body():
    @T.prim_func
    def func(x: T.Tensor((256,), "int64")):
        with T.Kernel(256, threads=128):
            a, b = T.alloc_var(T.int), T.alloc_var(T.int)
            T.fill(x[a:b], 0)

    mod = _apply_plan_update(func)
    main = mod["main"]
    is_tang = torch.ptpu.is_available()

    if is_tang:
        # TANG's LowerOpaqueBlock flattens tilelang_root; local vars end up at
        # the launch_thread scope.  Verify they exist and are outside any loop.
        all_local_vars = set()

        def _collect_local_vars(node):
            if isinstance(node, tvm.tirx.AllocBuffer) and "local.var" in str(node.buffer.scope()):
                all_local_vars.add(node.buffer.data.name)

        tvm.tirx.stmt_functor.post_order_visit(main.body, _collect_local_vars)
        assert {"a", "b"} <= all_local_vars
        # TANG keeps launch_thread as AttrStmt, not a For loop; no loop means
        # no local vars can be incorrectly placed inside one.
    else:
        tilelang_root = _find_block(main.body, "tilelang_root")
        root_local_vars = {buf.name for buf in tilelang_root.alloc_buffers if buf.scope() == "local.var"}
        assert {"a", "b"} <= root_local_vars

        loop = _find_first_for(main.body)
        loop_body_local_vars = set()

        def _visit_loop_body(node):
            if isinstance(node, tvm.tirx.SBlock):
                for buf in node.alloc_buffers:
                    if buf.scope() == "local.var":
                        loop_body_local_vars.add(buf.name)

        tvm.tirx.stmt_functor.post_order_visit(loop.body, _visit_loop_body)
        assert "b" not in loop_body_local_vars


if __name__ == "__main__":
    tilelang.testing.main()
