import torch

import tilelang.testing
from inference_kernel import ref_sparse_attn, sparse_attn


def test_sparse_attn():
    """Test with small parameters"""
    B, S, H, D = 1, 128, 64, 64
    SKV = 256
    topk = 64

    torch.manual_seed(42)
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16).ptpu()
    kv = torch.randn(B, SKV, D, dtype=torch.bfloat16).ptpu()
    attn_sink = torch.randn(H, dtype=torch.float32).ptpu()
    topk_idxs = torch.randint(0, SKV, (B, S, topk), dtype=torch.int32).ptpu()
    softmax_scale = 1.0 / (D**0.5)

    c = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

    q_cpu = q.clone().cpu().float()
    kv_cpu = kv.clone().cpu().float()
    attn_sink_cpu = attn_sink.clone().cpu().float()
    topk_idxs_cpu = topk_idxs.clone().cpu()
    ref_c = ref_sparse_attn(q_cpu, kv_cpu, attn_sink_cpu, topk_idxs_cpu, softmax_scale)

    c_cpu = c.cpu()
    ref_c_cpu = ref_c.cpu()
    diff = torch.abs(c_cpu.float() - ref_c_cpu)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print("Test 1: Small parameters")
    print(f"  B={B}, S={S}, H={H}, D={D}, SKV={SKV}, topk={topk}")
    print(f"  Output shape: {c.shape}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Pass: {max_diff < 1e-2}")
    print()


def test_sparse_attn_deepseek_v4():
    """Test with DeepSeek-V4 model parameters (reduced for shared memory)"""
    # The full D=512, topk=512 configuration exceeds TANG shared memory.
    B, S, H, D = 1, 128, 64, 256
    SKV = 512
    topk = 256

    torch.manual_seed(42)
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16).ptpu()
    kv = torch.randn(B, SKV, D, dtype=torch.bfloat16).ptpu()
    attn_sink = torch.randn(H, dtype=torch.float32).ptpu()
    topk_idxs = torch.randint(0, SKV, (B, S, topk), dtype=torch.int32).ptpu()
    softmax_scale = 1.0 / (D**0.5)

    c = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

    q_cpu = q.clone().cpu().float()
    kv_cpu = kv.clone().cpu().float()
    attn_sink_cpu = attn_sink.clone().cpu().float()
    topk_idxs_cpu = topk_idxs.clone().cpu()
    ref_c = ref_sparse_attn(q_cpu, kv_cpu, attn_sink_cpu, topk_idxs_cpu, softmax_scale)

    c_cpu = c.cpu()
    ref_c_cpu = ref_c.cpu()
    diff = torch.abs(c_cpu.float() - ref_c_cpu)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print("Test 2: DeepSeek-V4 parameters")
    print(f"  B={B}, S={S}, H={H}, D={D}, SKV={SKV}, topk={topk}")
    print(f"  Output shape: {c.shape}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Pass: {max_diff < 1e-2}")


if __name__ == "__main__":
    tilelang.testing.main()
