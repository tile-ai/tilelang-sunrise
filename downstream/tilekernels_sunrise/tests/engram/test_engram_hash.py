import os

import pytest
import torch

from tile_kernels.config import get_device, get_is_tang
from tile_kernels.engram import engram_hash
from tile_kernels.testing.bench import make_param_id
from tile_kernels.testing.generator import generate_num_tokens, get_test_level
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.torch.engram import engram_hash_ref, make_offsets

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    num_tokens = params['num_tokens']
    max_ngram_size = params['ngram']
    num_ngram_layers = params['layers']
    num_embed_table_per_ngram = params['tables']
    device = get_device()
    ngram_token_ids = torch.randint(0, 100000, (num_tokens, max_ngram_size), dtype=torch.int32, device=device)
    multipliers = torch.randint(0, 100000, (num_ngram_layers, max_ngram_size), dtype=torch.int64, device=device)
    vocab_sizes = torch.randint(
        100000,
        1000000,
        (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram),
        dtype=torch.int32,
        device=device,
    )
    offsets = make_offsets(vocab_sizes)
    image_token_mask = torch.rand(num_tokens, device=device) < 0.5 if params['with_mask'] else None
    return (ngram_token_ids, multipliers, vocab_sizes, offsets, image_token_mask)


def generate_test_params(level: int, mask_options: tuple[bool, ...] = (False, True)) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'ngram': max_ngram_size,
            'layers': 2,
            'tables': 8,
            'with_mask': with_mask,
        }
        for num_tokens in generate_num_tokens(level)
        for max_ngram_size in (3, 4)
        for with_mask in mask_options
    ]


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_engram_hash(params):
    ngram_token_ids, multipliers, vocab_sizes, offsets, image_token_mask = generate_test_data(params)

    output = engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets, image_token_mask)
    if get_is_tang():
        assert output.device == ngram_token_ids.device
        torch.ptpu.synchronize()
        reference_inputs = tuple(
            tensor.cpu() if tensor is not None else None
            for tensor in (ngram_token_ids, multipliers, vocab_sizes, offsets, image_token_mask)
        )
        output_ref = engram_hash_ref(*reference_inputs)
        assert output.dtype == output_ref.dtype
        assert output.shape == output_ref.shape
        assert output.numel() == 0 or output.stride() == output_ref.stride()
        assert_equal(output.cpu(), output_ref)
        return
    output_ref = engram_hash_ref(ngram_token_ids, multipliers, vocab_sizes, offsets, image_token_mask)
    assert_equal(output, output_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0, mask_options=(False,)), ids=make_param_id)
def test_engram_hash_benchmark(benchmark_timer, benchmark_record, params):
    ngram_token_ids, multipliers, vocab_sizes, offsets, image_token_mask = generate_test_data(params)
    output = engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets, image_token_mask)

    t_us = benchmark_timer(lambda: engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets, image_token_mask))

    num_bytes = count_bytes(ngram_token_ids, multipliers, vocab_sizes, offsets, image_token_mask, output)
    bandwidth_gbs = num_bytes / t_us / 1e3
    benchmark_record(
        kernel='engram_hash',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
