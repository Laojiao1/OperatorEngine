"""文件职责：验证 Transformer Block 基线、逐项算子替换、causal 语义和输出契约。"""

import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "transformer_block.py"
MODULE_SPEC = importlib.util.spec_from_file_location("operator_engine_transformer_block", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"无法加载 Transformer Block 示例：{MODULE_PATH}")

TRANSFORMER_BLOCK_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(TRANSFORMER_BLOCK_MODULE)

RMSNorm = TRANSFORMER_BLOCK_MODULE.RMSNorm
TorchTransformerBlock = TRANSFORMER_BLOCK_MODULE.TorchTransformerBlock
OperatorEngineAttentionBlock = TRANSFORMER_BLOCK_MODULE.OperatorEngineAttentionBlock
OperatorEngineGemmBlock = TRANSFORMER_BLOCK_MODULE.OperatorEngineGemmBlock
OperatorEngineTransformerBlock = TRANSFORMER_BLOCK_MODULE.OperatorEngineTransformerBlock
TorchUnfusedAttentionBlock = TRANSFORMER_BLOCK_MODULE.TorchUnfusedAttentionBlock
OperatorEngineSoftmaxBlock = TRANSFORMER_BLOCK_MODULE.OperatorEngineSoftmaxBlock
apply_rotary_embedding = TRANSFORMER_BLOCK_MODULE.apply_rotary_embedding


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA CUDA GPU")


def test_rms_norm_matches_explicit_reference():
    torch.manual_seed(0)
    hidden_states = torch.randn((2, 5, 64), dtype=torch.float32)
    norm = RMSNorm(64, eps=1e-5)
    norm.weight.data.uniform_(0.5, 1.5)

    actual = norm(hidden_states)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    expected = hidden_states * torch.rsqrt(variance + norm.eps) * norm.weight

    torch.testing.assert_close(actual, expected)


def test_rope_preserves_position_zero_and_pair_norms():
    torch.manual_seed(1)
    q = torch.randn((2, 3, 7, 64), dtype=torch.float32)
    k = torch.randn_like(q)

    q_rotated, k_rotated = apply_rotary_embedding(q, k)

    torch.testing.assert_close(q_rotated[:, :, 0], q[:, :, 0])
    torch.testing.assert_close(k_rotated[:, :, 0], k[:, :, 0])

    q_pair_norm = q.reshape(2, 3, 7, 32, 2).pow(2).sum(dim=-1)
    q_rotated_pair_norm = q_rotated.reshape(2, 3, 7, 32, 2).pow(2).sum(dim=-1)
    torch.testing.assert_close(q_rotated_pair_norm, q_pair_norm, atol=1e-5, rtol=1e-5)


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("hidden_size", "num_heads", "intermediate_size", "shape"),
    [(256, 4, 512, (2, 17, 256)), (512, 4, 768, (1, 8, 512))],
)
def test_torch_transformer_block_preserves_output_contract(hidden_size, num_heads, intermediate_size, shape):
    torch.manual_seed(2)
    block = TorchTransformerBlock(hidden_size, num_heads, intermediate_size).cuda().half().eval()
    hidden_states = torch.randn(shape, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        output = block(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert output.device == hidden_states.device
    assert torch.isfinite(output).all()


@CUDA_REQUIRED
def test_torch_transformer_block_is_causal():
    torch.manual_seed(3)
    block = TorchTransformerBlock(256, 4, 512).cuda().half().eval()
    original = torch.randn((1, 8, 256), device="cuda", dtype=torch.float16)
    changed_future = original.clone()
    changed_future[:, 4:] = torch.randn_like(changed_future[:, 4:])

    with torch.inference_mode():
        original_output = block(original)
        changed_output = block(changed_future)

    # 前四个位置看不到未来 token，因此它们的输出应保持不变。
    torch.testing.assert_close(original_output[:, :4], changed_output[:, :4], atol=2e-2, rtol=2e-2)
    assert not torch.allclose(original_output[:, 4:], changed_output[:, 4:])


@CUDA_REQUIRED
def test_torch_transformer_block_zero_input_stays_zero():
    block = TorchTransformerBlock(256, 4, 512).cuda().half().eval()
    hidden_states = torch.zeros((1, 8, 256), device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        output = block(hidden_states)

    torch.testing.assert_close(output, hidden_states)


@pytest.mark.parametrize(
    "arguments",
    [(0, 4, 512), (256, 0, 512), (256, 4, 0), (192, 4, 512)],
)
def test_torch_transformer_block_rejects_invalid_configuration(arguments):
    with pytest.raises(ValueError):
        TorchTransformerBlock(*arguments)


def test_torch_transformer_block_rejects_invalid_input_shape():
    block = TorchTransformerBlock(256, 4, 512)

    with pytest.raises(ValueError, match="三维"):
        block(torch.randn((8, 256)))
    with pytest.raises(ValueError, match="hidden_size"):
        block(torch.randn((1, 8, 128)))


def test_attention_replaced_block_has_identical_state_dict_structure():
    baseline = TorchTransformerBlock(256, 4, 512)
    replaced = OperatorEngineAttentionBlock(256, 4, 512)

    assert baseline.state_dict().keys() == replaced.state_dict().keys()
    replaced.load_state_dict(baseline.state_dict())


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("provider", "hidden_size", "num_heads", "intermediate_size", "sequence_length"),
    [
        ("sdpa", 256, 4, 512, 17),
        ("triton", 256, 4, 512, 33),
        ("auto", 256, 4, 512, 33),
        ("cpp", 256, 4, 512, 17),
        ("cpp", 256, 4, 512, 33),
        ("cpp", 512, 4, 768, 17),
    ],
)
def test_attention_replacement_matches_same_weight_baseline(provider, hidden_size, num_heads, intermediate_size, sequence_length):
    torch.manual_seed(4)
    baseline = TorchTransformerBlock(hidden_size, num_heads, intermediate_size).cuda().half().eval()
    replaced = OperatorEngineAttentionBlock(hidden_size, num_heads, intermediate_size, attention_provider=provider).cuda().half().eval()
    replaced.load_state_dict(baseline.state_dict())
    hidden_states = torch.randn((1, sequence_length, hidden_size), device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        expected = baseline(hidden_states)
        actual = replaced(hidden_states)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


def test_attention_replaced_block_rejects_unknown_provider():
    with pytest.raises(ValueError, match="attention_provider"):
        OperatorEngineAttentionBlock(256, 4, 512, attention_provider="unknown")


def test_gemm_replaced_block_has_identical_state_dict_structure():
    baseline = TorchTransformerBlock(256, 4, 512)
    replaced = OperatorEngineGemmBlock(256, 4, 512)

    assert baseline.state_dict().keys() == replaced.state_dict().keys()
    replaced.load_state_dict(baseline.state_dict())


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("hidden_size", "num_heads", "intermediate_size", "sequence_length"),
    [(256, 4, 512, 17), (256, 4, 512, 128), (512, 4, 768, 17)],
)
def test_gemm_replacement_matches_same_weight_baseline(hidden_size, num_heads, intermediate_size, sequence_length):
    torch.manual_seed(5)
    baseline = TorchTransformerBlock(hidden_size, num_heads, intermediate_size).cuda().half().eval()
    replaced = OperatorEngineGemmBlock(hidden_size, num_heads, intermediate_size).cuda().half().eval()
    replaced.load_state_dict(baseline.state_dict())
    hidden_states = torch.randn((1, sequence_length, hidden_size), device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        expected = baseline(hidden_states)
        actual = replaced(hidden_states)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


def test_gemm_replaced_block_keeps_pytorch_attention():
    assert OperatorEngineGemmBlock._attention_forward is TorchTransformerBlock._attention_forward


def test_combined_block_has_expected_structure_and_method_routing():
    baseline = TorchTransformerBlock(256, 4, 512)
    combined = OperatorEngineTransformerBlock(256, 4, 512, attention_provider="auto")

    assert baseline.state_dict().keys() == combined.state_dict().keys()
    assert combined._attention_forward.__func__ is OperatorEngineAttentionBlock._attention_forward
    assert combined._linear_forward.__func__ is OperatorEngineGemmBlock._linear_forward
    combined.load_state_dict(baseline.state_dict())


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("hidden_size", "num_heads", "intermediate_size", "sequence_length"),
    [(256, 4, 512, 17), (512, 4, 768, 17)],
)
def test_combined_block_matches_same_weight_baseline(hidden_size, num_heads, intermediate_size, sequence_length):
    torch.manual_seed(6)
    baseline = TorchTransformerBlock(hidden_size, num_heads, intermediate_size).cuda().half().eval()
    combined = OperatorEngineTransformerBlock(hidden_size, num_heads, intermediate_size, attention_provider="auto").cuda().half().eval()
    combined.load_state_dict(baseline.state_dict())
    hidden_states = torch.randn((1, sequence_length, hidden_size), device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        expected = baseline(hidden_states)
        actual = combined(hidden_states)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


def test_unfused_softmax_blocks_have_expected_structure_and_method_routing():
    baseline = TorchTransformerBlock(256, 4, 512)
    torch_unfused = TorchUnfusedAttentionBlock(256, 4, 512)
    custom_softmax = OperatorEngineSoftmaxBlock(256, 4, 512)

    assert baseline.state_dict().keys() == torch_unfused.state_dict().keys()
    assert baseline.state_dict().keys() == custom_softmax.state_dict().keys()
    assert custom_softmax._attention_forward.__func__ is TorchUnfusedAttentionBlock._attention_forward
    assert custom_softmax._softmax_forward.__func__ is OperatorEngineSoftmaxBlock._softmax_forward


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("hidden_size", "num_heads", "intermediate_size", "sequence_length"),
    [(256, 4, 512, 17), (512, 4, 768, 17)],
)
def test_torch_unfused_attention_matches_fused_baseline(hidden_size, num_heads, intermediate_size, sequence_length):
    torch.manual_seed(7)
    baseline = TorchTransformerBlock(hidden_size, num_heads, intermediate_size).cuda().half().eval()
    torch_unfused = TorchUnfusedAttentionBlock(hidden_size, num_heads, intermediate_size).cuda().half().eval()
    torch_unfused.load_state_dict(baseline.state_dict())
    hidden_states = torch.randn((1, sequence_length, hidden_size), device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        expected = baseline(hidden_states)
        actual = torch_unfused(hidden_states)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("hidden_size", "num_heads", "intermediate_size", "sequence_length"),
    [(256, 4, 512, 17), (256, 4, 512, 128)],
)
def test_custom_softmax_block_matches_torch_unfused_attention(hidden_size, num_heads, intermediate_size, sequence_length):
    torch.manual_seed(8)
    torch_unfused = TorchUnfusedAttentionBlock(hidden_size, num_heads, intermediate_size).cuda().half().eval()
    custom_softmax = OperatorEngineSoftmaxBlock(hidden_size, num_heads, intermediate_size).cuda().half().eval()
    custom_softmax.load_state_dict(torch_unfused.state_dict())
    hidden_states = torch.randn((1, sequence_length, hidden_size), device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        expected = torch_unfused(hidden_states)
        actual = custom_softmax(hidden_states)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
