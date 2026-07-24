import pytest
import torch

from physicshead import PhysicsHead, reshape_sequence_outputs


def test_default_parameter_count_and_shapes():
    head = PhysicsHead()
    assert sum(parameter.numel() for parameter in head.parameters()) == 18530
    features = torch.randn(6, 64, 12, 10)
    support, motion = head(features)
    assert support.shape == (6, 1, 12, 10)
    assert motion.shape == (6, 2, 12, 10)
    assert torch.all((support >= 0) & (support <= 1))


def test_motion_scale_is_linear_with_matched_weights():
    base = PhysicsHead(motion_scale=1.0).eval()
    scaled = PhysicsHead(motion_scale=5.0).eval()
    scaled.load_state_dict(base.state_dict())
    features = torch.randn(3, 64, 9, 11)
    base_support, base_motion = base(features)
    scaled_support, scaled_motion = scaled(features)
    assert torch.allclose(base_support, scaled_support)
    assert torch.allclose(scaled_motion, base_motion * 5.0)
    assert scaled_motion.abs().max() <= 5.0 + 1e-6


def test_gradients_reach_both_branches():
    head = PhysicsHead()
    features = torch.randn(2, 64, 8, 8, requires_grad=True)
    support, motion = head(features)
    (support.mean() + motion.square().mean()).backward()
    assert features.grad is not None
    assert features.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_reshape_sequence_outputs():
    support = torch.randn(6, 1, 4, 5)
    motion = torch.randn(6, 2, 4, 5)
    support_seq, motion_seq = reshape_sequence_outputs(support, motion, 2, 3)
    assert support_seq.shape == (2, 3, 1, 4, 5)
    assert motion_seq.shape == (2, 3, 2, 4, 5)
    assert torch.equal(support_seq.reshape_as(support), support)
    with pytest.raises(ValueError):
        reshape_sequence_outputs(support, motion, 4, 2)
