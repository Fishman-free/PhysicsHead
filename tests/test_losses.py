import torch

from physicshead import FlowSupervisionLoss, MaskSupervisionLoss, WarpingLoss


def test_mask_loss_perfect_and_empty_targets_are_finite():
    loss_fn = MaskSupervisionLoss()
    target = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
    prediction = target.clamp(1e-6, 1.0 - 1e-6)
    assert loss_fn(prediction, target).item() < 2e-6

    empty = torch.zeros(2, 1, 1, 3, 3)
    prediction_empty = torch.full_like(empty, 1e-6)
    loss = loss_fn(prediction_empty, empty)
    assert torch.isfinite(loss)
    assert loss >= 0


def test_flow_loss_perfect_empty_and_gradient():
    loss_fn = FlowSupervisionLoss()
    motion = torch.zeros(1, 2, 2, 3, 3, requires_grad=True)
    target = torch.zeros_like(motion)
    mask = torch.ones(1, 2, 1, 3, 3)
    assert loss_fn(motion, target, mask).item() == 0.0

    nonzero = torch.ones_like(motion, requires_grad=True)
    empty_mask = torch.zeros_like(mask)
    empty_loss = loss_fn(nonzero, target, empty_mask)
    assert empty_loss.item() == 0.0
    empty_loss.backward()
    assert torch.equal(nonzero.grad, torch.zeros_like(nonzero))


def test_warping_loss_perfect_t_less_than_two_and_detach():
    loss_fn = WarpingLoss()
    mask = torch.zeros(1, 2, 1, 5, 5, requires_grad=True)
    with torch.no_grad():
        mask[:, :, :, 2, 2] = 1.0
    motion = torch.zeros(1, 2, 2, 5, 5, requires_grad=True)
    loss = loss_fn(mask, motion)
    assert loss.item() == 0.0
    loss.backward()
    assert motion.grad is not None
    assert mask.grad is None

    single_motion = torch.zeros(1, 1, 2, 5, 5)
    single_mask = torch.zeros(1, 1, 1, 5, 5)
    assert loss_fn(single_mask, single_motion).item() == 0.0


def test_warping_gradient_for_mismatched_masks():
    loss_fn = WarpingLoss()
    mask = torch.zeros(1, 2, 1, 7, 7)
    mask[:, 0, 0, 3, 2] = 1.0
    mask[:, 1, 0, 3, 3] = 1.0
    motion = torch.zeros(1, 2, 2, 7, 7, requires_grad=True)
    loss_fn(mask, motion).backward()
    assert motion.grad is not None
    assert motion.grad.abs().sum() > 0
