# PhysicsHead

PhysicsHead is a compact, removable PyTorch auxiliary head for video-prediction
backbones. It reads a shared 64-channel decoder feature through two sibling
branches:

- a sigmoid branch supervised by precomputed binary masks, interpreted as a
  **visible-support proxy** rather than measured dye concentration; and
- a tanh stream-function branch whose replicate-padded centered rotated
  gradient is interpreted as an **image-plane motion proxy**, not calibrated
  fluid velocity.

The head does not sit in the RGB prediction path. The accompanying checkpoint
numbers are a paired preservation audit, not evidence that the auxiliary head
causally improves RGB prediction.

## Installation

```bash
python -m pip install .
```

PyTorch is the only required runtime dependency. OpenSTL is optional and appears
only in `examples/tau_integration.py`; install it separately when integrating
with a TAU model.

## Usage

```python
import torch
from physicshead import PhysicsHead, reshape_sequence_outputs

batch_size, steps = 2, 5
features = torch.randn(batch_size * steps, 64, 64, 64)
head = PhysicsHead(in_channels=64, motion_scale=5.0)
support_flat, motion_flat = head(features)
support, motion = reshape_sequence_outputs(
    support_flat, motion_flat, batch_size, steps
)
```

`PhysicsHead.forward` deliberately returns flat tensors matching common decoder
readout layouts:

- support: `(B*T, 1, H, W)`, in `[0, 1]`;
- motion: `(B*T, 2, H, W)`.

The default module contains exactly 18,530 trainable parameters. Its two
branches each use `Conv(64→16, 3×3, no bias) → BatchNorm → ReLU → Conv(16→1,
1×1)`. The support branch ends in sigmoid. The motion branch ends in a bounded
stream function and converts it to `(dψ/dy, -dψ/dx)` with replicate-padded
centered differences before applying `motion_scale`.

## Losses

The package exposes:

- `MaskSupervisionLoss`: BCE plus `0.5 ×` per-sample Dice loss by default;
- `FlowSupervisionLoss`: masked MSE by default, with optional Smooth L1;
- `WarpingLoss`: adjacent-mask backward-warp consistency with detached targets.

No PDE residual is included. For the reported training convention, offline
apparent-flow targets and head motion used the same scale factor; divide the
motion proxy by that factor before `WarpingLoss` to restore the pixel-frame
warp convention.

## Reported audit and provenance

`results/checkpoint_audit.csv` contains the two 6,664-window checkpoint rows at
full recorded precision. The run fitted the head while backbone trainable
tensors were frozen; the selected row is epoch 9. These measurements establish
checkpoint preservation only and are not a causal comparison.

`results/provenance.json` records the release counts and accounting definitions:
328 archive clips, 309 release clips from 46 sources, 30,247 windows, and zero
source overlap between partitions. Data, masks, optical-flow targets,
checkpoints, logs, older designs, and version-control history are excluded from
this source release.

## Testing

```bash
python -m pytest
python -m build
```

## License

MIT. See `LICENSE` and `THIRD_PARTY_NOTICE.md`.
