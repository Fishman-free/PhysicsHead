"""Minimal integration with an externally installed OpenSTL TAU model.

OpenSTL is optional and is intentionally not bundled by this package. The caller
constructs and loads ``tau_model`` using OpenSTL, then passes it to the helper.
"""

from __future__ import annotations

import torch

from physicshead import PhysicsHead, reshape_sequence_outputs


def run_with_proxies(
    tau_model: torch.nn.Module,
    observed_frames: torch.Tensor,
    head: PhysicsHead,
    output_steps: int = 5,
):
    """Run TAU and decode matching shared-feature slots with PhysicsHead."""
    captured: dict[str, torch.Tensor] = {}

    def capture_readout_input(_module, inputs):
        captured["features"] = inputs[0]

    handle = tau_model.dec.readout.register_forward_pre_hook(capture_readout_input)
    try:
        rgb_prediction = tau_model(observed_frames)
    finally:
        handle.remove()

    features = captured["features"]
    batch_size = observed_frames.shape[0]
    total_steps = features.shape[0] // batch_size
    support_flat, motion_flat = head(features)
    support, motion = reshape_sequence_outputs(
        support_flat, motion_flat, batch_size, total_steps
    )
    return rgb_prediction, support[:, :output_steps], motion[:, :output_steps]


if __name__ == "__main__":
    raise SystemExit("Construct an OpenSTL TAU model and call run_with_proxies().")
