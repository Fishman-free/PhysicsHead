import physicshead


def test_public_api_is_intentionally_small():
    assert physicshead.__all__ == [
        "PhysicsHead",
        "reshape_sequence_outputs",
        "MaskSupervisionLoss",
        "FlowSupervisionLoss",
        "WarpingLoss",
    ]
    assert physicshead.PhysicsHead.__name__ == "PhysicsHead"
