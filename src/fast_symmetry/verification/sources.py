"""The frozen 512 deterministic source indices used by both exact verifiers."""

import numpy as np

SOURCE_COUNT = 512


def deterministic_stride_indices(count: int) -> np.ndarray:
    if count < SOURCE_COUNT:
        raise ValueError(f"exact readout requires at least {SOURCE_COUNT} points")
    return (np.arange(SOURCE_COUNT, dtype=np.int64) * count // SOURCE_COUNT).astype(
        np.int32
    )
