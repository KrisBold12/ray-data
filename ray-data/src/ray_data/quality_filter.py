import numpy as np
import ray

LUMINOSITY_THRESHOLD = 127.5


def quality_filter(batch: dict[str, np.ndarray]):
    luminosity = batch["frames"].mean(axis=(1, 2, 3, 4))
    mask = luminosity > LUMINOSITY_THRESHOLD
    return {k: v[mask] for k, v in batch.items()}