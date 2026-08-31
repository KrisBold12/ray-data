import time
import numpy as np

NUM_FRAMES = 8
SECONDS_FOR_SIMULATION = 0.05


def decode(batch: dict[str, np.ndarray]):
    n = len(batch["id"])
    frames = []
    for i in range(n):
        time.sleep(SECONDS_FOR_SIMULATION) # Simulate decoding
        frames.append(np.random.randint(0, 256, size=(NUM_FRAMES, 64, 64, 3), dtype=np.uint8))
    batch["frames"] = np.stack(frames)
    return batch