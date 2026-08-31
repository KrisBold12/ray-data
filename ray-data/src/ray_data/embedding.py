import time
import numpy as np

from ray_data.decode import NUM_FRAMES


EMBEDDING_VECTOR_SIZE = 32


class MockEmbedder:
    def __init__(self):
        time.sleep(1) # Simulate model loading (CLIP / DINO)
        self.proj_mat = np.random.randn(64 * 64 * 3 * NUM_FRAMES, EMBEDDING_VECTOR_SIZE).astype(np.float32)

    def __call__(self, batch: dict[str, np.ndarray]):
        reshaped_frames = batch["frames"].reshape(len(batch["frames"]), -1).astype(np.float32)
        emb = reshaped_frames @ self.proj_mat
        batch["embedding"] = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        return batch
                                                                          