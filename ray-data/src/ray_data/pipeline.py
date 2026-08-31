import ray
import time

from ray_data.dataset import create_mock_dataset
from ray_data.decode import decode


def main():
    ds = ray.data.from_items(create_mock_dataset(1000))
    print(f"Size: {ds.count()}\nSchema: {ds.schema()}\nSamples: {ds.take(2)}\n# blocks: {ds.num_blocks()}")

    ds = ds.map_batches(decode)

    start = time.perf_counter()
    mat = ds.materialize()
    end = time.perf_counter()

    print(f"Execution time: {end - start} | Expected execution time: {1000 * 0.05 / 8} | Throughput: {1000 / (end - start)}")
    print(mat.stats())


if __name__ == "__main__":
    main()