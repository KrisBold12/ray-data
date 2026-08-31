import ray
import time

from ray_data.dataset import create_mock_dataset
from ray_data.decode import decode, SECONDS_FOR_SIMULATION


DS_SIZE = 1000
CORES_ON_MACHINE_CPU = 8


def main():
    ds = ray.data.from_items(create_mock_dataset(DS_SIZE))
    print(f"Size: {ds.count()}\nSchema: {ds.schema()}\nSamples: {ds.take(2)}\n# blocks: {ds.num_blocks()}")

    ds = ds.map_batches(decode)

    start = time.perf_counter()
    mat = ds.materialize()
    end = time.perf_counter()

    print(f"Execution time: {end - start} | Expected execution time: {DS_SIZE * SECONDS_FOR_SIMULATION / CORES_ON_MACHINE_CPU} | Throughput: {DS_SIZE / (end - start)}")
    print(mat.stats())


if __name__ == "__main__":
    main()