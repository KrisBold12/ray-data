import pyarrow as pa

LUMINOSITY_THRESHOLD = 127.5


def quality_filter(batch: pa.Table) -> pa.Table:
    """Drop clips whose mean luminosity is below the threshold.

    Filtering in Arrow rather than NumPy keeps the tensor type on "frames" when a
    batch is emptied out: Ray only routes a column through ArrowTensorArray when
    it has at least one row, so returning 0-row NumPy arrays makes it fall back to
    pickling the column as python objects.
    """
    if batch.num_rows == 0:
        return batch

    frames = batch["frames"].combine_chunks().to_numpy(zero_copy_only=False)
    luminosity = frames.mean(axis=(1, 2, 3, 4))
    return batch.filter(pa.array(luminosity > LUMINOSITY_THRESHOLD))
