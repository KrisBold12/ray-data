from pathlib import Path


def create_mock_dataset(n: int) -> list[dict]:
    return [{"id": i, "path": str(Path.cwd() / str(i))} for i in range(n)]

