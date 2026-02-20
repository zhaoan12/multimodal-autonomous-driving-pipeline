"""Dataset split helpers for manifest-driven experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class DatasetSplit:
    """Serializable train/validation/test split specification."""

    train: list[str]
    val: list[str]
    test: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return asdict(self)


def create_dataset_split(
    scene_paths: list[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> DatasetSplit:
    """Create a deterministic split by sorted path order."""

    ordered = [str(path).replace("\\", "/") for path in sorted(scene_paths)]
    total = len(ordered)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return DatasetSplit(
        train=ordered[:train_end],
        val=ordered[train_end:val_end],
        test=ordered[val_end:],
    )
