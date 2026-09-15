from dataclasses import dataclass


@dataclass(frozen=True)
class SparseQuery:
    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(int(index) for index in self.indices))
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))
        if len(self.indices) != len(self.values):
            raise ValueError("SparseQuery.indices and SparseQuery.values must have the same length.")
