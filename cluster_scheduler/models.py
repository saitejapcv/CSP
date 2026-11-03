from dataclasses import dataclass

@dataclass(frozen=True)
class Node:
    id: str
    zone:str
    cpu: int
    ram: int
    gpu: bool = False

    def __post_init__(self):
        if self.cpu <= 0:
            raise ValueError("cpu must be greater than 0")
        if self.ram <= 0:
            raise ValueError("ram must be greater than 0")
        if not self.id.strip():
            raise ValueError("id must not be empty")

@dataclass(frozen=True)
class Job:
    id: str
    service: str
    cpu: int
    ram: int
    needs_gpu: bool = False
    allowed_zones: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.cpu <= 0:
            raise ValueError("cpu must be greater than 0")
        if self.ram <= 0:
            raise ValueError("ram must be greater than 0")
        if not self.id.strip():
            raise ValueError("id must not be empty")

@dataclass(frozen=True)
class Cluster:
    nodes: tuple[Node, ...]

