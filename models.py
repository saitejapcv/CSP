from dataclasses import dataclass

@dataclass(frozen=True)
class Node:
    id: str
    zone:str
    cpu: int
    ram: int
    gpu: bool = False

@dataclass(frozen=True)
class Job:
    id: str
    service: str
    cpu: int
    ram: int
    need_gpu: bool = False
    allowed_zones: tuple[str, ...] | None = None


