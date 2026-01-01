from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class BaseConfig:
    def __getitem__(self, key: str):
        return getattr(self, key)

    @classmethod
    def from_dict(cls, data: dict):
        data = dict(data)
        if "features" in data and isinstance(data["features"], list):
            data["features"] = tuple(data["features"])
        if "seed" in data and isinstance(data["seed"], list):
            data["seed"] = tuple(data["seed"])
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

