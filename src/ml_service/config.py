import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    model_path: str
    test_size: float
    random_state: int
    n_estimators: int
    max_depth: int | None


def load_config(path: str = "configs/config.yaml") -> Config:
    config_path = Path(path)
    with config_path.open() as f:
        data = yaml.safe_load(f)
    return Config(
        model_path=data["model_path"],
        test_size=data["test_size"],
        random_state=data["random_state"],
        n_estimators=data["n_estimators"],
        max_depth=data.get("max_depth"),
    )
