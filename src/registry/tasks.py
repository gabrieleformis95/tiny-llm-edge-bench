"""Load task definitions from configs/tasks.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from src.config import TaskSpec

_TASKS_YAML = Path(__file__).parents[2] / "configs" / "tasks.yaml"


@lru_cache(maxsize=1)
def load_tasks() -> list[TaskSpec]:
    """Return all TaskSpec entries from tasks.yaml."""
    with open(_TASKS_YAML) as f:
        data = yaml.safe_load(f)
    return [TaskSpec(**t) for t in data["tasks"]]


def get_task(task_id: str) -> TaskSpec:
    """Return a single TaskSpec by id. Raises KeyError if not found."""
    for t in load_tasks():
        if t.id == task_id:
            return t
    raise KeyError(f"Task not found in registry: {task_id!r}")
