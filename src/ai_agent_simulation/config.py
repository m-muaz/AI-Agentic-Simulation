from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class AgentScenario:
    agent_id: str
    initial_wealth: float
    initial_health: float
    name: Optional[str] = None
    household_history: str = ""
    starting_parameters: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_agent_configs(path: Optional[str] = None) -> List[AgentScenario]:
    """
    Load agent definitions from a YAML file.
    """
    config_path = Path(path or "configs/agents.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Agent config file not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}

    configs: List[AgentScenario] = []
    for entry in config_data.get("agents", []):
        agent_id = entry.get("id") or entry.get("agent_id")
        if not agent_id:
            raise ValueError("Each agent entry must include an 'id'.")
        configs.append(
            AgentScenario(
                agent_id=agent_id,
                initial_wealth=_coerce_float(entry.get("initial_wealth", 0.0), 0.0),
                initial_health=_coerce_float(entry.get("initial_health", 1.0), 1.0),
                name=entry.get("name"),
                household_history=entry.get("household_history", ""),
                starting_parameters=entry.get("starting_parameters", {}),
                memory=entry.get("memory", {}),
            )
        )
    return configs
