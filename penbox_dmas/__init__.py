"""PenBox-DMAS: Distributed Multi-Agent Security Appliance — Simulation Package."""
from .equations import (
    SecurityState, vulnerability_set, control_set,
    attack_surface_metric, vulnerability_discovery,
    vulnerability_validation, vulnerability_state_evolution,
    risk_score_evolution, equilibrium_vulnerability,
    swarm_convergence_time, llm_routing, bellman_q_update,
    rl_exploit_probability, OS_RISK_PARAMS, PARAMS,
)
from .simulation import run_full_simulation

__version__ = "1.0.0"
__all__ = [
    "SecurityState", "vulnerability_set", "control_set",
    "attack_surface_metric", "vulnerability_discovery",
    "vulnerability_validation", "vulnerability_state_evolution",
    "risk_score_evolution", "equilibrium_vulnerability",
    "swarm_convergence_time", "llm_routing", "bellman_q_update",
    "rl_exploit_probability", "OS_RISK_PARAMS", "PARAMS",
    "run_full_simulation",
]
