# RL algorithms
from .sac_agent import SACAgent

RL_ALGOS = {
    "sac": SACAgent,
}

def get_agent_by_name(algo):
    """
    Returns RL or IL agent.
    """
    if algo in RL_ALGOS:
        return RL_ALGOS[algo]
    else:
        raise ValueError("--algo %s is not supported" % algo)
