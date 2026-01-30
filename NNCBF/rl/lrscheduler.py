from dataclasses import dataclass
import numpy as np


@dataclass
class LearningRateScheduler:
    """
    Stores learning-rate scheduling state.
    """
    alpha: float
    patience_threshold: int
    lr_decay_factor: float

    best_stage_cost: float = np.inf
    best_params: dict | None = None
    current_patience: int = 0
    

def update_learning_rate(current_stage_cost, params, scheduler):
    """
    Update the learning rate based on the current stage cost metric.
    """

    if current_stage_cost < scheduler.best_stage_cost:
        scheduler.best_params = params.copy()
        scheduler.best_stage_cost = current_stage_cost
        scheduler.current_patience = 0
    else:
        scheduler.current_patience += 1

    if scheduler.current_patience >= scheduler.patience_threshold:
        old_alpha = scheduler.alpha
        scheduler.alpha *= scheduler.lr_decay_factor  # decay
        print(
            f"Learning rate decreased from {old_alpha} to {scheduler.alpha} "
            f"due to no stage cost improvement."
        )
        scheduler.current_patience = 0  # reset
        params = scheduler.best_params  # revert to best params

    return params