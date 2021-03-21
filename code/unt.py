from utils.visualization import Visualizations


class SigmaAnnealer:
    """Annealer for sigma.
    Args:
        init (float): Initial value.
        final (float): Final value.
        constant (float): Constant value for pre-train
        steps (int): Number of annealing steps.
        pretrain (int): Number of pre-training steps.
    """
    def __init__(self, init: float, final: float, constant: float, steps: int,
                 pretrain: int, current_step: int = 0, **kwargs) -> None:

        self.init = init
        self.final = final
        self.constant = constant
        self.steps = steps
        self.pretrain = pretrain

        if steps < pretrain:
            self.steps += pretrain

        # Current time step
        self.t = current_step

    def __iter__(self):
        return self

    def __next__(self) -> float:
        self.t += 1

        if self.t <= self.pretrain:
            value = self.constant
        else:
            value = max((self.final + (self.init - self.final) *
                         (1 - self.t / (self.steps - self.pretrain))),
                        self.final)

        # Return sigma
        return value


sigma_scheduler_params = {
    "init": 2.0,
    "final": 0.7,
    "constant": 1.0,
    "steps": 250000,
    "pretrain": 50000,
    "current_step": 50000
}
