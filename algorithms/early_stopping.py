class EarlyStopping:
    """
    Tracks an exponential moving average (EMA) of a "higher is better" metric
    (e.g. mean eval return) and signals convergence once it has not improved
    by more than `min_delta` for `patience` consecutive checks.
    """

    def __init__(self, patience: int, min_delta: float = 0.0, ema_alpha: float = 0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.ema_alpha = ema_alpha

        self.ema = None
        self.best = -float("inf")
        self.num_bad_checks = 0
        self.improved = False

    def step(self, value: float) -> bool:
        """
        Update with the latest raw metric value (pass None to skip a check,
        e.g. on iterations where no evaluation ran).
        Returns True once convergence has been detected.
        """
        if value is None:
            self.improved = False
            return False

        self.ema = value if self.ema is None else (
            self.ema_alpha * value + (1 - self.ema_alpha) * self.ema
        )

        self.improved = self.ema > self.best + self.min_delta
        if self.improved:
            self.best = self.ema
            self.num_bad_checks = 0
        else:
            self.num_bad_checks += 1

        return self.num_bad_checks >= self.patience
