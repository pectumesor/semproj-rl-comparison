class EarlyStopping:
    """
    Tracks an exponential moving average (EMA) of a "higher is better" metric
    (e.g. mean eval return) and signals convergence once it has not improved
    by more than `min_delta` for `patience` consecutive checks.

    Convergence is only ever signalled once the best EMA seen so far has
    reached `min_converged_value`. This prevents the run from stopping while
    the policy is still bad and merely stuck on a temporary plateau early in
    training.
    """

    def __init__(self, patience: int, min_delta: float = 0.0, ema_alpha: float = 0.1,
                 min_converged_value: float = -float("inf")):
        self.patience = patience
        self.min_delta = min_delta
        self.ema_alpha = ema_alpha
        self.min_converged_value = min_converged_value

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

        plateaued = self.num_bad_checks >= self.patience
        return plateaued and self.best >= self.min_converged_value


class ConvergenceMonitor:
    """
    Early-stopping driver that combines two signals:

    * the eval return  -- the primary "is the policy actually good?" signal,
      gated by `min_converged_value` so we never stop while it is still bad;
    * (optionally) the training return -- a secondary "is the policy still
      visibly learning?" signal.

    A run is only stopped once BOTH EMAs have plateaued for `patience`
    consecutive checks. The eval return alone is noisy (few eval episodes,
    stochastic starts) and can look flat for a while even when the stochastic
    training return -- and entropy / action std -- are still moving. Requiring
    the training return to have settled as well avoids stopping too early.

    Set `track_train_return: false` in the config to fall back to eval-only.
    """

    def __init__(self, es_cfg):
        self.enabled = bool(es_cfg.enabled)
        self.patience = es_cfg.patience

        self.eval_stopper = EarlyStopping(
            patience=es_cfg.patience,
            min_delta=es_cfg.min_delta,
            ema_alpha=es_cfg.ema_alpha,
            min_converged_value=es_cfg.min_converged_value,
        )

        self.track_train_return = bool(es_cfg.get("track_train_return", False))
        self.train_stopper = None
        if self.track_train_return:
            self.train_stopper = EarlyStopping(
                patience=es_cfg.patience,
                # train return lives on a per-step scale, so it needs its own
                # (smaller) plateau threshold.
                min_delta=es_cfg.get("train_min_delta", es_cfg.min_delta),
                ema_alpha=es_cfg.ema_alpha,
            )

    def step(self, eval_return: float, train_return: float = None) -> bool:
        """Returns True once training is considered converged."""
        if not self.enabled:
            return False

        eval_converged = self.eval_stopper.step(eval_return)

        train_plateaued = True
        if self.train_stopper is not None:
            train_plateaued = self.train_stopper.step(train_return)

        return eval_converged and train_plateaued

    @property
    def improved(self) -> bool:
        """Whether the eval-return EMA just hit a new best (for checkpointing)."""
        return self.eval_stopper.improved

    @property
    def best(self) -> float:
        return self.eval_stopper.best

    @property
    def train_best(self):
        return self.train_stopper.best if self.train_stopper is not None else None
