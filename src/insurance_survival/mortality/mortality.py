"""
mortality.py — CauseSpecificMortality: DMP framework for coherent cause-specific
mortality forecasting.

Implements the Dirichlet-Multinomial-Poisson (DMP) model from:
    Nigri, Shang & Ungolo (2026). arXiv:2603.00973.

Two model variants:
- LC: Lee-Carter with shared kappa_t latent trend (recommended)
- AP: Additive age-period with RW2 priors (more flexible, less stable)

Both enforce coherence by construction:
    m_{a,t,c} = m_{a,t} * gamma_{a,t,c}
    => sum_c m_{a,t,c} = m_{a,t}  at every posterior draw.

Backend: NumPyro (JAX-based NUTS). No C++ compilation — pip-installable.
NumPyro is an optional dependency; a clear ImportError is raised if absent.

RW2 prior implementation
-------------------------
The second-order random walk prior for pi_t (AP model) is:

    Delta^2 pi_t = pi_t - 2*pi_{t-1} + pi_{t-2} ~ N(0, sigma^2)

In NumPyro this is implemented via lax.scan over:

    pi[t] = 2*pi[t-1] - pi[t-2] + eps[t],  eps[t] ~ N(0, sigma^2)

with pi[0] and pi[1] given flat (diffuse) priors. The RW2 prior
induces smooth linear trends rather than the rougher RW1.

Identifiability constraints (LC model)
---------------------------------------
- sum_t kappa_t = 0  (kappa is zero-centred)
- sum_a beta_a = 1   (beta sums to one — pin the scale of kappa)

Implemented by: computing raw_kappa and applying soft centering at
sampling time; normalising beta via softplus then dividing by sum.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .forecast import MortalityForecast


def _require_numpyro() -> tuple:
    """Lazy import of NumPyro and JAX with a clear error if not installed."""
    try:
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        import jax
        import jax.numpy as jnp
        from jax import lax
        return numpyro, dist, MCMC, NUTS, jax, jnp, lax
    except ImportError as e:
        raise ImportError(
            "NumPyro is required for CauseSpecificMortality fitting. "
            "Install it with: pip install insurance-survival[mortality]\n"
            f"Original error: {e}"
        ) from e


class CauseSpecificMortality:
    """Coherent cause-specific mortality model using the DMP framework.

    Implements the Dirichlet-Multinomial-Poisson hierarchy from
    Nigri, Shang & Ungolo (2026, arXiv:2603.00973).

    The key insight: rather than fitting cause-specific mortality models
    independently (which produces incoherent forecasts), this model enforces
    coherence through a probabilistic hierarchy. Total mortality is modelled
    via Lee-Carter or additive age-period; cause allocation is modelled
    separately via a Dirichlet-Multinomial, with the *same* period index
    kappa_t linking both levels.

    This means m_{a,t,c} = m_{a,t} * gamma_{a,t,c} at every posterior
    draw — no post-hoc reconciliation needed.

    Parameters
    ----------
    model_type : {"LC", "AP"}
        Model variant.
        ``"LC"`` (recommended): Lee-Carter structure. Total mortality follows
        log m_{a,t} = nu + delta_a + beta_a * kappa_t. Cause composition
        follows eta_{a,t,c} = phi0_c + zeta_{a,c} + theta_c * kappa_t.
        The shared kappa_t links aggregate and composition.
        ``"AP"``: Additive age-period. Both total and cause effects have
        independent RW2 period effects. More flexible, but slower and less
        stable on short data runs.
    n_chains : int
        Number of MCMC chains. Default 3 matches the paper.
    n_samples : int
        Posterior samples per chain. Default 1500 (half the paper's 3000 for
        tractability; increase for production runs).
    n_warmup : int
        NUTS warm-up (adaptation) steps per chain. Default 750.
    seed : int
        Random seed for JAX PRNG.
    target_accept_prob : float
        NUTS target acceptance probability. Default 0.8.

    Attributes
    ----------
    is_fitted_ : bool
        True after a successful call to ``fit()``.
    posterior_ : dict
        Raw NumPyro posterior samples keyed by parameter name.
    age_labels_ : list[str]
        Age group labels (set during fit).
    year_labels_ : list[int]
        Calendar year labels (set during fit).
    cause_names_ : list[str]
        Cause group names (set during fit).
    n_ages_ : int
        Number of age groups.
    n_years_ : int
        Number of calendar years.
    n_causes_ : int
        Number of cause groups.

    Examples
    --------
    >>> from insurance_survival.mortality import CauseSpecificMortality, HMDLoader
    >>> deaths, exposure, ages, years = HMDLoader.load_synthetic(
    ...     n_ages=10, n_years=20, n_causes=4, seed=0
    ... )
    >>> model = CauseSpecificMortality(
    ...     model_type="LC", n_chains=1, n_samples=200, n_warmup=100
    ... )
    >>> model.fit(deaths, exposure, age_labels=ages, year_labels=years)
    >>> forecast = model.forecast(horizon=10)
    >>> forecast.coherence_check()
    True
    """

    def __init__(
        self,
        model_type: str = "LC",
        n_chains: int = 3,
        n_samples: int = 1500,
        n_warmup: int = 750,
        seed: int = 42,
        target_accept_prob: float = 0.8,
    ) -> None:
        if model_type not in ("LC", "AP"):
            raise ValueError(f"model_type must be 'LC' or 'AP', got {model_type!r}")
        self.model_type = model_type
        self.n_chains = n_chains
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.seed = seed
        self.target_accept_prob = target_accept_prob

        self.is_fitted_: bool = False
        self.posterior_: dict = {}
        self.age_labels_: list[str] = []
        self.year_labels_: list[int] = []
        self.cause_names_: list[str] = []
        self.n_ages_: int = 0
        self.n_years_: int = 0
        self.n_causes_: int = 0

        # Store data for forecast extrapolation
        self._deaths: Optional[np.ndarray] = None
        self._exposure: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        deaths: np.ndarray,
        exposure: np.ndarray,
        age_labels: Optional[list[str]] = None,
        year_labels: Optional[list[int]] = None,
        cause_names: Optional[list[str]] = None,
    ) -> "CauseSpecificMortality":
        """Fit the DMP model to observed mortality data.

        Parameters
        ----------
        deaths : np.ndarray, shape (n_age, n_year, n_cause)
            Cause-specific death counts. Must be non-negative integers (floats
            are accepted and rounded internally).
        exposure : np.ndarray, shape (n_age, n_year)
            Central exposed-to-risk in person-years. Must be strictly positive.
        age_labels : list[str], optional
            Age group labels. Defaults to ["0", "5", "10", ...].
        year_labels : list[int], optional
            Calendar years. Defaults to [2000, 2001, ...].
        cause_names : list[str], optional
            Names for cause groups. Defaults to ["cause_0", "cause_1", ...].

        Returns
        -------
        self
            Fitted estimator.

        Raises
        ------
        ImportError
            If NumPyro is not installed.
        ValueError
            If input shapes are inconsistent or exposure contains zeros.
        """
        self._validate_inputs(deaths, exposure)

        n_age, n_year, n_cause = deaths.shape
        self.n_ages_ = n_age
        self.n_years_ = n_year
        self.n_causes_ = n_cause

        self.age_labels_ = age_labels or [str(i * 5) for i in range(n_age)]
        self.year_labels_ = year_labels or list(range(2000, 2000 + n_year))
        self.cause_names_ = cause_names or [f"cause_{i}" for i in range(n_cause)]

        if len(self.age_labels_) != n_age:
            raise ValueError(
                f"age_labels has {len(self.age_labels_)} entries but deaths has "
                f"{n_age} age groups"
            )
        if len(self.year_labels_) != n_year:
            raise ValueError(
                f"year_labels has {len(self.year_labels_)} entries but deaths has "
                f"{n_year} years"
            )
        if len(self.cause_names_) != n_cause:
            raise ValueError(
                f"cause_names has {len(self.cause_names_)} entries but deaths has "
                f"{n_cause} causes"
            )

        self._deaths = deaths.astype(float)
        self._exposure = exposure.astype(float)

        numpyro, dist, MCMC, NUTS, jax, jnp, lax = _require_numpyro()

        deaths_jax = jnp.array(deaths, dtype=jnp.float32)
        exposure_jax = jnp.array(exposure, dtype=jnp.float32)

        if self.model_type == "LC":
            model_fn = self._build_lc_model(
                n_age, n_year, n_cause, deaths_jax, exposure_jax,
                numpyro, dist, jnp, lax
            )
        else:
            model_fn = self._build_ap_model(
                n_age, n_year, n_cause, deaths_jax, exposure_jax,
                numpyro, dist, jnp, lax
            )

        nuts = NUTS(model_fn, target_accept_prob=self.target_accept_prob)
        mcmc = MCMC(
            nuts,
            num_warmup=self.n_warmup,
            num_samples=self.n_samples,
            num_chains=self.n_chains,
        )

        rng_key = jax.random.PRNGKey(self.seed)
        mcmc.run(rng_key)

        self.posterior_ = {
            k: np.array(v) for k, v in mcmc.get_samples().items()
        }
        self._mcmc = mcmc
        self.is_fitted_ = True
        return self

    def _validate_inputs(self, deaths: np.ndarray, exposure: np.ndarray) -> None:
        """Validate input arrays."""
        if deaths.ndim != 3:
            raise ValueError(
                f"deaths must be 3D (n_age, n_year, n_cause), got shape {deaths.shape}"
            )
        if exposure.ndim != 2:
            raise ValueError(
                f"exposure must be 2D (n_age, n_year), got shape {exposure.shape}"
            )
        if deaths.shape[:2] != exposure.shape:
            raise ValueError(
                f"deaths shape {deaths.shape[:2]} must match exposure shape {exposure.shape}"
            )
        if np.any(deaths < 0):
            raise ValueError("deaths must be non-negative")
        if np.any(exposure <= 0):
            raise ValueError("exposure must be strictly positive (no zero cells)")
        if np.any(np.isnan(deaths)) or np.any(np.isnan(exposure)):
            raise ValueError("deaths and exposure must not contain NaN values")

    # ------------------------------------------------------------------
    # Model definitions
    # ------------------------------------------------------------------

    def _build_lc_model(self, n_age, n_year, n_cause, deaths, exposure,
                        numpyro, dist, jnp, lax):
        """Build the LC-DM NumPyro model function.

        Model:
            Total: log m_{a,t} = nu + delta_a + beta_a * kappa_t
            Composition: eta_{a,t,c} = phi0_c + zeta_{a,c} + theta_c * kappa_t
            Cause probs: gamma = softmax(eta)
            Total deaths: Y_{a,t} ~ Poisson(E_{a,t} * m_{a,t})
            Cause deaths: Y_bar_{a,t} ~ DM(phi * gamma_{a,t}, Y_{a,t})

        Identifiability:
            sum_t kappa_t = 0 (enforce by subtracting mean)
            sum_a beta_a = 1  (enforce by softplus + normalise)
        """

        def model():
            # --- Total mortality (Lee-Carter) ---
            nu = numpyro.sample("nu", dist.Normal(0.0, 5.0))

            # Age effects delta_a: free with diffuse prior
            delta = numpyro.sample("delta", dist.Normal(jnp.zeros(n_age), 5.0))

            # beta_a: non-negative, sum to 1 (identifiability)
            raw_beta = numpyro.sample("raw_beta", dist.Normal(jnp.zeros(n_age), 1.0))
            beta = jnp.exp(raw_beta) / jnp.exp(raw_beta).sum()  # softmax -> simplex

            # kappa_t: RW1 with drift, centred
            sigma_kappa = numpyro.sample("sigma_kappa", dist.HalfNormal(0.5))
            raw_kappa = numpyro.sample(
                "raw_kappa",
                dist.GaussianRandomWalk(scale=sigma_kappa, num_steps=n_year)
            )
            # Enforce sum_t kappa_t = 0
            kappa = raw_kappa - raw_kappa.mean()

            # log mortality rate: (n_age, n_year)
            log_m = nu + delta[:, None] + beta[:, None] * kappa[None, :]
            mu_total = exposure * jnp.exp(log_m)

            # Poisson likelihood for total deaths
            total_deaths_obs = deaths.sum(axis=2)  # (n_age, n_year)
            numpyro.sample(
                "total_deaths",
                dist.Poisson(jnp.clip(mu_total, a_min=1e-8)),
                obs=total_deaths_obs,
            )

            # --- Cause composition (Dirichlet-Multinomial) ---
            # Intercepts per cause
            phi0 = numpyro.sample("phi0", dist.Normal(jnp.zeros(n_cause), 5.0))

            # Age effects per cause
            sigma_zeta = numpyro.sample("sigma_zeta", dist.HalfNormal(jnp.ones(n_cause)))
            zeta = numpyro.sample(
                "zeta",
                dist.Normal(jnp.zeros((n_age, n_cause)), sigma_zeta[None, :])
            )

            # Cause loadings on shared kappa
            theta = numpyro.sample("theta", dist.Normal(jnp.zeros(n_cause), 1.0))

            # Linear predictor: (n_age, n_year, n_cause)
            eta = (
                phi0[None, None, :]
                + zeta[:, None, :]
                + theta[None, :] * kappa[None, :, None]  # broadcast over age
            )

            # Cause probabilities via softmax
            eta_max = eta.max(axis=2, keepdims=True)
            eta_exp = jnp.exp(eta - eta_max)
            gamma = eta_exp / eta_exp.sum(axis=2, keepdims=True)  # (n_age, n_year, n_cause)

            # Overdispersion parameter phi (Dirichlet concentration)
            phi_dm = numpyro.sample("phi_dm", dist.LogNormal(2.0, 1.0))

            # Dirichlet-Multinomial likelihood for cause deaths
            # NumPyro's DirichletMultinomial: concentration = phi * gamma, total = n
            alpha = phi_dm * gamma  # (n_age, n_year, n_cause)

            numpyro.sample(
                "cause_deaths",
                dist.DirichletMultinomial(concentration=alpha, total_count=total_deaths_obs[..., None].squeeze(-1)),
                obs=deaths,
            )

        return model

    def _build_ap_model(self, n_age, n_year, n_cause, deaths, exposure,
                        numpyro, dist, jnp, lax):
        """Build the AP-DM NumPyro model function.

        Model:
            Total: log m_{a,t} = nu + delta_a + pi_t
            Composition: eta_{a,t,c} = phi0_c + zeta_{a,c} + lambda_{t,c}
            Both delta, pi and zeta, lambda have RW2 priors.

        RW2 prior: pi[t] = 2*pi[t-1] - pi[t-2] + eps[t], eps ~ N(0, sigma^2)
        Implemented via lax.scan for efficiency.
        """

        def rw2(init_vals, innovations):
            """Scan function for RW2: state = (pi_{t-1}, pi_{t-2})."""
            def step(carry, eps):
                p_prev, p_prev2 = carry
                p_new = 2.0 * p_prev - p_prev2 + eps
                return (p_new, p_prev), p_new

            (_, _), series = lax.scan(step, (init_vals[1], init_vals[0]), innovations)
            return jnp.concatenate([init_vals, series])

        def model():
            # --- Total mortality (Additive Age-Period, RW2) ---
            nu = numpyro.sample("nu", dist.Normal(0.0, 5.0))

            # Age effects with RW2 prior
            sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(0.5))
            delta_init = numpyro.sample("delta_init", dist.Normal(jnp.zeros(2), 5.0))
            delta_innov = numpyro.sample(
                "delta_innov",
                dist.Normal(jnp.zeros(n_age - 2), sigma_delta)
            )
            delta = rw2(delta_init, delta_innov)  # (n_age,)

            # Period effects with RW2 prior
            sigma_pi = numpyro.sample("sigma_pi", dist.HalfNormal(0.5))
            pi_init = numpyro.sample("pi_init", dist.Normal(jnp.zeros(2), 5.0))
            pi_innov = numpyro.sample(
                "pi_innov",
                dist.Normal(jnp.zeros(n_year - 2), sigma_pi)
            )
            pi = rw2(pi_init, pi_innov)  # (n_year,)

            # log mortality rate: (n_age, n_year)
            log_m = nu + delta[:, None] + pi[None, :]
            mu_total = exposure * jnp.exp(log_m)

            total_deaths_obs = deaths.sum(axis=2)
            numpyro.sample(
                "total_deaths",
                dist.Poisson(jnp.clip(mu_total, a_min=1e-8)),
                obs=total_deaths_obs,
            )

            # --- Cause composition (Dirichlet-Multinomial, RW2 per cause) ---
            phi0 = numpyro.sample("phi0", dist.Normal(jnp.zeros(n_cause), 5.0))

            # Age effects per cause (RW2)
            sigma_zeta = numpyro.sample("sigma_zeta", dist.HalfNormal(jnp.ones(n_cause)))
            zeta_init = numpyro.sample(
                "zeta_init", dist.Normal(jnp.zeros((2, n_cause)), 5.0)
            )
            zeta_innov = numpyro.sample(
                "zeta_innov",
                dist.Normal(jnp.zeros((n_age - 2, n_cause)), sigma_zeta[None, :])
            )

            # Build RW2 for each cause via vmap over cause axis
            def rw2_cause(init_c, innov_c):
                return rw2(init_c, innov_c)

            zeta = jnp.stack([
                rw2(zeta_init[:, c], zeta_innov[:, c])
                for c in range(n_cause)
            ], axis=1)  # (n_age, n_cause)

            # Period effects per cause (RW2)
            sigma_lambda = numpyro.sample(
                "sigma_lambda", dist.HalfNormal(jnp.ones(n_cause))
            )
            lambda_init = numpyro.sample(
                "lambda_init", dist.Normal(jnp.zeros((2, n_cause)), 5.0)
            )
            lambda_innov = numpyro.sample(
                "lambda_innov",
                dist.Normal(jnp.zeros((n_year - 2, n_cause)), sigma_lambda[None, :])
            )
            lambda_t = jnp.stack([
                rw2(lambda_init[:, c], lambda_innov[:, c])
                for c in range(n_cause)
            ], axis=1)  # (n_year, n_cause)

            # Linear predictor: (n_age, n_year, n_cause)
            eta = (
                phi0[None, None, :]
                + zeta[:, None, :]
                + lambda_t[None, :, :]
            )

            eta_max = eta.max(axis=2, keepdims=True)
            eta_exp = jnp.exp(eta - eta_max)
            gamma = eta_exp / eta_exp.sum(axis=2, keepdims=True)

            phi_dm = numpyro.sample("phi_dm", dist.LogNormal(2.0, 1.0))
            alpha = phi_dm * gamma

            numpyro.sample(
                "cause_deaths",
                dist.DirichletMultinomial(
                    concentration=alpha,
                    total_count=total_deaths_obs[..., None].squeeze(-1),
                ),
                obs=deaths,
            )

        return model

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self,
        horizon: int,
        n_draws: Optional[int] = None,
    ) -> MortalityForecast:
        """Extrapolate fitted model forward by ``horizon`` years.

        Extends period effects using their fitted dynamics:
        - LC model: kappa_t extrapolated as RW1 with posterior drift
        - AP model: pi_t extrapolated as RW2 with posterior sigma_pi

        Coherence is guaranteed by computing:
            m_{a,t,c} = m_{a,t} * gamma_{a,t,c}

        at each draw, ensuring sum_c m_{a,t,c} = m_{a,t}.

        Parameters
        ----------
        horizon : int
            Number of future years to project.
        n_draws : int, optional
            Number of posterior draws to use. Defaults to all available draws
            (n_chains * n_samples). Set lower for fast exploratory analysis.

        Returns
        -------
        MortalityForecast
            Container with total_rates and cause_rates arrays, both
            shape (n_age, horizon, *) with the draw axis last.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Model must be fitted before forecasting. Call fit() first."
            )
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")

        total_draws = list(self.posterior_.values())[0].shape[0]
        if n_draws is None:
            n_draws = total_draws
        elif n_draws > total_draws:
            warnings.warn(
                f"n_draws={n_draws} exceeds available draws ({total_draws}). "
                f"Using all {total_draws} draws.",
                stacklevel=2,
            )
            n_draws = total_draws

        rng = np.random.default_rng(self.seed + 999)

        if self.model_type == "LC":
            total_rates, cause_rates = self._forecast_lc(horizon, n_draws, rng)
        else:
            total_rates, cause_rates = self._forecast_ap(horizon, n_draws, rng)

        future_years = [
            self.year_labels_[-1] + t + 1 for t in range(horizon)
        ]

        return MortalityForecast(
            total_rates=total_rates,
            cause_rates=cause_rates,
            ages=self.age_labels_,
            periods=future_years,
            cause_names=self.cause_names_,
        )

    def _forecast_lc(
        self, horizon: int, n_draws: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extrapolate LC-DM model forward.

        kappa_t extended as RW1: kappa_{T+h} = kappa_{T+h-1} + drift + eps,
        where drift is estimated as the mean first difference of kappa over
        the observed period, and eps ~ N(0, sigma_kappa^2).
        """
        posterior = self.posterior_
        draw_idx = rng.choice(list(self.posterior_.values())[0].shape[0], n_draws, replace=False)

        # Extract posterior draws
        nu = posterior["nu"][draw_idx]                    # (n_draws,)
        delta = posterior["delta"][draw_idx]              # (n_draws, n_age)
        raw_beta = posterior["raw_beta"][draw_idx]        # (n_draws, n_age)
        raw_kappa = posterior["raw_kappa"][draw_idx]      # (n_draws, n_year)
        sigma_kappa = posterior["sigma_kappa"][draw_idx]  # (n_draws,)
        phi0 = posterior["phi0"][draw_idx]                # (n_draws, n_cause)
        zeta = posterior["zeta"][draw_idx]                # (n_draws, n_age, n_cause)
        theta = posterior["theta"][draw_idx]              # (n_draws, n_cause)
        phi_dm = posterior["phi_dm"][draw_idx]            # (n_draws,) — not used in forecast mean

        # Re-apply identifiability constraints
        beta = np.exp(raw_beta) / np.exp(raw_beta).sum(axis=1, keepdims=True)  # (n_draws, n_age)
        kappa = raw_kappa - raw_kappa.mean(axis=1, keepdims=True)               # (n_draws, n_year)

        # Estimate RW1 drift from observed kappa
        drift = np.diff(kappa, axis=1).mean(axis=1)  # (n_draws,)

        # Extrapolate kappa forward
        kappa_future = np.zeros((n_draws, horizon))
        kappa_last = kappa[:, -1]  # (n_draws,)
        for h in range(horizon):
            eps = rng.normal(0, 1, n_draws) * sigma_kappa
            kappa_future[:, h] = (
                (kappa_last if h == 0 else kappa_future[:, h - 1]) + drift + eps
            )

        # Note: we do NOT re-centre kappa_future — the centering constraint
        # applies to the observed kappa only; future kappa follows the RW freely.

        # Compute forecast total mortality rates
        # log m_{a,t} = nu + delta_a + beta_a * kappa_t
        # Shape: (n_draws, n_age, horizon)
        log_m_future = (
            nu[:, None, None]
            + delta[:, :, None]
            + beta[:, :, None] * kappa_future[:, None, :]
        )
        total_rates_daw = np.exp(log_m_future)  # (n_draws, n_age, horizon)

        # Compute cause composition
        # eta_{a,t,c} = phi0_c + zeta_{a,c} + theta_c * kappa_t
        # Shape: (n_draws, n_age, horizon, n_cause)
        eta_future = (
            phi0[:, None, None, :]
            + zeta[:, :, None, :]
            + theta[:, None, :] * kappa_future[:, None, :, None]  # broadcast
        )

        # Softmax over causes
        eta_max = eta_future.max(axis=3, keepdims=True)
        eta_exp = np.exp(eta_future - eta_max)
        gamma_future = eta_exp / eta_exp.sum(axis=3, keepdims=True)  # (n_draws, n_age, horizon, n_cause)

        # Cause rates = total rate * gamma
        cause_rates_daw = (
            total_rates_daw[:, :, :, np.newaxis] * gamma_future
        )  # (n_draws, n_age, horizon, n_cause)

        # Reorder to (n_age, horizon, n_draw) and (n_age, horizon, n_cause, n_draw)
        total_rates = np.transpose(total_rates_daw, (1, 2, 0))
        cause_rates = np.transpose(cause_rates_daw, (1, 2, 3, 0))

        return total_rates, cause_rates

    def _forecast_ap(
        self, horizon: int, n_draws: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extrapolate AP-DM model forward.

        pi_t extended as RW2: pi_{T+h} = 2*pi_{T+h-1} - pi_{T+h-2} + eps,
        eps ~ N(0, sigma_pi^2).

        For cause composition, lambda_{t,c} extended independently per cause.
        """
        posterior = self.posterior_
        draw_idx = rng.choice(list(self.posterior_.values())[0].shape[0], n_draws, replace=False)

        nu = posterior["nu"][draw_idx]                       # (n_draws,)
        delta = posterior["delta"][draw_idx]                 # (n_draws, n_age)
        sigma_pi = posterior["sigma_pi"][draw_idx]           # (n_draws,)
        phi0 = posterior["phi0"][draw_idx]                   # (n_draws, n_cause)
        zeta = posterior["zeta"][draw_idx]                   # (n_draws, n_age, n_cause)
        sigma_lambda = posterior["sigma_lambda"][draw_idx]   # (n_draws, n_cause)

        # Reconstruct pi from stored components
        pi_init = posterior["pi_init"][draw_idx]             # (n_draws, 2)
        pi_innov = posterior["pi_innov"][draw_idx]           # (n_draws, n_year-2)
        pi_obs = self._reconstruct_rw2(pi_init, pi_innov)    # (n_draws, n_year)

        # Extrapolate pi forward via RW2
        pi_future = np.zeros((n_draws, horizon))
        for h in range(horizon):
            p_prev = pi_obs[:, -1] if h == 0 else pi_future[:, h - 1]
            p_prev2 = pi_obs[:, -2] if h == 0 else (
                pi_obs[:, -1] if h == 1 else pi_future[:, h - 2]
            )
            eps = rng.normal(0, 1, n_draws) * sigma_pi
            pi_future[:, h] = 2.0 * p_prev - p_prev2 + eps

        # Reconstruct lambda_t per cause
        lambda_init = posterior["lambda_init"][draw_idx]     # (n_draws, 2, n_cause)
        lambda_innov = posterior["lambda_innov"][draw_idx]   # (n_draws, n_year-2, n_cause)

        lambda_future = np.zeros((n_draws, horizon, self.n_causes_))
        for c in range(self.n_causes_):
            lam_obs_c = self._reconstruct_rw2(
                lambda_init[:, :, c], lambda_innov[:, :, c]
            )  # (n_draws, n_year)
            for h in range(horizon):
                p_prev = lam_obs_c[:, -1] if h == 0 else lambda_future[:, h - 1, c]
                p_prev2 = lam_obs_c[:, -2] if h == 0 else (
                    lam_obs_c[:, -1] if h == 1 else lambda_future[:, h - 2, c]
                )
                eps = rng.normal(0, 1, n_draws) * sigma_lambda[:, c]
                lambda_future[:, h, c] = 2.0 * p_prev - p_prev2 + eps

        # Total mortality forecast
        log_m_future = (
            nu[:, None, None]
            + delta[:, :, None]
            + pi_future[:, None, :]
        )  # (n_draws, n_age, horizon)
        total_rates_daw = np.exp(log_m_future)

        # Cause composition forecast
        eta_future = (
            phi0[:, None, None, :]
            + zeta[:, :, None, :]
            + lambda_future[:, None, :, :]
        )  # (n_draws, n_age, horizon, n_cause)

        eta_max = eta_future.max(axis=3, keepdims=True)
        eta_exp = np.exp(eta_future - eta_max)
        gamma_future = eta_exp / eta_exp.sum(axis=3, keepdims=True)

        cause_rates_daw = total_rates_daw[:, :, :, np.newaxis] * gamma_future

        total_rates = np.transpose(total_rates_daw, (1, 2, 0))
        cause_rates = np.transpose(cause_rates_daw, (1, 2, 3, 0))

        return total_rates, cause_rates

    @staticmethod
    def _reconstruct_rw2(init: np.ndarray, innov: np.ndarray) -> np.ndarray:
        """Reconstruct a RW2 series from initial values and innovations.

        Parameters
        ----------
        init : np.ndarray, shape (n_draws, 2)
            Initial two values of the series.
        innov : np.ndarray, shape (n_draws, n_steps)
            Innovations (second differences).

        Returns
        -------
        np.ndarray, shape (n_draws, 2 + n_steps)
        """
        n_draws, n_steps = innov.shape
        series = np.zeros((n_draws, 2 + n_steps))
        series[:, :2] = init
        for t in range(n_steps):
            series[:, t + 2] = (
                2.0 * series[:, t + 1] - series[:, t] + innov[:, t]
            )
        return series

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict:
        """Compute MCMC convergence diagnostics.

        Returns R-hat and effective sample size (ESS) for all sampled
        parameters. Uses ArviZ if available; falls back to a simple
        split-chain R-hat approximation.

        Returns
        -------
        dict with keys:
            - "rhat": dict mapping parameter name to max R-hat value
            - "ess": dict mapping parameter name to min ESS value
            - "n_divergences": int (0 if ArviZ not available)

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before diagnostics.")

        try:
            import arviz as az
            idata = az.from_numpyro(self._mcmc)
            rhat = az.rhat(idata)
            ess = az.ess(idata)

            rhat_max = {}
            ess_min = {}
            for var in rhat.data_vars:
                rhat_max[var] = float(rhat[var].values.max())
                ess_min[var] = float(ess[var].values.min())

            n_div = int(
                idata.sample_stats["diverging"].values.sum()
                if "diverging" in idata.sample_stats
                else 0
            )

            return {
                "rhat": rhat_max,
                "ess": ess_min,
                "n_divergences": n_div,
                "backend": "arviz",
            }

        except ImportError:
            # Fallback: simple split-chain R-hat per parameter
            # Reshape posterior to (n_chains, n_samples, ...)
            rhat_approx = {}
            n_total = list(self.posterior_.values())[0].shape[0]
            n_chains = self.n_chains
            n_per_chain = n_total // n_chains

            for name, samples in self.posterior_.items():
                # Flatten all but the draw axis, compute across chains
                flat = samples[:n_chains * n_per_chain].reshape(
                    n_chains, n_per_chain, -1
                )
                # Between-chain variance
                chain_means = flat.mean(axis=1)  # (n_chains, n_flat)
                grand_mean = chain_means.mean(axis=0)
                B = n_per_chain * ((chain_means - grand_mean) ** 2).mean(axis=0)
                # Within-chain variance
                W = flat.var(axis=1, ddof=1).mean(axis=0)
                # Var+ estimate
                var_plus = ((n_per_chain - 1) * W + B) / n_per_chain
                rhat_vals = np.sqrt(var_plus / np.where(W > 0, W, np.nan))
                rhat_approx[name] = float(np.nanmax(rhat_vals))

            return {
                "rhat": rhat_approx,
                "ess": {},  # not computed without ArviZ
                "n_divergences": 0,
                "backend": "simple",
            }

    def cause_fractions(self, age_idx: int, year_idx: int) -> np.ndarray:
        """Posterior mean cause fractions for a specific age and year cell.

        Parameters
        ----------
        age_idx : int
            Index into age_labels_ (0-based).
        year_idx : int
            Index into year_labels_ (0-based).

        Returns
        -------
        np.ndarray, shape (n_causes,)
            Posterior mean gamma_{a,t,c} for the requested cell.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        IndexError
            If age_idx or year_idx are out of bounds.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before computing cause fractions.")

        if not (0 <= age_idx < self.n_ages_):
            raise IndexError(
                f"age_idx {age_idx} out of range [0, {self.n_ages_})"
            )
        if not (0 <= year_idx < self.n_years_):
            raise IndexError(
                f"year_idx {year_idx} out of range [0, {self.n_years_})"
            )

        if self.model_type == "LC":
            raw_kappa = self.posterior_["raw_kappa"]          # (n_draws, n_year)
            kappa = raw_kappa - raw_kappa.mean(axis=1, keepdims=True)
            kappa_t = kappa[:, year_idx]                      # (n_draws,)

            phi0 = self.posterior_["phi0"]                    # (n_draws, n_cause)
            zeta = self.posterior_["zeta"]                    # (n_draws, n_age, n_cause)
            theta = self.posterior_["theta"]                  # (n_draws, n_cause)

            eta = (
                phi0
                + zeta[:, age_idx, :]
                + theta * kappa_t[:, np.newaxis]
            )  # (n_draws, n_cause)
        else:
            phi0 = self.posterior_["phi0"]
            zeta = self.posterior_["zeta"]
            pi_init = self.posterior_["pi_init"]
            pi_innov = self.posterior_["pi_innov"]
            lambda_init = self.posterior_["lambda_init"]
            lambda_innov = self.posterior_["lambda_innov"]

            lambda_series = np.stack([
                self._reconstruct_rw2(lambda_init[:, :, c], lambda_innov[:, :, c])
                for c in range(self.n_causes_)
            ], axis=2)  # (n_draws, n_year, n_cause)

            eta = (
                phi0
                + zeta[:, age_idx, :]
                + lambda_series[:, year_idx, :]
            )

        # Softmax
        eta_max = eta.max(axis=1, keepdims=True)
        eta_exp = np.exp(eta - eta_max)
        gamma = eta_exp / eta_exp.sum(axis=1, keepdims=True)  # (n_draws, n_cause)

        return gamma.mean(axis=0)
