"""The sandwich: laplace transform in front, Prophet inside, exact inverse behind."""
from __future__ import annotations

import math

import pandas as pd

from skaters import laplace
from skaters.tails import _phi_inv

_EPS = 1e-12
_SQRT2 = math.sqrt(2.0)


def _phi(z: float) -> float:
    return 0.5 * math.erfc(-z / _SQRT2)


class _Predictive:
    """A y-space predictive assembled from Prophet's z-space Gaussian and
    laplace's h-step transport, with exact change-of-variables accounting."""

    def __init__(self, transport, z_mu: float, z_sigma: float):
        self._t = transport
        self._mu = z_mu
        self._sd = max(z_sigma, 1e-9)

    def quantile(self, p: float) -> float:
        z = self._mu + self._sd * _phi_inv(min(max(p, _EPS), 1.0 - _EPS))
        u = min(max(_phi(z), _EPS), 1.0 - _EPS)
        return self._t.quantile(u)

    def logpdf(self, y: float) -> float:
        u = min(max(self._t.cdf(y), _EPS), 1.0 - _EPS)
        z = _phi_inv(u)
        r = (z - self._mu) / self._sd
        log_fz = -0.5 * r * r - math.log(self._sd) - 0.5 * math.log(2 * math.pi)
        log_phi = -0.5 * z * z - 0.5 * math.log(2 * math.pi)
        return log_fz + self._t.logpdf(y) - log_phi

    def cdf(self, y: float) -> float:
        u = min(max(self._t.cdf(y), _EPS), 1.0 - _EPS)
        z = _phi_inv(u)
        return _phi((z - self._mu) / self._sd)


class SandwichedProphet:
    """Drop-in Prophet with laplace-sandwiched coordinates.

    Prophet keeps its API, calendar machinery, holidays, and extra
    regressors; it just operates in the z-coordinates of a laplace
    forecaster (the Rosenblatt transform of each observation under the
    predictive issued for it), and every forecast maps back through the
    exact inverse.

    ``k`` is the forecast horizon in observations; ``predict`` supports
    future frames up to k steps past the training data (rows beyond k
    reuse the k-step transport, a disclosed approximation).
    """

    def __init__(self, k: int = 1, **prophet_kwargs):
        from prophet import Prophet
        self.k = int(k)
        prophet_kwargs.setdefault("interval_width", 0.6827)
        self._m = Prophet(**prophet_kwargs)
        self._interval_z = 1.0          # 0.6827 central interval = +-1 sd
        self._fitted = False
        self._pending = None            # k-step transport past training end

    def add_regressor(self, name: str, **kw) -> "SandwichedProphet":
        self._m.add_regressor(name, **kw)
        return self

    def fit(self, df: pd.DataFrame, **fit_kwargs) -> "SandwichedProphet":
        f = laplace(self.k)
        state = None
        pend = None                     # dists issued for the arriving y
        zs = []
        for y in df["y"].astype(float).tolist():
            if pend is None:
                zs.append(0.0)
            else:
                u = min(max(pend[0].cdf(y), _EPS), 1.0 - _EPS)
                zs.append(_phi_inv(u))
            pend, state = f(y, state)
        self._pending = pend            # transport for steps 1..k ahead
        zdf = df.copy()
        zdf["y"] = zs
        self._m.fit(zdf, **fit_kwargs)
        self._last_ds = pd.to_datetime(df["ds"]).max()
        self._fitted = True
        return self

    def _transport(self, step: int):
        return self._pending[min(step, self.k) - 1]

    def predictive(self, step: int, z_mu: float, z_sigma: float) -> _Predictive:
        return _Predictive(self._transport(step), z_mu, z_sigma)

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "call fit first"
        ds = pd.to_datetime(future["ds"])
        assert (ds > self._last_ds).all(), (
            "sandwich v0 maps future rows only; pass a frame strictly "
            "after the training data")
        fc = self._m.predict(future)
        sds = ((fc["yhat_upper"] - fc["yhat_lower"]) / 2.0
               / self._interval_z).clip(lower=1e-9)
        out = fc.copy()
        for i in range(len(fc)):
            pred = self.predictive(i + 1, float(fc["yhat"].iloc[i]),
                                   float(sds.iloc[i]))
            out.loc[out.index[i], "yhat"] = pred.quantile(0.5)
            out.loc[out.index[i], "yhat_lower"] = pred.quantile(0.5 - 0.6827 / 2)
            out.loc[out.index[i], "yhat_upper"] = pred.quantile(0.5 + 0.6827 / 2)
        return out
