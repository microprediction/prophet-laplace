import math
import random

import pandas as pd
import pytest

from prophet_skaters import SandwichedProphet


def _df(n=400, seed=3):
    rng = random.Random(seed)
    ds = pd.date_range("2020-01-01", periods=n, freq="D")
    lvl, ys = 0.0, []
    for t in range(n):
        vol = 2.0 if (t // 60) % 2 else 0.5
        lvl += rng.gauss(0, vol)
        ys.append(lvl + 3.0 * math.sin(2 * math.pi * t / 7))
    return pd.DataFrame({"ds": ds, "y": ys})


def test_fit_predict_roundtrip():
    df = _df()
    m = SandwichedProphet(k=5).fit(df)
    future = pd.DataFrame(
        {"ds": pd.date_range(df["ds"].max() + pd.Timedelta(days=1),
                             periods=5, freq="D")})
    fc = m.predict(future)
    assert len(fc) == 5
    assert (fc["yhat_lower"] <= fc["yhat"]).all()
    assert (fc["yhat"] <= fc["yhat_upper"]).all()
    assert fc["yhat"].abs().max() < 1e6


def test_predictive_density_is_wellformed():
    df = _df()
    m = SandwichedProphet(k=1).fit(df)
    pred = m.predictive(1, 0.0, 1.0)
    qs = [pred.quantile(p) for p in (0.01, 0.25, 0.5, 0.75, 0.99)]
    assert all(b >= a for a, b in zip(qs, qs[1:]))
    y0 = qs[2]
    assert math.isfinite(pred.logpdf(y0))
    assert 0.0 <= pred.cdf(y0) <= 1.0
    assert abs(pred.cdf(qs[1]) - 0.25) < 0.05


def test_past_frames_rejected():
    df = _df()
    m = SandwichedProphet(k=1).fit(df)
    with pytest.raises(AssertionError):
        m.predict(df[["ds"]].tail(3))
