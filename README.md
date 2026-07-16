# prophet-laplace

*"He's not the messiah. He's a very naughty boy."* ([why](https://medium.com/geekculture/is-facebooks-prophet-the-time-series-messiah-or-just-a-very-naughty-boy-8b71b136bc8c)) *This package makes him useful anyway.*

![The prophet skating in laplace's coordinates, out on T and back on T-inverse](https://raw.githubusercontent.com/microprediction/prophet-laplace/main/docs/assets/prophet-skating.jpg)

Prophet in a laplace sandwich: the same Prophet API, with calibrated predictive densities. [Try the live demo.](https://prophet.microprediction.org/demo.html)

`SandwichedProphet` fits [Prophet](https://github.com/facebook/prophet) in the z-coordinates of a [skaters](https://github.com/microprediction/skaters) `laplace` forecaster and maps every forecast back through the exact inverse. Prophet keeps its calendar decomposition, holidays, and extra regressors. The sandwich adds the volatility clock, repeated-value handling, and tails whose stated probabilities come true.

## Install

```bash
pip install prophet-laplace
```

This pulls in `prophet`, `skaters`, and `pandas`. Python 3.9+.

## Quickstart

The API mirrors Prophet. Fit on a frame with `ds` and `y`, then predict on a future frame.

```python
import pandas as pd
from prophet_laplace import SandwichedProphet

m = SandwichedProphet(k=30)          # k = forecast horizon, in observations
m.fit(df)                            # df has columns ds, y (+ any regressors)

future = pd.DataFrame({"ds": pd.date_range(df["ds"].max() + pd.Timedelta(days=1),
                                           periods=30, freq="D")})
fc = m.predict(future)               # yhat / yhat_lower / yhat_upper, mapped back exactly
```

`predict` returns Prophet's usual frame. The point and interval columns are the median and the central 68.27% band of the sandwiched predictive, so the interval reflects the volatility clock and tails rather than Prophet's Gaussian assumption.

Extra regressors and Prophet keyword arguments pass straight through:

```python
m = SandwichedProphet(k=14, seasonality_mode="multiplicative", weekly_seasonality=True)
m.add_regressor("temperature")
m.fit(df)                            # df now also has a temperature column
```

## Calibrated densities

The reason to sandwich is the full predictive density, not just the interval. `predictive` returns a y-space distribution with `logpdf`, `cdf`, and `quantile`, built from Prophet's z-space forecast (mean `z_mu`, standard deviation `z_sigma`) and laplace's transport, with exact change-of-variables accounting.

```python
pred = m.predictive(step=1, z_mu=0.0, z_sigma=1.0)

pred.quantile(0.5)     # median in y-space
pred.quantile(0.99)    # upper 1% level, tail-aware
pred.logpdf(y_obs)     # score a realised observation
pred.cdf(y_obs)        # PIT value; uniform under calibration
```

Use `logpdf` for likelihood scoring, `cdf` for PIT and calibration checks, and `quantile` for value-at-risk style levels.

## How it works

`laplace` defines a causal bijection on paths: the Rosenblatt transform

```
z_t = Phi^{-1}( F_t(y_t) )
```

where `F_t` is the predictive cdf `laplace` issued for `y_t`. Under calibration the z stream is close to i.i.d. standard normal, so Prophet is handed a stationarised, unit-scale series and its calendar model works on structure that survives the transform.

Every density maps back with no approximation in the accounting:

```
log f_Y(y) = log f_Z(z) + log f_t(y) - log phi(z)
```

`log f_t(y)` is laplace's own log-density at `y`, `f_Z` is Prophet's Gaussian in z, and `phi` is the standard normal density. Running an opponent between the transform and its inverse is the *sandwich*; whatever the inner model finds is, by construction, structure `laplace` alone did not capture.

## Why: the measured gap

Measured on 921 non-price FRED series under a pre-registered protocol (statements filed before results, frozen universe, harness and results committed in the skaters repository):

|                    | median one-step LL vs laplace | family-weighted (120 families) |
|--------------------|-------------------------------|--------------------------------|
| Prophet raw        | −0.755 nats                   | −4.60 nats                     |
| Prophet sandwiched | **−0.020 nats**               | **−0.025 nats**                |

The sandwich closes 97% of Prophet's density gap without retraining anything. The residual 0.02 nats is the epsilon: conditional structure that neither model captures. On FRED-30 with rolling refits, Prophet's calendar machinery adds a small positive epsilon over laplace alone, so on calendar-driven series the sandwich can exceed it.

The same construction lifts other forecasters and detectors. See the [sandwich page](https://skaters.microprediction.org/sandwich.html) for the wider table.

## API

### `SandwichedProphet(k=1, **prophet_kwargs)`

`k` is the forecast horizon in observations. Any keyword argument is forwarded to Prophet (`seasonality_mode`, `weekly_seasonality`, `holidays`, and so on). `interval_width` defaults to 0.6827 so the reported band is one standard deviation.

- **`fit(df, **fit_kwargs)`** — `df` has `ds` and `y` plus any declared regressors. Streams the training series through `laplace`, transforms `y` to z, fits Prophet on z. Returns `self`.
- **`predict(future)`** — `future` must be strictly after the training data. Returns Prophet's frame with `yhat`, `yhat_lower`, `yhat_upper` mapped back to y-space.
- **`predictive(step, z_mu, z_sigma)`** — returns a y-space predictive (see below) for a given horizon `step` and Prophet z-space mean and standard deviation.
- **`add_regressor(name, **kw)`** — declares an extra regressor, as in Prophet. Returns `self`, so it chains.

### the predictive object

Returned by `predictive`. Exact y-space density.

- **`logpdf(y)`** — log predictive density at `y`.
- **`cdf(y)`** — predictive cdf at `y`, in `[0, 1]`.
- **`quantile(p)`** — inverse cdf at probability `p`.

## Status and caveats

v0 demonstrates interoperability. It maps future frames only, and horizons past `k` reuse the k-step transport, a disclosed approximation. `predict` raises if the future frame is not strictly after the training data. The aim is to show the construction works end to end, then propose it as an option upstream.

## Related

- [skaters](https://github.com/microprediction/skaters) — the online forecasting core, and `laplace` itself.
- [The sandwich](https://skaters.microprediction.org/sandwich.html) — the construction, and the full table of fronted forecasters and detectors.
- [Prophet](https://github.com/facebook/prophet) — the model in the middle.

## License

MIT. See [LICENSE](LICENSE).
