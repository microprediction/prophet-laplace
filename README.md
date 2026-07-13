# prophet-laplace

*"He's not the messiah. He's a very naughty boy."* ([why](https://medium.com/geekculture/is-facebooks-prophet-the-time-series-messiah-or-just-a-very-naughty-boy-8b71b136bc8c)) *This package makes him useful anyway.*

Prophet in a laplace sandwich: same API, calibrated densities.

```python
from prophet_laplace import SandwichedProphet

m = SandwichedProphet(k=30)          # accepts Prophet's kwargs too
m.fit(df)                            # ds, y (+ regressors as usual)
fc = m.predict(future)               # yhat / yhat_lower / yhat_upper,
                                     # mapped back exactly
```

## What it does

`SandwichedProphet` fits Prophet in the z-coordinates of a
[skaters](https://github.com/microprediction/skaters) laplace forecaster
(the Rosenblatt transform of each observation under the predictive issued
for it) and maps every forecast back through the exact inverse. Prophet
keeps its calendar decomposition, holidays, and extra regressors; the
sandwich supplies the volatility clock, repeated-value handling, and
tails whose stated probabilities come true.

## Why

Measured on 921 non-price FRED series under a pre-registered protocol
(statements, frozen universe, and results in the skaters repository):

|            | median one-step LL vs laplace | family-weighted (120 families) |
|------------|------------------------------|-------------------------------|
| Prophet raw | -0.755 nats | -4.60 nats |
| Prophet sandwiched | **-0.020 nats** | **-0.025 nats** |

The sandwich closes 97% of Prophet's density gap without retraining
anything. `predictive(step, z_mu, z_sigma)` exposes the full y-space
density (logpdf, cdf, quantile) for scoring and risk use.

## Status

v0: future frames only, horizons past `k` reuse the k-step transport
(disclosed approximation). The goal is to demonstrate interoperability,
then propose the option upstream.
