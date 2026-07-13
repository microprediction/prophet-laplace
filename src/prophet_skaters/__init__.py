"""prophet-skaters: Prophet in a laplace sandwich.

Same Prophet API, calibrated densities. A SandwichedProphet fits Prophet
in the z-coordinates of a laplace forecaster (the Rosenblatt transform of
each observation under the predictive issued for it) and maps every
forecast back through the exact inverse. Prophet keeps its calendar
machinery, holidays, and extra regressors; the sandwich supplies the
volatility clock, the lattice handling, and tails that keep their stated
rates.

Measured on 921 non-price FRED series (pre-registered, see the skaters
repository): raw Prophet trails laplace by 0.76 nats median one-step
log-likelihood (family-weighted 4.6); sandwiched, the gap is 0.02.
"""
from prophet_skaters.sandwich import SandwichedProphet

__all__ = ["SandwichedProphet"]
__version__ = "0.0.1"
