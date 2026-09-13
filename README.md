# rebalance-backtest

A small, inspectable Python backtester for the strategy idea we discussed:

- **check the portfolio weekly** rather than only monthly,
- measure trend with **multi-horizon log returns**,
- normalize trend by realized volatility,
- squash extreme signals with **`tanh`**,
- give **positive momentum a slightly larger multiplier** than negative momentum,
- tilt toward stronger assets instead of mechanically resetting all weights,
- **de-risk faster than adding risk**,
- use a **no-trade band** so weekly checks do not automatically mean weekly turnover,
- include transaction costs and a one-session execution lag.

The repository also compares the idea against buy-and-hold, fixed monthly rebalancing, and fixed weekly rebalancing.

## Strategy

For asset `i` and lookback `h`:

```text
z(i,h) = ln(P_t / P_(t-h)) / (sigma_daily * sqrt(h))
```

The default model blends 20, 60, and 120 trading-day signals:

```text
m_i = 0.5*z(i,20) + 0.3*z(i,60) + 0.2*z(i,120)
s_i = tanh(k * m_i)
```

Positive and negative signals are deliberately asymmetric:

```text
s*_i = 1.25*s_i   if s_i >= 0
       0.85*s_i   if s_i < 0
```

The signal changes the composition around the base weights. A portfolio-level
signal can also reduce total risky exposure and move the remainder to cash.
The default exit speed is faster than the entry speed.

## Look-ahead handling

A signal calculated using prices through day `t` is not applied until the next
available trading session. The engine therefore does not use the same close to
both generate a signal and claim an already-earned return.

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -e .[dev]
```

## Run the comparison

Equal-weight example:

```bash
rebalance-backtest \
  --tickers SPY QQQ TLT GLD \
  --start 2010-01-01 \
  --transaction-cost-bps 5
```

Custom base weights:

```bash
rebalance-backtest \
  --tickers SPY QQQ TLT GLD \
  --weights 0.35 0.25 0.20 0.20 \
  --start 2010-01-01 \
  --transaction-cost-bps 5
```

Results are written to `results/`:

- `summary.csv`
- `equity_curves.csv`
- `equity_curves.png`
- weight histories for each strategy
- trade histories for each strategy

The summary includes total return, CAGR, annualized volatility, Sharpe, Sortino,
maximum drawdown, Calmar, trade count, turnover, and modeled transaction cost.

## Baselines

The CLI runs four variants on exactly the same price data:

1. `buy_hold` — initial allocation, no rebalancing
2. `fixed_monthly` — mechanical monthly reset to base weights
3. `fixed_weekly` — mechanical weekly reset to base weights
4. `momentum_weekly` — weekly evaluation with the asymmetric momentum overlay

This makes the main hypothesis test explicit: **does weekly evaluation plus a
momentum-aware threshold improve return / drawdown after turnover costs, rather
than merely trading more often?**

## Parameters worth sweeping next

The first research pass should sweep at least:

- positive multiplier: `1.0 -> 1.5`
- negative multiplier: `0.5 -> 1.0`
- no-trade band: `0% -> 5%`
- entry speed: `0.25 -> 1.0`
- exit speed: `0.5 -> 1.0`
- tilt strength: `0.1 -> 1.0`
- transaction cost: `0 -> 25 bps`
- evaluation frequency: weekly vs biweekly vs monthly

Do not pick the best in-sample parameter set and call it finished. Split the
sample, use walk-forward or rolling out-of-sample tests, and inspect whether the
result survives different market regimes and cost assumptions.

## Notes

- v1 is long-only.
- cash earns 0% in v1.
- leverage and borrowing costs are intentionally excluded for now.
- Yahoo Finance is used only as a convenient data source; the engine accepts any
  aligned daily close-price DataFrame.
- This is a research/backtesting project, not investment advice.
