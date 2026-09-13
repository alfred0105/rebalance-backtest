# Tolerance-band rebalancing

The adaptive rotation strategy uses two separate controls:

- `rebalance_band`: do nothing while each asset remains within the allowed percentage-point deviation from its strategic target.
- `max_weekly_shift`: when a deviation breaches the band, move toward the target by at most this many percentage points per weekly evaluation.

The safe asset acts as the balancing sleeve. This prevents a regime or ranking change from forcing an immediate full portfolio rotation.

Example with a 5%p band and 10%p weekly shift:

- Strategic target: 95% stock / 5% safe
- Current: 55% stock / 45% safe
- Next weekly steps: 65/35 -> 75/25 -> 85/15 -> 95/5, assuming the signal remains unchanged.
