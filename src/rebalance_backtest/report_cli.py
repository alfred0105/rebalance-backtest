from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .console import finish_status, live_status


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build terminal + standalone HTML visualization of strategy behavior."
    )
    p.add_argument("--initial-capital", type=float, default=300_000.0)
    p.add_argument("--snapshot", default=None)
    p.add_argument("--active", default="universes/active_universe.json")
    p.add_argument("--output", default="runs/latest_dashboard.html")
    return p.parse_args()


def _load_active(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ticker_map(snapshot: pd.DataFrame) -> dict[str, str]:
    return dict(
        zip(
            snapshot["yahoo_ticker"].astype(str),
            snapshot["name"].astype(str),
        )
    )


def _price_map(snapshot: pd.DataFrame) -> dict[str, float]:
    prices = pd.to_numeric(snapshot["price_listing"], errors="coerce")
    return {
        str(ticker): float(price)
        for ticker, price in zip(snapshot["yahoo_ticker"], prices)
        if np.isfinite(price) and price > 0
    }


def _integer_plan(
    weights: dict[str, float],
    prices: dict[str, float],
    names: dict[str, str],
    capital: float,
) -> tuple[pd.DataFrame, float]:
    items = []
    for ticker, weight in weights.items():
        price = prices.get(ticker)
        if price is None or price <= 0 or weight <= 0:
            continue
        target_value = capital * float(weight)
        shares = int(np.floor(target_value / price))
        items.append(
            {
                "ticker": ticker,
                "name": names.get(ticker, ticker),
                "target_weight": float(weight),
                "target_value": target_value,
                "price": price,
                "shares": shares,
            }
        )

    if not items:
        return pd.DataFrame(), capital

    def actual_values(rows: list[dict]) -> tuple[np.ndarray, float]:
        values = np.asarray(
            [row["shares"] * row["price"] for row in rows],
            dtype=float,
        )
        cash = capital - float(values.sum())
        return values, cash

    def error(rows: list[dict]) -> float:
        values, cash = actual_values(rows)
        actual_w = values / capital
        target_w = np.asarray([row["target_weight"] for row in rows], dtype=float)
        return float(np.square(actual_w - target_w).sum() + (cash / capital) ** 2)

    while True:
        values, cash = actual_values(items)
        base_error = error(items)
        best_idx = None
        best_error = base_error
        for idx, row in enumerate(items):
            if row["price"] > cash + 1e-9:
                continue
            row["shares"] += 1
            trial_error = error(items)
            row["shares"] -= 1
            if trial_error + 1e-12 < best_error:
                best_error = trial_error
                best_idx = idx
        if best_idx is None:
            break
        items[best_idx]["shares"] += 1

    values, cash = actual_values(items)
    frame = pd.DataFrame(items)
    frame["actual_value"] = values
    frame["actual_weight"] = frame["actual_value"] / capital
    frame["weight_gap"] = frame["actual_weight"] - frame["target_weight"]
    frame = frame.sort_values("target_weight", ascending=False).reset_index(drop=True)
    return frame, cash


def _read_csv(path: Path, *, index_col: int | None = 0) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, index_col=index_col)
    if index_col is not None and len(frame):
        try:
            frame.index = pd.to_datetime(frame.index)
        except Exception:
            pass
    return frame


def _image_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _equity_chart(rotation_curves: pd.DataFrame, hedge_curve: pd.DataFrame) -> str | None:
    if rotation_curves.empty and hedge_curve.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 4.8))
    if not rotation_curves.empty:
        for col in rotation_curves.columns:
            ax.plot(rotation_curves.index, rotation_curves[col], label=col)
    if not hedge_curve.empty:
        col = hedge_curve.columns[0]
        ax.plot(hedge_curve.index, hedge_curve[col], label="peak_hedge")
    ax.set_title("Portfolio value over time")
    ax.set_ylabel("KRW")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _image_uri(fig)


def _drawdown_chart(rotation_curves: pd.DataFrame, hedge_curve: pd.DataFrame) -> str | None:
    series: dict[str, pd.Series] = {}
    if "adaptive_rotation_daily" in rotation_curves.columns:
        series["adaptive_rotation_daily"] = rotation_curves["adaptive_rotation_daily"]
    if not hedge_curve.empty:
        series["peak_hedge"] = hedge_curve.iloc[:, 0]
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for name, values in series.items():
        values = pd.to_numeric(values, errors="coerce").dropna()
        dd = values / values.cummax() - 1.0
        ax.plot(dd.index, dd.values, label=name)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _image_uri(fig)


def _weights_chart(weights: pd.DataFrame) -> str | None:
    if weights.empty:
        return None
    sample = weights.copy()
    if isinstance(sample.index, pd.DatetimeIndex) and len(sample):
        cutoff = sample.index.max() - pd.Timedelta(days=730)
        sample = sample.loc[sample.index >= cutoff]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    sample.plot.area(ax=ax, linewidth=0)
    ax.set_title("Adaptive rotation weights - recent 2 years")
    ax.set_ylabel("Weight")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return _image_uri(fig)


def _turnover_chart(trades: pd.DataFrame) -> str | None:
    if trades.empty or "turnover" not in trades.columns:
        return None
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.vlines(
        trades.index,
        0,
        pd.to_numeric(trades["turnover"], errors="coerce").fillna(0.0),
        linewidth=0.8,
    )
    ax.set_title("Trade turnover timeline")
    ax.set_ylabel("Turnover")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _image_uri(fig)


def _explode_trade_log(
    trades: pd.DataFrame,
    names: dict[str, str],
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=["date", "action", "ticker", "name", "delta_weight", "turnover", "cost"]
        )
    delta_cols = [col for col in trades.columns if str(col).startswith("delta_")]
    rows: list[dict[str, object]] = []
    for date, row in trades.iterrows():
        for col in delta_cols:
            ticker = str(col)[len("delta_") :]
            if ticker == "CASH":
                continue
            delta = float(row[col])
            if abs(delta) < 0.005:
                continue
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "action": "BUY" if delta > 0 else "SELL",
                    "ticker": ticker,
                    "name": names.get(ticker, ticker),
                    "delta_weight": delta,
                    "turnover": float(row.get("turnover", np.nan)),
                    "cost": float(row.get("cost", np.nan)),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values("date")
    if limit is not None:
        frame = frame.tail(limit)
    return frame.reset_index(drop=True)


def _fmt_plan(frame: pd.DataFrame, cash: float, capital: float) -> str:
    if frame.empty:
        return "(no executable plan)"
    shown = frame.copy()
    shown["target_weight"] = shown["target_weight"].map(lambda x: f"{x:.1%}")
    shown["actual_weight"] = shown["actual_weight"].map(lambda x: f"{x:.1%}")
    shown["weight_gap"] = shown["weight_gap"].map(lambda x: f"{x:+.1%}")
    for col in ["target_value", "price", "actual_value"]:
        shown[col] = shown[col].map(lambda x: f"{x:,.0f}")
    text = shown[
        [
            "ticker",
            "name",
            "target_weight",
            "price",
            "shares",
            "actual_weight",
            "weight_gap",
        ]
    ].to_string(index=False)
    return text + f"\nCASH {cash:,.0f} KRW ({cash/capital:.1%})"


def _html_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "<p class='muted'>No rows.</p>"
    shown = frame.tail(max_rows).copy()
    for col in shown.columns:
        if pd.api.types.is_datetime64_any_dtype(shown[col]):
            shown[col] = shown[col].dt.strftime("%Y-%m-%d")
    return shown.to_html(index=False, escape=True, classes="data-table", border=0)


def main() -> None:
    args = _parse_args()
    capital = float(args.initial_capital)

    active_path = Path(args.active)
    if not active_path.exists():
        raise FileNotFoundError(active_path)
    if args.snapshot:
        snapshot_path = Path(args.snapshot)
        if not snapshot_path.exists():
            raise FileNotFoundError(snapshot_path)
    else:
        snapshots = sorted(Path("universes/snapshots").glob("*.csv"))
        if not snapshots:
            raise FileNotFoundError("No monthly universe snapshot found.")
        snapshot_path = snapshots[-1]

    live_status("[report] loading strategy outputs and 300k execution plan")
    active = _load_active(active_path)
    snapshot = pd.read_csv(snapshot_path)
    names = _ticker_map(snapshot)
    prices = _price_map(snapshot)

    ideal_weights = {
        str(k): float(v)
        for k, v in active.get("allocation", {}).get("ideal_target_weights", {}).items()
    }
    plan, cash = _integer_plan(ideal_weights, prices, names, capital)
    plan_path = Path("runs/latest_execution_plan.csv")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(plan_path, index=False)

    rotation_curves = _read_csv(Path("results_rotation/equity_curves.csv"))
    rotation_weights = _read_csv(
        Path("results_rotation/weights_adaptive_rotation_daily.csv")
    )
    rotation_trades = _read_csv(
        Path("results_rotation/trades_adaptive_rotation_daily.csv")
    )
    rotation_summary = _read_csv(Path("results_rotation/summary.csv"), index_col=0)

    hedge_curve = _read_csv(Path("results_hedge/equity_peak_hedge.csv"))
    hedge_trades = _read_csv(Path("results_hedge/trades_peak_hedge.csv"))
    hedge_summary = _read_csv(Path("results_hedge/hedge_summary.csv"), index_col=0)

    trade_log = _explode_trade_log(rotation_trades, names)
    trade_log_path = Path("runs/latest_trade_log.csv")
    trade_log.to_csv(trade_log_path, index=False)

    print("\n=== 300,000 KRW executable allocation ===")
    print(_fmt_plan(plan, cash, capital))
    print("\n=== Dynamic selector events ===")
    events = active.get("events") or []
    if events:
        for event in events:
            print(json.dumps(event, ensure_ascii=False, sort_keys=True))
    else:
        print("(no selector changes)")
    print("\n=== Recent strategy trades ===")
    recent = _explode_trade_log(rotation_trades, names, limit=12)
    if recent.empty:
        print("(no recorded trades)")
    else:
        shown = recent.copy()
        shown["date"] = shown["date"].dt.strftime("%Y-%m-%d")
        shown["delta_weight"] = shown["delta_weight"].map(lambda x: f"{x:+.1%}")
        shown["turnover"] = shown["turnover"].map(lambda x: f"{x:.1%}")
        shown["cost"] = shown["cost"].map(lambda x: f"{x:,.0f}")
        print(shown.to_string(index=False))

    eq_img = _equity_chart(rotation_curves, hedge_curve)
    dd_img = _drawdown_chart(rotation_curves, hedge_curve)
    weights_img = _weights_chart(rotation_weights)
    turnover_img = _turnover_chart(rotation_trades)

    plan_html = plan.copy()
    if not plan_html.empty:
        plan_html["target_weight"] = plan_html["target_weight"].map(lambda x: f"{x:.1%}")
        plan_html["actual_weight"] = plan_html["actual_weight"].map(lambda x: f"{x:.1%}")
        plan_html["weight_gap"] = plan_html["weight_gap"].map(lambda x: f"{x:+.1%}")
        for col in ["target_value", "price", "actual_value"]:
            plan_html[col] = plan_html[col].map(lambda x: f"{x:,.0f}")

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df = events_df.fillna("")

    metrics_rows = []
    for label, frame, key in [
        ("Adaptive Rotation", rotation_summary, "adaptive_rotation_daily"),
        ("Peak + Hedge", hedge_summary, "peak_hedge"),
        ("Peak only", hedge_summary, "peak_only"),
    ]:
        if frame.empty or key not in frame.index:
            continue
        row = frame.loc[key]
        metrics_rows.append(
            {
                "strategy": label,
                "CAGR": f"{float(row['cagr']):.2%}",
                "Sharpe": f"{float(row['sharpe']):.3f}",
                "MDD": f"{float(row['max_drawdown']):.2%}",
                "trades": int(float(row["trade_count"])),
                "cost_KRW": f"{float(row['total_transaction_cost']):,.0f}",
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)

    charts = []
    for title, uri in [
        ("Portfolio value", eq_img),
        ("Drawdown", dd_img),
        ("Portfolio weights", weights_img),
        ("Trade timing / turnover", turnover_img),
    ]:
        if uri:
            charts.append(
                f"<section class='card'><h2>{html.escape(title)}</h2>"
                f"<img src='{uri}' alt='{html.escape(title)}'></section>"
            )

    event_html = _html_table(events_df)
    recent_html = _html_table(trade_log.tail(40))
    metrics_html = _html_table(metrics_df)
    plan_table_html = _html_table(plan_html)

    allocation = active.get("allocation", {})
    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:0;background:#f5f6f8;color:#17191c}
    main{max-width:1180px;margin:0 auto;padding:28px}
    h1{margin-bottom:4px}.muted{color:#6b7280}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
    .kpi,.card{background:white;border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:12px 0}
    .kpi b{display:block;font-size:24px;margin-top:6px}
    img{width:100%;height:auto}.data-table{width:100%;border-collapse:collapse;font-size:13px}
    .data-table th,.data-table td{padding:7px 8px;border-bottom:1px solid #eceff3;text-align:right}
    .data-table th:first-child,.data-table td:first-child{text-align:left}
    .scroll{overflow-x:auto}
    code{background:#eef0f3;padding:2px 5px;border-radius:5px}
    """

    html_doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rebalance Backtest Dashboard</title><style>{css}</style></head>
<body><main>
<h1>Rebalance Backtest Dashboard</h1>
<p class="muted">As of {html.escape(str(active.get('as_of','')))} · paper capital {capital:,.0f} KRW · whole-share execution plan</p>

<div class="grid">
  <div class="kpi">Market score<b>{float(allocation.get('market_score',0)):.3f}</b></div>
  <div class="kpi">Stock target<b>{float(allocation.get('stock_target',0)):.1%}</b></div>
  <div class="kpi">Defensive target<b>{float(allocation.get('defensive_target',0)):.1%}</b></div>
  <div class="kpi">Executable cash<b>{cash:,.0f} KRW</b></div>
</div>

<section class="card"><h2>30만원 실제 정수주 매수안</h2>
<p class="muted">목표비중을 그대로 소수점 매수할 수 없으므로 1주 단위에서 목표오차가 작아지도록 배정한다.</p>
<div class="scroll">{plan_table_html}</div>
<p><b>남는 현금:</b> {cash:,.0f} KRW ({cash/capital:.1%})</p></section>

<section class="card"><h2>Strategy metrics</h2><div class="scroll">{metrics_html}</div></section>
{''.join(charts)}

<section class="card"><h2>Dynamic selector events</h2>
<p class="muted">종목 선정, 긴급퇴출, cooldown 등 현재 실행에서 발생한 의사결정.</p>
<div class="scroll">{event_html}</div></section>

<section class="card"><h2>Historical buy / sell timeline</h2>
<p class="muted">과거 백테스트의 실제 주식 수가 아니라 목표 비중 변화 기준 BUY/SELL 로그다.</p>
<div class="scroll">{recent_html}</div></section>

<section class="card"><h2>How the engine is operating</h2>
<p>Market emergency: <code>{html.escape(str(active.get('market_emergency')))}</code> ·
Peak lock: <code>{html.escape(str(allocation.get('peak_lock')))}</code> ·
Peak DD 60d: <code>{float(allocation.get('peak_drawdown_60d',0)):.1%}</code></p>
<p class="muted">Historical strategy curves remain fractional-weight backtests for comparability. The 300k table above is a separate whole-share execution feasibility layer.</p>
</section>
</main></body></html>"""

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")
    finish_status(
        f"[report] done | dashboard={output} | plan={plan_path} | trade_log={trade_log_path}"
    )


if __name__ == "__main__":
    main()
