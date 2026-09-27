from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
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


STRATEGY_LABELS = {
    "kodex200_buy_hold": "KODEX 200 매수 후 보유",
    "static_60_40": "주식 60% · 채권 40%",
    "safe_only": "단기채권 100%",
    "adaptive_rotation_daily": "적응형 로테이션",
    "broad_signal": "광역 시장신호",
    "peak_only": "피크 보호",
    "hedge_only": "헤지 로테이션",
    "peak_hedge": "피크 보호 + 헤지",
    "peak_hedge_low_turnover": "피크 보호 + 헤지(저회전)",
    "adaptive_whole_shares": "적응형 로테이션(정수주)",
    "peak_hedge_whole_shares": "피크 보호 + 헤지(정수주)",
}

ACTION_LABELS = {
    "BUY": "매수",
    "SELL": "매도",
    "FILL_SLOT": "신규 편입",
    "EMERGENCY_EXIT": "긴급 퇴출",
    "MONTHLY_REPLACE": "월간 교체",
}


def _configure_korean_font() -> None:
    candidates = [
        "AppleGothic",
        "Noto Sans CJK KR",
        "NanumGothic",
        "Malgun Gothic",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


def _ko_bool(value: object) -> str:
    return "예" if bool(value) else "아니오"


def _translate_reason(value: object) -> str:
    text = str(value or "")
    if text.startswith("1D_DROP_"):
        return f"1일 수익률 급락({text.removeprefix('1D_DROP_')})"
    if text.startswith("5D_DROP_"):
        return f"5일 수익률 급락({text.removeprefix('5D_DROP_')})"
    if text.startswith("MARKET_5D_"):
        return f"시장 5일 수익률 급락({text.removeprefix('MARKET_5D_')})"
    if text.startswith("MARKET_20D_"):
        return f"시장 20일 수익률 급락({text.removeprefix('MARKET_20D_')})"
    if text.startswith("PEAK_BREAK_"):
        return "고점 대비 큰 폭 하락과 음의 모멘텀 동시 발생"
    if text == "MARKET_EMERGENCY":
        return "시장 비상상태"
    return text


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


def _equity_chart(
    rotation_curves: pd.DataFrame,
    hedge_curve: pd.DataFrame,
    integer_adaptive: pd.DataFrame,
    integer_peak: pd.DataFrame,
) -> str | None:
    if (
        rotation_curves.empty
        and hedge_curve.empty
        and integer_adaptive.empty
        and integer_peak.empty
    ):
        return None
    fig, ax = plt.subplots(figsize=(11, 4.8))
    if not rotation_curves.empty:
        for col in rotation_curves.columns:
            ax.plot(
                rotation_curves.index,
                rotation_curves[col],
                label=STRATEGY_LABELS.get(str(col), str(col)),
            )
    if not hedge_curve.empty:
        col = hedge_curve.columns[0]
        ax.plot(hedge_curve.index, hedge_curve[col], label="피크 보호 + 헤지")
    if not integer_adaptive.empty:
        ax.plot(
            integer_adaptive.index,
            integer_adaptive.iloc[:, 0],
            label="적응형 로테이션(정수주)",
            linestyle="--",
        )
    if not integer_peak.empty:
        ax.plot(
            integer_peak.index,
            integer_peak.iloc[:, 0],
            label="피크 보호 + 헤지(정수주)",
            linestyle="--",
        )
    ax.set_title("포트폴리오 가치 변화")
    ax.set_ylabel("평가금액(원)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _image_uri(fig)


def _drawdown_chart(rotation_curves: pd.DataFrame, hedge_curve: pd.DataFrame) -> str | None:
    series: dict[str, pd.Series] = {}
    if "adaptive_rotation_daily" in rotation_curves.columns:
        series["적응형 로테이션"] = rotation_curves["adaptive_rotation_daily"]
    if not hedge_curve.empty:
        series["피크 보호 + 헤지"] = hedge_curve.iloc[:, 0]
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for name, values in series.items():
        values = pd.to_numeric(values, errors="coerce").dropna()
        dd = values / values.cummax() - 1.0
        ax.plot(dd.index, dd.values, label=name)
    ax.set_title("고점 대비 낙폭")
    ax.set_ylabel("낙폭")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _image_uri(fig)


def _weights_chart(
    weights: pd.DataFrame,
    names: dict[str, str] | None = None,
) -> str | None:
    if weights.empty:
        return None
    names = names or {}
    sample = weights.copy()
    if isinstance(sample.index, pd.DatetimeIndex) and len(sample):
        cutoff = sample.index.max() - pd.Timedelta(days=730)
        sample = sample.loc[sample.index >= cutoff]

    # Backtest weights are long-only, but floating-point arithmetic can make
    # CASH a tiny negative number such as -1e-16 when risky weights sum to
    # 1.0000000000000002. Pandas stacked-area charts reject any mixed-sign
    # column, so sanitize display-only weights and renormalize each row.
    sample = sample.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    sample = sample.clip(lower=0.0)
    row_sums = sample.sum(axis=1)
    valid = row_sums > 0
    sample.loc[valid] = sample.loc[valid].div(row_sums.loc[valid], axis=0)

    sample = sample.rename(
        columns={
            column: ("현금" if str(column) == "CASH" else names.get(str(column), str(column)))
            for column in sample.columns
        }
    )

    fig, ax = plt.subplots(figsize=(11, 4.2))
    sample.plot.area(ax=ax, linewidth=0)
    ax.set_title("적응형 로테이션 자산 비중 변화 - 최근 2년")
    ax.set_ylabel("포트폴리오 비중")
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
    ax.set_title("매매 발생 시점과 회전율")
    ax.set_ylabel("회전율")
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

    integer_adaptive_equity = _read_csv(
        Path("results_rotation/integer_adaptive_equity.csv")
    )
    integer_adaptive_trades = _read_csv(
        Path("results_rotation/integer_adaptive_trades.csv")
    )
    integer_adaptive_summary = _read_csv(
        Path("results_rotation/integer_adaptive_summary.csv"),
        index_col=0,
    )
    integer_peak_equity = _read_csv(
        Path("results_hedge/integer_peak_hedge_equity.csv")
    )
    integer_peak_trades = _read_csv(
        Path("results_hedge/integer_peak_hedge_trades.csv")
    )
    integer_peak_summary = _read_csv(
        Path("results_hedge/integer_peak_hedge_summary.csv"),
        index_col=0,
    )

    trade_log = (
        integer_adaptive_trades.reset_index()
        if not integer_adaptive_trades.empty
        else _explode_trade_log(rotation_trades, names)
    )
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
        print("(종목 변경 없음)")
    print("\n=== Recent strategy trades ===")
    if not integer_adaptive_trades.empty:
        recent = integer_adaptive_trades.tail(12).reset_index()
    else:
        recent = _explode_trade_log(rotation_trades, names, limit=12)
    if recent.empty:
        print("(기록된 매매 없음)")
    else:
        shown = recent.copy()
        shown["date"] = pd.to_datetime(shown["date"]).dt.strftime("%Y-%m-%d")
        if "delta_weight" in shown:
            shown["delta_weight"] = shown["delta_weight"].map(lambda x: f"{x:+.1%}")
        if "turnover" in shown:
            shown["turnover"] = shown["turnover"].map(lambda x: f"{x:.1%}")
        if "cost" in shown:
            shown["cost"] = shown["cost"].map(lambda x: f"{x:,.0f}")
        if "cost_event" in shown:
            shown["cost_event"] = shown["cost_event"].map(lambda x: f"{x:,.0f}")
        print(shown.to_string(index=False))

    _configure_korean_font()

    eq_img = _equity_chart(
        rotation_curves,
        hedge_curve,
        integer_adaptive_equity,
        integer_peak_equity,
    )
    dd_img = _drawdown_chart(rotation_curves, hedge_curve)
    weights_img = _weights_chart(rotation_weights, names)
    turnover_img = _turnover_chart(rotation_trades)

    plan_html = plan.copy()
    if not plan_html.empty:
        plan_html["target_weight"] = plan_html["target_weight"].map(lambda x: f"{x:.1%}")
        plan_html["actual_weight"] = plan_html["actual_weight"].map(lambda x: f"{x:.1%}")
        plan_html["weight_gap"] = plan_html["weight_gap"].map(lambda x: f"{x:+.1%}")
        for col in ["target_value", "price", "actual_value"]:
            plan_html[col] = plan_html[col].map(lambda x: f"{x:,.0f}")
        plan_html = plan_html[
            ["name", "target_weight", "price", "shares", "actual_weight", "weight_gap"]
        ].rename(
            columns={
                "name": "종목명",
                "target_weight": "목표 비중",
                "price": "현재가",
                "shares": "매수 수량",
                "actual_weight": "실제 비중",
                "weight_gap": "비중 차이",
            }
        )

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        for column in ["ticker", "out", "in"]:
            if column in events_df.columns:
                events_df[column] = events_df[column].map(
                    lambda ticker: names.get(str(ticker), str(ticker)) if ticker else ""
                )
        if "name" in events_df.columns:
            events_df["name"] = events_df["name"].astype(str)
        if "out_name" in events_df.columns:
            events_df["out_name"] = events_df["out_name"].astype(str)
        if "in_name" in events_df.columns:
            events_df["in_name"] = events_df["in_name"].astype(str)
        if "action" in events_df.columns:
            events_df["action"] = events_df["action"].map(
                lambda value: ACTION_LABELS.get(str(value), str(value))
            )
        if "reason" in events_df.columns:
            events_df["reason"] = events_df["reason"].map(_translate_reason)
        if "bucket" in events_df.columns:
            events_df["bucket"] = events_df["bucket"].replace(
                {"aggressive": "공격형", "defensive": "방어형"}
            )
        if "ticker" in events_df.columns and "name" in events_df.columns:
            events_df = events_df.drop(columns=["name"])
        if "out" in events_df.columns and "out_name" in events_df.columns:
            events_df = events_df.drop(columns=["out_name"])
        if "in" in events_df.columns and "in_name" in events_df.columns:
            events_df = events_df.drop(columns=["in_name"])

        events_df = events_df.rename(
            columns={
                "action": "동작",
                "bucket": "구분",
                "ticker": "종목명",
                "score": "점수",
                "reason": "사유",
                "cooldown_until": "재진입 제한 종료일",
                "out": "퇴출 종목",
                "out_score": "퇴출 점수",
                "in": "편입 종목",
                "in_score": "편입 점수",
                "margin": "점수 차이",
            }
        )
        events_df = events_df.fillna("")

    metrics_rows = []
    for label, frame, key in [
        ("적응형 로테이션(이론 비중)", rotation_summary, "adaptive_rotation_daily"),
        ("적응형 로테이션(30만원 정수주)", integer_adaptive_summary, "adaptive_rotation_whole_shares"),
        ("피크 보호 + 헤지(이론 비중)", hedge_summary, "peak_hedge"),
        ("피크 보호 + 헤지(30만원 정수주)", integer_peak_summary, "peak_hedge_whole_shares"),
        ("피크 보호", hedge_summary, "peak_only"),
    ]:
        if frame.empty or key not in frame.index:
            continue
        row = frame.loc[key]
        metrics_rows.append(
            {
                "전략": label,
                "연복리수익률": f"{float(row['cagr']):.2%}",
                "샤프지수": f"{float(row['sharpe']):.3f}",
                "최대낙폭": f"{float(row['max_drawdown']):.2%}",
                "매매횟수": int(float(row["trade_count"])),
                "토스 수수료(원)": f"{float(row['total_transaction_cost']):,.0f}",
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)

    charts = []
    for title, uri in [
        ("포트폴리오 가치 변화", eq_img),
        ("고점 대비 낙폭", dd_img),
        ("포트폴리오 자산 비중", weights_img),
        ("매매 시점과 회전율", turnover_img),
    ]:
        if uri:
            charts.append(
                f"<section class='card'><h2>{html.escape(title)}</h2>"
                f"<img src='{uri}' alt='{html.escape(title)}'></section>"
            )

    event_html = _html_table(events_df)

    trade_log_html = trade_log.tail(40).copy()
    if not trade_log_html.empty:
        if "ticker" in trade_log_html.columns:
            trade_log_html["ticker"] = trade_log_html["ticker"].map(
                lambda ticker: names.get(str(ticker), str(ticker))
            )
        if "action" in trade_log_html.columns:
            trade_log_html["action"] = trade_log_html["action"].map(
                lambda value: ACTION_LABELS.get(str(value), str(value))
            )
        trade_log_html = trade_log_html.rename(
            columns={
                "date": "날짜",
                "ticker": "종목명",
                "name": "종목",
                "action": "매매",
                "shares": "수량",
                "price": "체결가",
                "notional": "거래금액",
                "turnover_event": "회전율",
                "cost_event": "거래비용",
                "delta_weight": "비중 변화",
                "turnover": "회전율",
                "cost": "거래비용",
            }
        )
    recent_html = _html_table(trade_log_html)
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
<title>리밸런싱 전략 대시보드</title><style>{css}</style></head>
<body><main>
<h1>리밸런싱 전략 대시보드</h1>
<p class="muted">기준일 {html.escape(str(active.get('as_of','')))} · 모의 투자금 {capital:,.0f}원 · ETF 1주 단위 실행 기준</p>
<p class="muted">거래비용 가정: 토스증권 KRX 국내 ETF 위탁수수료 0.015% · 매수/매도 체결금액별 적용 · 원 미만 절사 · ETF 증권거래세 0원</p>

<div class="grid">
  <div class="kpi">시장 점수<b>{float(allocation.get('market_score',0)):.3f}</b></div>
  <div class="kpi">주식 목표 비중<b>{float(allocation.get('stock_target',0)):.1%}</b></div>
  <div class="kpi">방어자산 목표 비중<b>{float(allocation.get('defensive_target',0)):.1%}</b></div>
  <div class="kpi">남는 현금<b>{cash:,.0f}원</b></div>
</div>

<section class="card"><h2>30만원 실제 정수주 매수안</h2>
<p class="muted">목표비중을 그대로 소수점 매수할 수 없으므로 1주 단위에서 목표오차가 작아지도록 배정한다.</p>
<div class="scroll">{plan_table_html}</div>
<p><b>남는 현금:</b> {cash:,.0f} KRW ({cash/capital:.1%})</p></section>

<section class="card"><h2>전략 성과 지표</h2><div class="scroll">{metrics_html}</div></section>
{''.join(charts)}

<section class="card"><h2>동적 종목선정 이벤트</h2>
<p class="muted">신규 편입, 긴급 퇴출, 재진입 제한 등 이번 실행에서 발생한 의사결정을 표시한다.</p>
<div class="scroll">{event_html}</div></section>

<section class="card"><h2>과거 매수·매도 내역</h2>
<p class="muted">30만원 정수주 재현 결과를 기준으로 실제 몇 주를 매수·매도했는지 표시한다.</p>
<div class="scroll">{recent_html}</div></section>

<section class="card"><h2>현재 전략 동작 상태</h2>
<p>시장 비상상태: <code>{_ko_bool(active.get('market_emergency'))}</code> ·
피크 보호 활성화: <code>{_ko_bool(allocation.get('peak_lock'))}</code> ·
최근 60일 고점 대비 낙폭: <code>{float(allocation.get('peak_drawdown_60d',0)):.1%}</code></p>
<p class="muted">이론 비중 결과는 전략 자체의 성능 비교용이고, 정수주 결과는 같은 신호를 30만원으로 실제 ETF 1주 단위에 맞춰 실행했을 때의 오차와 잔여 현금을 보여준다.</p>
</section>
</main></body></html>"""

    _configure_korean_font()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")
    finish_status(
        f"[report] done | dashboard={output} | plan={plan_path} | trade_log={trade_log_path}"
    )


if __name__ == "__main__":
    main()
