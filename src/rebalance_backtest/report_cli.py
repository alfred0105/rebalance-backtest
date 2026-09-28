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
from .execution_plan import build_execution_plan


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a Korean terminal + HTML strategy execution dashboard."
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
    "adaptive_rotation_whole_shares": "적응형 로테이션(정수주)",
    "peak_hedge_whole_shares": "피크 보호 + 헤지(정수주)",
}

ACTION_LABELS = {
    "BUY": "매수",
    "SELL": "매도",
    "FILL_SLOT": "신규 편입",
    "EMERGENCY_EXIT": "긴급 퇴출",
    "MONTHLY_REPLACE": "월간 교체",
}

RISK_STATE_LABELS = {
    "NORMAL": "정상",
    "PEAK_LOCK": "피크 보호",
    "RECOVERY_1": "회복 1단계",
    "RECOVERY_2": "회복 2단계",
    "RECOVERY_3": "회복 3단계",
    "EMERGENCY": "시장 비상",
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
    mappings = (
        ("1D_HARD_STOP_", "1일 절대 급락 한도 도달"),
        ("5D_HARD_STOP_", "5일 절대 급락 한도 도달"),
        ("1D_VOL_SHOCK_", "1일 변동성 대비 비정상 급락"),
        ("5D_VOL_SHOCK_", "5일 변동성 대비 비정상 급락"),
        ("1D_DROP_", "1일 수익률 급락"),
        ("5D_DROP_", "5일 수익률 급락"),
        ("MARKET_5D_", "시장 5일 수익률 급락"),
        ("MARKET_20D_", "시장 20일 수익률 급락"),
    )
    for prefix, label in mappings:
        if text.startswith(prefix):
            return f"{label} ({text.removeprefix(prefix)})"
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
    result = build_execution_plan(weights, prices, names, capital)
    return result.positions, result.cash


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
    if not integer_peak.empty:
        ax.plot(
            integer_peak.index,
            integer_peak.iloc[:, 0],
            label="피크 보호 + 헤지(30만원 정수주)",
            linewidth=2.0,
        )
    if not integer_adaptive.empty:
        ax.plot(
            integer_adaptive.index,
            integer_adaptive.iloc[:, 0],
            label="적응형 로테이션(30만원 정수주)",
            linewidth=1.2,
        )
    if integer_peak.empty and not hedge_curve.empty:
        ax.plot(
            hedge_curve.index,
            hedge_curve.iloc[:, 0],
            label="피크 보호 + 헤지",
        )
    ax.set_title("30만원 기준 포트폴리오 가치 변화")
    ax.set_ylabel("평가금액(원)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _image_uri(fig)


def _drawdown_chart(
    integer_adaptive: pd.DataFrame,
    integer_peak: pd.DataFrame,
) -> str | None:
    series: dict[str, pd.Series] = {}
    if not integer_adaptive.empty:
        series["적응형 로테이션"] = integer_adaptive.iloc[:, 0]
    if not integer_peak.empty:
        series["피크 보호 + 헤지"] = integer_peak.iloc[:, 0]
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

    sample = sample.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    sample = sample.clip(lower=0.0)
    row_sums = sample.sum(axis=1)
    valid = row_sums > 0
    sample.loc[valid] = sample.loc[valid].div(row_sums.loc[valid], axis=0)
    sample = sample.rename(
        columns={
            column: (
                "현금"
                if str(column) == "CASH"
                else names.get(str(column), str(column))
            )
            for column in sample.columns
        }
    )

    fig, ax = plt.subplots(figsize=(11, 4.2))
    sample.plot.area(ax=ax, linewidth=0)
    ax.set_title("검증 전략의 자산 비중 변화 - 최근 2년")
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


def _html_table(frame: pd.DataFrame, *, max_rows: int = 50) -> str:
    if frame.empty:
        return "<p class='muted'>표시할 데이터가 없습니다.</p>"
    shown = frame.tail(max_rows).copy()
    for col in shown.columns:
        if pd.api.types.is_datetime64_any_dtype(shown[col]):
            shown[col] = shown[col].dt.strftime("%Y-%m-%d")
    return shown.to_html(index=False, escape=True, classes="data-table", border=0)


def _format_plan_for_html(plan: pd.DataFrame) -> pd.DataFrame:
    if plan.empty:
        return plan
    shown = plan.copy()
    shown["목표 비중"] = shown["target_weight"].map(lambda x: f"{x:.1%}")
    shown["실제 비중"] = shown["actual_weight"].map(lambda x: f"{x:.1%}")
    shown["비중 차이"] = shown["weight_gap"].map(lambda x: f"{x:+.1%}")
    shown["현재가"] = shown["price"].map(lambda x: f"{int(round(x)):,}원")
    shown["평가금액"] = shown["actual_value"].map(lambda x: f"{int(round(x)):,}원")
    shown["토스 수수료"] = shown["commission"].map(lambda x: f"{int(x):,}원")
    shown["상태"] = [
        (
            "자금제약으로 미편입"
            if bool(blocked) and int(shares) == 0
            else "미편입"
            if int(shares) == 0
            else "보유/매수"
        )
        for blocked, shares in zip(shown["blocked_by_size"], shown["shares"])
    ]
    return shown[
        [
            "name",
            "target_weight",
            "actual_weight",
            "price",
            "shares",
            "actual_value",
            "commission",
            "weight_gap",
            "상태",
        ]
    ].rename(
        columns={
            "name": "종목명",
            "target_weight": "목표 비중",
            "actual_weight": "실제 비중",
            "price": "현재가",
            "shares": "수량",
            "actual_value": "평가금액",
            "commission": "토스 수수료",
            "weight_gap": "비중 차이",
        }
    ).assign(
        **{
            "목표 비중": shown["목표 비중"],
            "실제 비중": shown["실제 비중"],
            "현재가": shown["현재가"],
            "평가금액": shown["평가금액"],
            "토스 수수료": shown["토스 수수료"],
            "비중 차이": shown["비중 차이"],
        }
    )[
        ["종목명", "목표 비중", "현재가", "수량", "평가금액", "실제 비중", "비중 차이", "토스 수수료", "상태"]
    ]


def _format_trade_table(
    trades: pd.DataFrame,
    names: dict[str, str],
) -> pd.DataFrame:
    if trades.empty:
        return trades
    shown = trades.copy().reset_index()
    if "date" not in shown.columns and shown.columns[0] == "index":
        shown = shown.rename(columns={"index": "date"})
    shown["종목명"] = shown["ticker"].map(
        lambda ticker: names.get(str(ticker), str(ticker))
    )
    shown["매매"] = shown["action"].map(
        lambda value: ACTION_LABELS.get(str(value), str(value))
    )
    shown["수량"] = shown["shares"].astype(int)
    shown["체결가"] = shown["price"].map(lambda x: f"{int(round(float(x))):,}원")
    shown["거래금액"] = shown["notional"].map(
        lambda x: f"{int(round(float(x))):,}원"
    )
    shown["토스 수수료"] = shown.get("commission", 0)
    shown["토스 수수료"] = shown["토스 수수료"].map(
        lambda x: f"{int(round(float(x))):,}원"
    )
    return shown[["종목명", "매매", "수량", "체결가", "거래금액", "토스 수수료"]]


def _portfolio_from_row(
    row: pd.Series,
    names: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in row.index:
        column = str(column)
        if not column.startswith("shares_"):
            continue
        ticker = column.removeprefix("shares_")
        shares = int(round(float(row[column])))
        if shares <= 0:
            continue
        value = float(row.get(f"value_{ticker}", 0.0))
        weight = float(row.get(f"weight_{ticker}", 0.0))
        rows.append(
            {
                "종목명": names.get(ticker, ticker),
                "수량": shares,
                "평가금액": f"{int(round(value)):,}원",
                "비중": f"{weight:.1%}",
            }
        )
    cash = float(row.get("CASH", 0.0))
    cash_weight = float(row.get("weight_CASH", 0.0))
    rows.append(
        {
            "종목명": "현금",
            "수량": "",
            "평가금액": f"{int(round(cash)):,}원",
            "비중": f"{cash_weight:.1%}",
        }
    )
    return pd.DataFrame(rows)


def _trade_change_summary(
    trades: pd.DataFrame,
    names: dict[str, str],
    *,
    max_parts: int = 4,
) -> str:
    if trades.empty:
        return "비중 조정"
    parts: list[str] = []
    for _, row in trades.iterrows():
        ticker = str(row["ticker"])
        action = "매수" if str(row["action"]) == "BUY" else "매도"
        parts.append(
            f"{names.get(ticker, ticker)} {action} {int(row['shares'])}주"
        )
    if len(parts) > max_parts:
        return " · ".join(parts[:max_parts]) + f" 외 {len(parts)-max_parts}건"
    return " · ".join(parts)


def _historical_change_rows(
    portfolio: pd.DataFrame,
    trades: pd.DataFrame,
    names: dict[str, str],
    *,
    max_events: int = 80,
) -> str:
    if portfolio.empty or trades.empty:
        return "<p class='muted'>정수주 변경 이력이 아직 없습니다.</p>"

    portfolio = portfolio.sort_index()
    trades = trades.sort_index()
    dates = list(pd.DatetimeIndex(trades.index.unique()))[-max_events:][::-1]
    rows: list[str] = []

    for idx, date in enumerate(dates):
        if date not in portfolio.index:
            continue
        loc = portfolio.index.get_indexer([date])[0]
        after = portfolio.iloc[loc]
        before = portfolio.iloc[max(0, loc - 1)]
        day_trades = trades.loc[[date]] if date in trades.index else pd.DataFrame()
        detail_id = f"hist-{idx}"
        change_text = html.escape(_trade_change_summary(day_trades, names))
        equity = float(after.get("equity", 0.0))
        cost = (
            float(pd.to_numeric(day_trades.get("commission", 0), errors="coerce").fillna(0).sum())
            if not day_trades.empty
            else 0.0
        )

        before_html = _html_table(_portfolio_from_row(before, names))
        trade_html = _html_table(_format_trade_table(day_trades, names))
        after_html = _html_table(_portfolio_from_row(after, names))

        rows.append(
            f"""
<tr class="change-row" onclick="toggleRow('{detail_id}')">
  <td>{date:%Y-%m-%d}</td>
  <td>리밸런싱</td>
  <td>{change_text}</td>
  <td>{equity:,.0f}원</td>
  <td>{cost:,.0f}원</td>
  <td>펼치기</td>
</tr>
<tr id="{detail_id}" class="detail-row" style="display:none">
  <td colspan="6">
    <div class="detail-grid">
      <div><h4>변경 전 포트폴리오</h4>{before_html}</div>
      <div><h4>실제 정수주 매매</h4>{trade_html}</div>
      <div><h4>변경 후 포트폴리오</h4>{after_html}</div>
    </div>
    <p class="reason"><b>동작 이유:</b> 피크 보호 + 헤지 검증전략의 목표비중 변경을 정수주로 재현.</p>
  </td>
</tr>"""
        )

    return (
        "<table class='data-table change-table'><thead><tr>"
        "<th>날짜</th><th>상태</th><th>주요 변화</th><th>평가금액</th>"
        "<th>수수료</th><th>상세</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _plan_signature(frame: pd.DataFrame) -> tuple:
    if frame.empty:
        return tuple()
    use = frame.loc[frame["ticker"].astype(str) != "CASH"].copy()
    if "shares" not in use:
        return tuple()
    return tuple(
        sorted(
            (str(row["ticker"]), int(row["shares"]))
            for _, row in use.iterrows()
            if int(row["shares"]) > 0
        )
    )


def _paper_portfolio_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    shown = frame.copy()
    if "ticker" in shown.columns:
        risky = shown["ticker"].astype(str) != "CASH"
        shares = pd.to_numeric(shown.get("shares", 0), errors="coerce").fillna(0)
        shown = shown.loc[(~risky) | (shares > 0)].copy()
    shown["종목명"] = shown.get("name", shown.get("ticker", "")).astype(str)
    shown["수량"] = pd.to_numeric(shown.get("shares", 0), errors="coerce").fillna(0).astype(int)
    shown["현재가"] = pd.to_numeric(shown.get("price", 0), errors="coerce").fillna(0).map(
        lambda x: "" if x <= 1 else f"{int(round(x)):,}원"
    )
    shown["평가금액"] = pd.to_numeric(shown.get("value", 0), errors="coerce").fillna(0).map(
        lambda x: f"{int(round(x)):,}원"
    )
    shown["비중"] = pd.to_numeric(shown.get("weight", 0), errors="coerce").fillna(0).map(
        lambda x: f"{float(x):.1%}"
    )
    shown.loc[shown.get("ticker", "").astype(str) == "CASH", "수량"] = ""
    return shown[["종목명", "수량", "현재가", "평가금액", "비중"]]


def _paper_change_rows(
    portfolio_dir: Path,
    trades_path: Path,
    active_dir: Path,
    names: dict[str, str],
    *,
    max_events: int = 100,
) -> str:
    files = sorted(portfolio_dir.glob("*.csv"))
    if not files:
        return "<p class='muted'>paper 계좌 snapshot이 아직 없습니다.</p>"

    trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    rows: list[str] = []
    previous_frame = pd.DataFrame()
    previous_signature: tuple = tuple()
    previous_risk: str | None = None

    for path in files:
        date_key = path.stem
        frame = pd.read_csv(path)
        signature = tuple(
            sorted(
                (
                    str(row["ticker"]),
                    int(row["shares"]),
                )
                for _, row in frame.iterrows()
                if str(row["ticker"]) != "CASH" and int(row.get("shares", 0)) > 0
            )
        )
        active_path = active_dir / f"{date_key}.json"
        active = (
            json.loads(active_path.read_text(encoding="utf-8"))
            if active_path.exists()
            else {}
        )
        allocation = active.get("allocation", {})
        risk = str(allocation.get("risk_state", "NORMAL"))
        events = active.get("events") or []

        day_trades = pd.DataFrame()
        if not trades.empty and "date" in trades.columns:
            day_trades = trades.loc[trades["date"].astype(str) == date_key].copy()

        changed = (
            not previous_frame.empty
            and (
                signature != previous_signature
                or risk != previous_risk
                or bool(events)
                or not day_trades.empty
            )
        )
        if previous_frame.empty:
            changed = True

        if changed:
            reasons: list[str] = []
            for event in events:
                action = ACTION_LABELS.get(
                    str(event.get("action")),
                    str(event.get("action", "")),
                )
                ticker = str(event.get("ticker") or event.get("in") or "")
                name = names.get(
                    ticker,
                    str(event.get("name") or event.get("in_name") or ticker),
                )
                reason = _translate_reason(event.get("reason"))
                item = f"{action}: {name}"
                if reason:
                    item += f" · {reason}"
                reasons.append(item)
            if not day_trades.empty:
                reasons.append(_trade_change_summary(day_trades, names))
            if not reasons:
                reasons.append("위험상태 또는 목표 포트폴리오 변경")

            after_equity = float(
                pd.to_numeric(frame.get("value", 0), errors="coerce").fillna(0).sum()
            )
            fees = (
                float(pd.to_numeric(day_trades.get("commission", 0), errors="coerce").fillna(0).sum())
                if not day_trades.empty
                else 0.0
            )
            detail_id = f"paper-{len(rows)}"
            before_html = (
                _html_table(_paper_portfolio_table(previous_frame))
                if not previous_frame.empty
                else "<p class='muted'>첫 paper 계좌 기록</p>"
            )
            after_html = _html_table(_paper_portfolio_table(frame))
            trade_html = (
                _html_table(_format_trade_table(day_trades.set_index("date"), names))
                if not day_trades.empty
                else "<p class='muted'>실제 매매 없음</p>"
            )
            reason_text = " / ".join(reasons)
            rows.append(
                f"""
<tr class="change-row" onclick="toggleRow('{detail_id}')">
  <td>{html.escape(date_key)}</td>
  <td>{html.escape(RISK_STATE_LABELS.get(risk, risk))}</td>
  <td>{html.escape(reason_text)}</td>
  <td>{after_equity:,.0f}원</td>
  <td>{fees:,.0f}원</td>
  <td>펼치기</td>
</tr>
<tr id="{detail_id}" class="detail-row" style="display:none">
  <td colspan="6">
    <div class="detail-grid">
      <div><h4>변경 전 paper 포트폴리오</h4>{before_html}</div>
      <div><h4>그날 실제 paper 매매</h4>{trade_html}</div>
      <div><h4>변경 후 paper 포트폴리오</h4>{after_html}</div>
    </div>
    <p class="reason"><b>변경 이유:</b> {html.escape(reason_text)}</p>
  </td>
</tr>"""
            )

        previous_frame = frame
        previous_signature = signature
        previous_risk = risk

    return (
        "<table class='data-table change-table'><thead><tr>"
        "<th>날짜</th><th>위험상태</th><th>주요 변화</th><th>평가금액</th>"
        "<th>수수료</th><th>상세</th></tr></thead><tbody>"
        + "".join(rows[-max_events:][::-1])
        + "</tbody></table>"
    )


def _forward_change_rows(
    execution_dir: Path,
    active_dir: Path,
    names: dict[str, str],
) -> str:
    files = sorted(execution_dir.glob("*.csv"))
    if not files:
        return "<p class='muted'>오늘부터 일별 동적 포트폴리오 기록이 쌓입니다.</p>"

    records: list[tuple[str, pd.DataFrame, dict]] = []
    for path in files:
        date_key = path.stem
        plan = pd.read_csv(path)
        active_path = active_dir / f"{date_key}.json"
        active = (
            json.loads(active_path.read_text(encoding="utf-8"))
            if active_path.exists()
            else {}
        )
        records.append((date_key, plan, active))

    rows: list[str] = []
    previous_plan = pd.DataFrame()
    previous_risk = None
    display_index = 0

    for date_key, plan, active in records:
        allocation = active.get("allocation", {})
        risk = str(allocation.get("risk_state", "NORMAL"))
        events = active.get("events") or []
        changed = (
            not records
            or _plan_signature(plan) != _plan_signature(previous_plan)
            or risk != previous_risk
            or bool(events)
        )
        if not changed:
            previous_plan = plan
            previous_risk = risk
            continue

        event_texts = []
        for event in events:
            action = ACTION_LABELS.get(str(event.get("action")), str(event.get("action", "")))
            ticker = str(event.get("ticker") or event.get("in") or "")
            name = names.get(ticker, str(event.get("name") or event.get("in_name") or ticker))
            reason = _translate_reason(event.get("reason"))
            text = f"{action}: {name}"
            if reason:
                text += f" · {reason}"
            event_texts.append(text)

        current_positions = plan.loc[
            (plan["ticker"].astype(str) != "CASH")
            & (pd.to_numeric(plan.get("shares", 0), errors="coerce").fillna(0) > 0)
        ].copy()
        current_positions["종목명"] = current_positions["ticker"].map(
            lambda t: names.get(str(t), str(t))
        )
        current_positions["수량"] = current_positions["shares"].astype(int)
        current_positions["평가금액"] = current_positions["actual_value"].map(
            lambda x: f"{int(round(float(x))):,}원"
        )
        current_positions["비중"] = current_positions["actual_weight"].map(
            lambda x: f"{float(x):.1%}"
        )
        current_table = current_positions[
            ["종목명", "수량", "평가금액", "비중"]
        ]
        cash_rows = plan.loc[plan["ticker"].astype(str) == "CASH"]
        if not cash_rows.empty:
            cash_value = float(cash_rows.iloc[0].get("actual_value", 0.0))
            cash_weight = float(cash_rows.iloc[0].get("actual_weight", 0.0))
            current_table = pd.concat(
                [
                    current_table,
                    pd.DataFrame(
                        [{
                            "종목명": "현금",
                            "수량": "",
                            "평가금액": f"{cash_value:,.0f}원",
                            "비중": f"{cash_weight:.1%}",
                        }]
                    ),
                ],
                ignore_index=True,
            )

        before_html = (
            _html_table(
                previous_plan.loc[
                    previous_plan["ticker"].astype(str) != "CASH"
                ][["name", "shares", "actual_value", "actual_weight"]].rename(
                    columns={
                        "name": "종목명",
                        "shares": "수량",
                        "actual_value": "평가금액",
                        "actual_weight": "비중",
                    }
                )
            )
            if not previous_plan.empty
            else "<p class='muted'>첫 기록</p>"
        )
        detail_id = f"forward-{display_index}"
        display_index += 1
        reason_text = " / ".join(event_texts) if event_texts else "위험상태 또는 목표 정수주 구성이 변경됨"
        rows.append(
            f"""
<tr class="change-row" onclick="toggleRow('{detail_id}')">
  <td>{html.escape(date_key)}</td>
  <td>{html.escape(RISK_STATE_LABELS.get(risk, risk))}</td>
  <td>{html.escape(reason_text)}</td>
  <td>{float(allocation.get('stock_target',0)):.1%}</td>
  <td>{float(allocation.get('safe_target',0)):.1%}</td>
  <td>펼치기</td>
</tr>
<tr id="{detail_id}" class="detail-row" style="display:none">
  <td colspan="6">
    <div class="detail-grid two">
      <div><h4>이전 포트폴리오</h4>{before_html}</div>
      <div><h4>변경 후 동적 포트폴리오</h4>{_html_table(current_table)}</div>
    </div>
    <p class="reason"><b>변경 이유:</b> {html.escape(reason_text)}</p>
  </td>
</tr>"""
        )
        previous_plan = plan
        previous_risk = risk

    return (
        "<table class='data-table change-table'><thead><tr>"
        "<th>날짜</th><th>위험상태</th><th>변경 이유</th><th>주식비중</th>"
        "<th>안전자산/현금</th><th>상세</th></tr></thead><tbody>"
        + "".join(rows[::-1])
        + "</tbody></table>"
    )


def _events_frame(events: list[dict], names: dict[str, str]) -> pd.DataFrame:
    frame = pd.DataFrame(events)
    if frame.empty:
        return frame
    for column in ["ticker", "out", "in"]:
        if column in frame.columns:
            frame[column] = frame[column].map(
                lambda ticker: names.get(str(ticker), str(ticker)) if ticker else ""
            )
    if "action" in frame.columns:
        frame["action"] = frame["action"].map(
            lambda value: ACTION_LABELS.get(str(value), str(value))
        )
    if "reason" in frame.columns:
        frame["reason"] = frame["reason"].map(_translate_reason)
    if "bucket" in frame.columns:
        frame["bucket"] = frame["bucket"].replace(
            {"aggressive": "공격형", "defensive": "방어형"}
        )
    for duplicate in ["name", "out_name", "in_name"]:
        if duplicate in frame.columns:
            frame = frame.drop(columns=[duplicate])
    return frame.rename(
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
            "theme": "테마",
            "defensive_role": "방어 역할",
            "margin": "점수 차이",
        }
    ).fillna("")


def _metric_value(frame: pd.DataFrame, key: str, field: str, default: float = 0.0) -> float:
    if frame.empty or key not in frame.index or field not in frame.columns:
        return default
    try:
        return float(frame.loc[key, field])
    except Exception:
        return default


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

    live_status("[report] 동적 전략/정수주 실행/성과 데이터 통합")
    active = _load_active(active_path)
    snapshot = pd.read_csv(snapshot_path)
    names = _ticker_map(snapshot)
    prices = _price_map(snapshot)

    ideal_weights = {
        str(k): float(v)
        for k, v in active.get("allocation", {}).get("ideal_target_weights", {}).items()
    }
    execution = build_execution_plan(
        ideal_weights,
        prices,
        names,
        capital,
    )
    plan = execution.positions
    cash = execution.cash

    plan_path = Path("runs/latest_execution_plan.csv")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(plan_path, index=False)

    # Persist today's forward dynamic executable portfolio. This is the source
    # for the live walk-forward timeline going forward.
    daily_execution_dir = Path("runs/daily_execution")
    daily_execution_dir.mkdir(parents=True, exist_ok=True)
    daily_plan = plan.copy()
    cash_row = pd.DataFrame(
        [{
            "ticker": "CASH",
            "name": "현금",
            "target_weight": 0.0,
            "target_value": 0.0,
            "price": 1,
            "shares": 0,
            "notional": int(round(cash)),
            "actual_value": int(round(cash)),
            "commission": 0,
            "actual_weight": cash / capital,
            "weight_gap": cash / capital,
            "blocked_by_size": False,
        }]
    )
    pd.concat([daily_plan, cash_row], ignore_index=True).to_csv(
        daily_execution_dir / f"{active.get('as_of')}.csv",
        index=False,
    )

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
    integer_peak_portfolio = _read_csv(
        Path("results_hedge/integer_peak_hedge_portfolio.csv")
    )
    integer_peak_summary = _read_csv(
        Path("results_hedge/integer_peak_hedge_summary.csv"),
        index_col=0,
    )

    trade_log_path = Path("runs/latest_trade_log.csv")
    if not integer_peak_trades.empty:
        integer_peak_trades.to_csv(trade_log_path)
    else:
        pd.DataFrame().to_csv(trade_log_path, index=False)

    print("\n=== 현재 30만원 동적 포트폴리오 ===")
    terminal = plan.loc[:, [
        "name", "target_weight", "price", "shares", "actual_weight",
        "weight_gap", "commission", "blocked_by_size"
    ]].copy()
    terminal["target_weight"] = terminal["target_weight"].map(lambda x: f"{x:.1%}")
    terminal["actual_weight"] = terminal["actual_weight"].map(lambda x: f"{x:.1%}")
    terminal["weight_gap"] = terminal["weight_gap"].map(lambda x: f"{x:+.1%}")
    terminal["price"] = terminal["price"].map(lambda x: f"{int(x):,}")
    terminal["commission"] = terminal["commission"].map(lambda x: f"{int(x):,}")
    print(terminal.to_string(index=False))
    print(
        f"현금 {cash:,.0f}원 ({cash/capital:.1%}) | "
        f"초기 매수 예상 토스 수수료 {execution.total_commission:,}원"
    )

    events = active.get("events") or []
    print("\n=== 이번 동적 종목선정 이벤트 ===")
    if events:
        for event in events:
            print(json.dumps(event, ensure_ascii=False, sort_keys=True))
    else:
        print("(종목 변경 없음)")

    _configure_korean_font()
    eq_img = _equity_chart(
        rotation_curves,
        hedge_curve,
        integer_adaptive_equity,
        integer_peak_equity,
    )
    dd_img = _drawdown_chart(integer_adaptive_equity, integer_peak_equity)
    weights_img = _weights_chart(rotation_weights, names)
    turnover_img = _turnover_chart(hedge_trades)

    plan_html = _format_plan_for_html(plan)
    events_df = _events_frame(events, names)

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
                "누적수익률": f"{float(row.get('total_return',0)):.2%}",
                "연복리수익률": f"{float(row.get('cagr',0)):.2%}",
                "변동성": f"{float(row.get('annual_vol',0)):.2%}",
                "샤프지수": f"{float(row.get('sharpe',0)):.3f}",
                "최대낙폭": f"{float(row.get('max_drawdown',0)):.2%}",
                "매매횟수": int(float(row.get("trade_count",0))),
                "토스 수수료": f"{float(row.get('total_transaction_cost',0)):,.0f}원",
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)

    primary_key = "peak_hedge_whole_shares"
    ending_value = _metric_value(
        integer_peak_summary,
        primary_key,
        "ending_value",
        capital,
    )
    total_return = _metric_value(
        integer_peak_summary,
        primary_key,
        "total_return",
        ending_value / capital - 1.0,
    )
    cagr = _metric_value(integer_peak_summary, primary_key, "cagr")
    sharpe = _metric_value(integer_peak_summary, primary_key, "sharpe")
    mdd = _metric_value(integer_peak_summary, primary_key, "max_drawdown")
    trade_count = _metric_value(integer_peak_summary, primary_key, "trade_count")
    total_fees = _metric_value(
        integer_peak_summary,
        primary_key,
        "total_transaction_cost",
    )
    distributions = _metric_value(
        integer_peak_summary,
        primary_key,
        "total_distributions",
    )

    allocation = active.get("allocation", {})
    risk_state = str(allocation.get("risk_state", "NORMAL"))

    paper_portfolio = _read_csv(
        Path("runs/latest_live_portfolio.csv"),
        index_col=None,
    )
    paper_history = _read_csv(
        Path("runs/paper_account_history.csv"),
        index_col=None,
    )
    paper_timeline = _paper_change_rows(
        Path("runs/paper_portfolios"),
        Path("runs/paper_account_trades.csv"),
        Path("universes/daily_active"),
        names,
    )
    historical_timeline = _historical_change_rows(
        integer_peak_portfolio,
        integer_peak_trades,
        names,
    )

    charts = []
    for title, uri in [
        ("포트폴리오 가치 변화", eq_img),
        ("고점 대비 낙폭", dd_img),
        ("자산 비중 변화", weights_img),
        ("매매 시점과 회전율", turnover_img),
    ]:
        if uri:
            charts.append(
                f"<section class='card'><h2>{html.escape(title)}</h2>"
                f"<img src='{uri}' alt='{html.escape(title)}'></section>"
            )

    css = """
    :root{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR",Segoe UI,sans-serif}
    body{margin:0;background:#f4f6f8;color:#17191c}
    main{max-width:1240px;margin:0 auto;padding:26px}
    h1{margin:0 0 6px} h2{margin:0 0 14px} h4{margin:4px 0 10px}
    .muted{color:#6b7280}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
    .kpi,.card{background:#fff;border:1px solid #e5e7eb;border-radius:13px;padding:16px;margin:12px 0}
    .kpi span{display:block;color:#6b7280;font-size:12px}.kpi b{display:block;font-size:23px;margin-top:5px}
    .section-title{margin-top:28px}.data-table{width:100%;border-collapse:collapse;font-size:13px}
    .data-table th,.data-table td{padding:8px 9px;border-bottom:1px solid #eceff3;text-align:right;vertical-align:top}
    .data-table th:first-child,.data-table td:first-child{text-align:left}
    .scroll{overflow-x:auto} img{width:100%;height:auto}
    code{background:#eef0f3;padding:2px 5px;border-radius:5px}
    .change-row{cursor:pointer}.change-row:hover{background:#f6f8fb}.change-row td:nth-child(3){text-align:left}
    .detail-row td{background:#fafbfc;padding:16px}.detail-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
    .detail-grid.two{grid-template-columns:repeat(2,minmax(0,1fr))}.reason{margin:14px 0 0}
    .badge{display:inline-block;padding:3px 8px;border-radius:999px;background:#eef2f7;font-size:12px}
    @media(max-width:850px){.detail-grid,.detail-grid.two{grid-template-columns:1fr}main{padding:14px}}
    """

    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>리밸런싱 전략 대시보드</title>
<style>{css}</style>
<script>
function toggleRow(id){{
  const row=document.getElementById(id);
  row.style.display=(row.style.display==='none'||!row.style.display)?'table-row':'none';
}}
</script>
</head>
<body><main>
<h1>리밸런싱 전략 대시보드</h1>
<p class="muted">기준일 {html.escape(str(active.get('as_of','')))} · 초기 모의투자금 {capital:,.0f}원 · 토스증권 KRX ETF 수수료 0.015% · ETF 증권거래세 0원</p>

<h2 class="section-title">현재 동적 전략 상태</h2>
<div class="grid">
  <div class="kpi"><span>위험 상태</span><b>{html.escape(RISK_STATE_LABELS.get(risk_state,risk_state))}</b></div>
  <div class="kpi"><span>시장 점수</span><b>{float(allocation.get('market_score',0)):.3f}</b></div>
  <div class="kpi"><span>주식 목표</span><b>{float(allocation.get('stock_target',0)):.1%}</b></div>
  <div class="kpi"><span>방어자산 목표</span><b>{float(allocation.get('defensive_target',0)):.1%}</b></div>
  <div class="kpi"><span>안전자산 목표</span><b>{float(allocation.get('safe_target',0)):.1%}</b></div>
  <div class="kpi"><span>정수주 후 현금</span><b>{cash:,.0f}원</b></div>
</div>

<h2 class="section-title">30만원 역사적 검증 핵심지표</h2>
<p class="muted">아래 성과는 현재 동적 ETF universe가 아니라, 날짜가 충분히 확보된 피크 보호+헤지 고정 universe의 정수주 검증값이다. 동적 전략은 오늘부터 일별 snapshot으로 forward 검증한다.</p>
<div class="grid">
  <div class="kpi"><span>최종 평가금액</span><b>{ending_value:,.0f}원</b></div>
  <div class="kpi"><span>누적 수익률</span><b>{total_return:.1%}</b></div>
  <div class="kpi"><span>연복리 수익률</span><b>{cagr:.1%}</b></div>
  <div class="kpi"><span>샤프지수</span><b>{sharpe:.3f}</b></div>
  <div class="kpi"><span>최대낙폭</span><b>{mdd:.1%}</b></div>
  <div class="kpi"><span>매매 횟수</span><b>{trade_count:,.0f}</b></div>
  <div class="kpi"><span>누적 토스 수수료</span><b>{total_fees:,.0f}원</b></div>
  <div class="kpi"><span>누적 분배금</span><b>{distributions:,.0f}원</b></div>
</div>

<section class="card">
<h2>현재 30만원 forward paper 포트폴리오</h2>
<p class="muted">전날 계좌를 이어받아 오늘 목표와의 차이만 매매한 실제 paper 계좌 상태다. 단순히 매일 30만원을 새로 배분한 표가 아니다.</p>
<div class="scroll">{_html_table(_paper_portfolio_table(paper_portfolio))}</div>
</section>

<section class="card">
<h2>동적 전략 변경일 타임라인</h2>
<p class="muted">실제 paper 계좌의 보유수량, 위험상태 또는 매매가 바뀐 날짜만 표시한다. 행을 누르면 변경 전 → 그날 매매 → 변경 후를 볼 수 있다.</p>
<div class="scroll">{paper_timeline}</div>
</section>

<section class="card">
<h2>오늘 목표 정수주 계산안</h2>
<p class="muted">현재 동적 목표비중을 1주 단위로 새로 환산한 참고표다. 실제 forward 계좌는 위 paper 포트폴리오를 기준으로 이어진다.</p>
<div class="scroll">{_html_table(plan_html)}</div>
<p><b>새 계좌 가정 잔여 현금:</b> {cash:,.0f}원 ({cash/capital:.1%}) · <b>초기 주문 예상 토스 수수료:</b> {execution.total_commission:,}원</p>
</section>

<section class="card">
<h2>전략 성과 비교</h2>
<div class="scroll">{_html_table(metrics_df)}</div>
</section>

{''.join(charts)}

<section class="card">
<h2>과거 변경일별 포트폴리오 재생</h2>
<p class="muted">피크 보호+헤지 정수주 검증에서 실제 매매가 발생한 날짜만 표시한다. 날짜 행을 누르면 변경 전 포트폴리오 → 실제 매매 → 변경 후 포트폴리오가 열린다.</p>
<div class="scroll">{historical_timeline}</div>
</section>

<section class="card">
<h2>이번 동적 종목선정 이벤트</h2>
<div class="scroll">{_html_table(events_df)}</div>
</section>

<section class="card">
<h2>현재 엔진 상태</h2>
<p>시장 비상상태: <code>{_ko_bool(active.get('market_emergency'))}</code> ·
피크 보호: <code>{_ko_bool(allocation.get('peak_lock'))}</code> ·
60일 고점 대비 낙폭: <code>{float(allocation.get('peak_drawdown_60d',0)):.1%}</code> ·
20일 수익률: <code>{float(allocation.get('return_20d',0)):.1%}</code></p>
<p class="muted">신호 계산은 배당 반영 조정가격, 정수주 체결 replay는 실제 Close와 분배금 현금흐름을 사용한다. 현재 동적 universe의 과거 성과는 날짜별 universe가 충분히 쌓이기 전까지 소급 검증값으로 표시하지 않는다.</p>
</section>
</main></body></html>"""

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")
    finish_status(
        f"[report] 완료 | dashboard={output} | plan={plan_path} | "
        f"risk_state={risk_state} | cash={cash:,.0f}원"
    )


if __name__ == "__main__":
    main()
