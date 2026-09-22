import pandas as pd

from rebalance_backtest.universe_discovery import (
    _classify_name,
    _prefilter,
)


def test_classifies_leverage_inverse_as_excluded():
    assert _classify_name("KODEX 레버리지") == "EXCLUDED"
    assert _classify_name("TIGER 200선물인버스2X") == "EXCLUDED"


def test_classifies_defensive_and_real_assets():
    assert _classify_name("KODEX 미국달러선물") == "DEFENSIVE"
    assert _classify_name("KODEX 국고채3년") == "DEFENSIVE"
    assert _classify_name("KODEX 골드선물(H)") == "REAL_ASSET"
    assert _classify_name("KODEX 반도체") == "EQUITY"


def test_prefilter_excludes_leverage_and_keeps_defensive_coverage():
    listing = pd.DataFrame(
        [
            {
                "symbol": "1",
                "name": "Core Equity",
                "bucket": "EQUITY",
                "market_cap_raw": 1000,
                "volume": 1000,
                "return_3m_listing": 2.0,
            },
            {
                "symbol": "2",
                "name": "Fast Equity",
                "bucket": "EQUITY",
                "market_cap_raw": 500,
                "volume": 800,
                "return_3m_listing": 20.0,
            },
            {
                "symbol": "3",
                "name": "Gold",
                "bucket": "REAL_ASSET",
                "market_cap_raw": 200,
                "volume": 300,
                "return_3m_listing": 1.0,
            },
            {
                "symbol": "4",
                "name": "Leveraged",
                "bucket": "EXCLUDED",
                "market_cap_raw": 5000,
                "volume": 5000,
                "return_3m_listing": 50.0,
            },
        ]
    )

    result = _prefilter(listing, count=20)

    assert "4" not in result["symbol"].tolist()
    assert "3" in result["symbol"].tolist()
