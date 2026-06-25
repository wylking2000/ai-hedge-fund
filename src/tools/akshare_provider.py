"""AKShare-backed data provider (mainland-China accessible).

Drop-in replacement for the financialdatasets.ai calls used in ``src/tools/api.py``.
Returns the project's existing Pydantic models so no downstream agent needs changes.

Notes / known limitations (see api.py for how they're surfaced):
  * Prices, financial metrics, line items and market cap are supported for US tickers.
  * Insider trades and US company news have no AKShare source -> handled as empty in api.py.
  * Many financialdatasets.ai ratios are not provided by AKShare; we compute what we
    can from the statements and leave the rest as ``None`` (all model fields are Optional).
  * Eastmoney/Baidu endpoints are reachable from mainland China. Some may be blocked in
    sandboxed/overseas networks; that is an environment limitation, not a code bug.
"""

from __future__ import annotations

import datetime
import logging

import akshare as ak
import pandas as pd

from src.data.models import (
    FinancialMetrics,
    LineItem,
    Price,
)

logger = logging.getLogger(__name__)

# Eastmoney market-id prefixes used by stock_us_hist (e.g. "105.AAPL").
# 105 = NASDAQ, 106 = NYSE, 107 = AMEX. We probe in this order as a fallback.
_US_MARKET_PREFIXES = ("105", "106", "107")
_symbol_prefix_cache: dict[str, str] = {}


def _to_float(value) -> float | None:
    """Coerce AKShare cell values (str/float/NaN) to float or None."""
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
            if value in ("", "-", "--", "None", "nan"):
                return None
        f = float(value)
        if pd.isna(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _ak_date(d: str) -> str:
    """Convert 'YYYY-MM-DD' -> 'YYYYMMDD' for AKShare."""
    return d.replace("-", "")


def _resolve_us_symbol(ticker: str) -> str:
    """Resolve a plain ticker (AAPL) to AKShare's 'prefix.TICKER' eastmoney format.

    Tries the eastmoney spot list once (authoritative), then falls back to probing
    the common market-id prefixes by attempting a tiny history pull.
    """
    t = ticker.upper()
    if t in _symbol_prefix_cache:
        return _symbol_prefix_cache[t]

    # Authoritative lookup via the US spot table.
    try:
        spot = ak.stock_us_spot_em()
        # '代码' values look like '105.AAPL'; match the suffix after the dot.
        match = spot[spot["代码"].str.upper().str.endswith("." + t, na=False)]
        if not match.empty:
            code = str(match.iloc[0]["代码"])
            _symbol_prefix_cache[t] = code
            return code
    except Exception as e:  # noqa: BLE001 - network/format issues fall through to probe
        logger.debug("stock_us_spot_em lookup failed for %s: %s", t, e)

    # Fallback: probe prefixes with a minimal request.
    for prefix in _US_MARKET_PREFIXES:
        candidate = f"{prefix}.{t}"
        try:
            df = ak.stock_us_hist(
                symbol=candidate, period="daily",
                start_date="20240101", end_date="20240105", adjust="",
            )
            if df is not None and not df.empty:
                _symbol_prefix_cache[t] = candidate
                return candidate
        except Exception:  # noqa: BLE001
            continue

    # Last resort: assume NASDAQ prefix; caller handles empty results.
    fallback = f"105.{t}"
    _symbol_prefix_cache[t] = fallback
    return fallback


# --------------------------------------------------------------------------- prices
# Column maps for the two US daily sources. Each maps the source's columns to
# (open, high, low, close, volume, date).
_EM_COLS = ("开盘", "最高", "最低", "收盘", "成交量", "日期")
_SINA_COLS = ("open", "high", "low", "close", "volume", "date")


def _rows_to_prices(df: pd.DataFrame, cols, start_date: str, end_date: str) -> list[Price]:
    o, h, l, c, v, d = cols
    prices: list[Price] = []
    for _, row in df.iterrows():
        time = str(row[d])[:10]
        if time < start_date or time > end_date:
            continue
        try:
            prices.append(
                Price(
                    open=_to_float(row[o]) or 0.0,
                    high=_to_float(row[h]) or 0.0,
                    low=_to_float(row[l]) or 0.0,
                    close=_to_float(row[c]) or 0.0,
                    volume=int(_to_float(row[v]) or 0),
                    time=time,
                )
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Skipping bad price row: %s", e)
    return prices


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """US daily prices. Tries Eastmoney first, falls back to Sina (both CN-accessible)."""
    # Primary: Eastmoney (server-side date filtering).
    symbol = _resolve_us_symbol(ticker)
    try:
        df = ak.stock_us_hist(
            symbol=symbol, period="daily",
            start_date=_ak_date(start_date), end_date=_ak_date(end_date), adjust="qfq",
        )
        if df is not None and not df.empty:
            prices = _rows_to_prices(df, _EM_COLS, start_date, end_date)
            if prices:
                return prices
    except Exception as e:  # noqa: BLE001
        logger.warning("Eastmoney price fetch failed for %s (%s); trying Sina: %s", ticker, symbol, e)

    # Fallback: Sina (returns full history; filter client-side).
    try:
        df = ak.stock_us_daily(symbol=ticker.upper(), adjust="qfq")
    except Exception as e:  # noqa: BLE001
        logger.warning("Sina price fetch also failed for %s: %s", ticker, e)
        return []
    if df is None or df.empty:
        return []
    return _rows_to_prices(df, _SINA_COLS, start_date, end_date)


# ------------------------------------------------------------------- statement load
_INDICATOR_BY_PERIOD = {"annual": "年报", "ttm": "年报", "quarterly": "单季报"}


def _load_reports(ticker: str, indicator: str) -> dict[str, dict[str, float]]:
    """Return {report_date(YYYY-MM-DD): {chinese_item_name: amount}} merged across
    the three US financial statements."""
    merged: dict[str, dict[str, float]] = {}
    for stmt in ("资产负债表", "综合损益表", "现金流量表"):
        try:
            df = ak.stock_financial_us_report_em(stock=ticker.upper(), symbol=stmt, indicator=indicator)
        except Exception as e:  # noqa: BLE001
            logger.debug("report %s/%s failed: %s", ticker, stmt, e)
            continue
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            period = str(row.get("REPORT_DATE", ""))[:10]
            name = row.get("ITEM_NAME")
            if not period or not name:
                continue
            merged.setdefault(period, {})[str(name)] = _to_float(row.get("AMOUNT"))
    return merged


# --------------------------------------------------------------- financial metrics
def get_financial_metrics(ticker: str, end_date: str, period: str, limit: int) -> list[FinancialMetrics]:
    indicator = _INDICATOR_BY_PERIOD.get(period, "年报")
    try:
        df = ak.stock_financial_us_analysis_indicator_em(symbol=ticker.upper(), indicator=indicator)
    except Exception as e:  # noqa: BLE001
        logger.warning("AKShare indicator fetch failed for %s: %s", ticker, e)
        return []
    if df is None or df.empty:
        return []

    reports = _load_reports(ticker, indicator)  # for ratios AKShare doesn't expose directly
    df = df.sort_values("REPORT_DATE", ascending=False)
    currency = str(df.iloc[0].get("CURRENCY_ABBR") or "USD")

    out: list[FinancialMetrics] = []
    for _, r in df.iterrows():
        report_period = str(r.get("REPORT_DATE", ""))[:10]
        if not report_period or report_period > end_date:
            continue
        stmt = reports.get(report_period, {})

        revenue = stmt.get("营业收入")
        operating_income = stmt.get("营业利润")
        operating_margin = (operating_income / revenue) if (operating_income and revenue) else None

        out.append(
            FinancialMetrics(
                ticker=ticker.upper(),
                report_period=report_period,
                period=period,
                currency=currency,
                market_cap=None,
                enterprise_value=None,
                price_to_earnings_ratio=None,
                price_to_book_ratio=None,
                price_to_sales_ratio=None,
                enterprise_value_to_ebitda_ratio=None,
                enterprise_value_to_revenue_ratio=None,
                free_cash_flow_yield=None,
                peg_ratio=None,
                gross_margin=_pct(r.get("GROSS_PROFIT_RATIO")),
                operating_margin=operating_margin,
                net_margin=_pct(r.get("NET_PROFIT_RATIO")),
                return_on_equity=_pct(r.get("ROE_AVG")),
                return_on_assets=_pct(r.get("ROA")),
                return_on_invested_capital=None,
                asset_turnover=_to_float(r.get("TOTAL_ASSETS_TR")),
                inventory_turnover=_to_float(r.get("INVENTORY_TR")),
                receivables_turnover=_to_float(r.get("ACCOUNTS_RECE_TR")),
                days_sales_outstanding=_to_float(r.get("ACCOUNTS_RECE_TDAYS")),
                operating_cycle=None,
                working_capital_turnover=None,
                current_ratio=_to_float(r.get("CURRENT_RATIO")),
                quick_ratio=_to_float(r.get("SPEED_RATIO")),
                cash_ratio=None,
                operating_cash_flow_ratio=_to_float(r.get("OCF_LIQDEBT")),
                debt_to_equity=_pct(r.get("EQUITY_RATIO")),
                debt_to_assets=_pct(r.get("DEBT_ASSET_RATIO")),
                interest_coverage=None,
                revenue_growth=_pct(r.get("OPERATE_INCOME_YOY")),
                earnings_growth=_pct(r.get("PARENT_HOLDER_NETPROFIT_YOY")),
                book_value_growth=None,
                earnings_per_share_growth=_pct(r.get("BASIC_EPS_YOY")),
                free_cash_flow_growth=None,
                operating_income_growth=None,
                ebitda_growth=None,
                payout_ratio=None,
                earnings_per_share=_to_float(r.get("BASIC_EPS")),
                book_value_per_share=None,
                free_cash_flow_per_share=None,
            )
        )
        if len(out) >= limit:
            break
    return out


def _pct(value) -> float | None:
    """AKShare ratio/growth columns are in percent units; convert to fraction."""
    f = _to_float(value)
    return f / 100.0 if f is not None else None


# ------------------------------------------------------------------------ line items
# Maps the project's requested line_item -> a function over one period's statement dict.
def _li_total_debt(s: dict) -> float | None:
    short = s.get("短期债务")
    long = s.get("长期负债")
    parts = [p for p in (short, long) if p is not None]
    return sum(parts) if parts else None


def _li_fcf(s: dict) -> float | None:
    ocf = s.get("经营活动产生的现金流量净额")
    capex = s.get("购买固定资产")
    if ocf is None:
        return None
    return ocf - abs(capex) if capex is not None else ocf


def _li_working_capital(s: dict) -> float | None:
    ca, cl = s.get("流动资产合计"), s.get("流动负债合计")
    return (ca - cl) if (ca is not None and cl is not None) else None


def _li_ebitda(s: dict) -> float | None:
    op, da = s.get("营业利润"), s.get("折旧及摊销")
    if op is None:
        return None
    return op + da if da is not None else op


def _li_goodwill_intangibles(s: dict) -> float | None:
    g, i = s.get("商誉"), s.get("无形资产")
    parts = [p for p in (g, i) if p is not None]
    return sum(parts) if parts else None


def _li_equity_share_activity(s: dict) -> float | None:
    issue, buyback = s.get("发行股份"), s.get("回购股份")
    parts = [p for p in (issue, buyback) if p is not None]
    return sum(parts) if parts else None


# direct name lookups
_LINE_ITEM_MAP = {
    "revenue": lambda s: s.get("营业收入"),
    "gross_profit": lambda s: s.get("毛利"),
    "operating_income": lambda s: s.get("营业利润"),
    "operating_expense": lambda s: s.get("营业费用"),
    "net_income": lambda s: s.get("净利润"),
    "research_and_development": lambda s: s.get("研发费用"),
    "interest_expense": lambda s: s.get("利息收入"),  # closest available item
    "earnings_per_share": lambda s: s.get("基本每股收益-普通股"),
    "depreciation_and_amortization": lambda s: s.get("折旧及摊销"),
    "capital_expenditure": lambda s: s.get("购买固定资产"),
    "dividends_and_other_cash_distributions": lambda s: s.get("股息支付"),
    "outstanding_shares": lambda s: s.get("摊薄加权平均股数-普通股") or s.get("基本加权平均股数-普通股"),
    "cash_and_equivalents": lambda s: s.get("现金及现金等价物"),
    "total_assets": lambda s: s.get("总资产"),
    "total_liabilities": lambda s: s.get("总负债"),
    "current_assets": lambda s: s.get("流动资产合计"),
    "current_liabilities": lambda s: s.get("流动负债合计"),
    "shareholders_equity": lambda s: s.get("股东权益合计") or s.get("归属于母公司股东权益"),
    "intangible_assets": lambda s: s.get("无形资产"),
    "gross_profit_": lambda s: s.get("毛利"),
    "ebit": lambda s: s.get("持续经营税前利润") or s.get("营业利润"),
    # computed
    "total_debt": _li_total_debt,
    "free_cash_flow": _li_fcf,
    "working_capital": _li_working_capital,
    "ebitda": _li_ebitda,
    "goodwill_and_intangible_assets": _li_goodwill_intangibles,
    "issuance_or_purchase_of_equity_shares": _li_equity_share_activity,
}


def get_line_items(ticker: str, line_items: list[str], end_date: str, period: str, limit: int) -> list[LineItem]:
    indicator = _INDICATOR_BY_PERIOD.get(period, "年报")
    reports = _load_reports(ticker, indicator)
    if not reports:
        return []

    # ratios that some agents request via line items (derive per period)
    def margins(s: dict) -> dict:
        rev, op, gp = s.get("营业收入"), s.get("营业利润"), s.get("毛利")
        eq, ni = s.get("股东权益合计"), s.get("净利润")
        shares = s.get("摊薄加权平均股数-普通股") or s.get("基本加权平均股数-普通股")
        return {
            "gross_margin": (gp / rev) if (gp and rev) else None,
            "operating_margin": (op / rev) if (op and rev) else None,
            "debt_to_equity": (s.get("总负债") / eq) if (s.get("总负债") and eq) else None,
            "return_on_invested_capital": (ni / eq) if (ni and eq) else None,
            "book_value_per_share": (eq / shares) if (eq and shares) else None,
        }

    currency = "USD"
    periods = sorted((p for p in reports if p <= end_date), reverse=True)[:limit]
    results: list[LineItem] = []
    for p in periods:
        s = reports[p]
        derived = margins(s)
        payload: dict = {
            "ticker": ticker.upper(),
            "report_period": p,
            "period": period,
            "currency": currency,
        }
        for li in line_items:
            if li in _LINE_ITEM_MAP:
                payload[li] = _LINE_ITEM_MAP[li](s)
            elif li in derived:
                payload[li] = derived[li]
            else:
                payload[li] = None
        results.append(LineItem(**payload))
    return results


# ------------------------------------------------------------------------ market cap
def get_market_cap(ticker: str, end_date: str) -> float | None:
    """Market cap = latest close (<= end_date) * most recent diluted share count.

    NOTE: We deliberately avoid ``stock_us_valuation_baidu`` — it relies on mini-racer
    (an embedded V8), which fatally crashes when called from the forked agent worker
    processes on macOS. Computing from price * shares is both safe and source-light.
    """
    # Most recent diluted (or basic) weighted-average shares from the income statement.
    reports = _load_reports(ticker, "年报")
    shares = None
    if reports:
        periods = sorted((p for p in reports if p <= end_date), reverse=True)
        for p in periods:
            s = reports[p]
            shares = s.get("摊薄加权平均股数-普通股") or s.get("基本加权平均股数-普通股")
            if shares:
                break
    if not shares:
        return None

    # Latest close on/just before end_date — pull a short recent window, not full history.
    try:
        start = (datetime.date.fromisoformat(end_date) - datetime.timedelta(days=14)).isoformat()
    except ValueError:
        start = end_date
    prices = get_prices(ticker, start, end_date)
    if not prices:
        return None
    return prices[-1].close * shares
