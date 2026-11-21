"""
Indian Long-Only Deep Value Strategy
Based on: Declerck (2020), "Deep Value Portfolios: a Long Only Implementation" (SSRN 3750134)

Paper summary:
- Build expected returns per asset and convert them to rolling 10-year percentiles.
- "Deep value" = high percentile (cheap) assets.
- Use three thresholds with hysteresis:
    IN         : enter when percentile >= IN
    OUT        : exit when percentile <= OUT
    ValueTrap  : exit when percentile >= ValueTrap (asset keeps getting cheaper)
- Rebalance daily at close, equal weight within asset class.

Indian market simplification:
- Universe: NIFTY 100 stocks (current constituents).
- Value proxy: composite score from earnings yield, book-to-market, dividend yield.
- Rolling percentile lookback: 5 years (instead of 10, to preserve data).
- Rebalancing: quarterly (instead of daily, to keep turnover realistic).
- Costs: 0.10% one-way.

Caveat: yfinance fundamentals are the latest reported, not truly point-in-time.
This means the backtest assumes current book/earnings/dividends were known historically.
Use results as an upper-bound estimate only.
"""

import os
import warnings
from datetime import date, datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.4f}".format

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
START_DATE = "2010-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")
IS_START = "2015-01-01"
IS_END = "2019-12-31"
OOS_START = "2020-01-01"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Deep value thresholds (paper suggested ranges: IN 80-90%, OUT 40-50%, VT 85-95%)
IN_THRESHOLD = 0.80
OUT_THRESHOLD = 0.50
VALUETRAP_THRESHOLD = 0.90

# Minimum history for percentile calculation (years)
PERCENTILE_LOOKBACK_YEARS = 5
MIN_HISTORY_YEARS = 3

# Rebalancing frequency
REBALANCE_FREQ = "QE"  # quarter-end

# Transaction costs
COST_BPS = 0.10  # one-way %
COST = COST_BPS / 100.0

# Capital
INITIAL_CAPITAL = 1_000_000

# NIFTY 100 tickers (current constituents; yfinance .NS suffix)
NIFTY100_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BANDHANBNK.NS", "BANKBARODA.NS",
    "BERGEPAINT.NS", "BEL.NS", "BPCL.NS", "BHARTIARTL.NS", "BOSCHLTD.NS",
    "BRITANNIA.NS", "CANBK.NS", "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS",
    "COLPAL.NS", "DABUR.NS", "DIVISLAB.NS", "DLF.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GAIL.NS", "GRASIM.NS", "HAL.NS", "HCLTECH.NS",
    "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "ITC.NS", "INDUSINDBK.NS",
    "INFY.NS", "IOC.NS", "IRCTC.NS", "IRFC.NS", "JSWSTEEL.NS",
    "JINDALSTEL.NS", "JIOFIN.NS", "KOTAKBANK.NS", "LT.NS", "LTIM.NS",
    "M&M.NS", "MARICO.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS",
    "ONGC.NS", "POWERGRID.NS", "PIDILITIND.NS", "PNB.NS", "RECLTD.NS",
    "RELIANCE.NS", "SAIL.NS", "SBIN.NS", "SBILIFE.NS", "SHREECEM.NS",
    "SIEMENS.NS", "SRF.NS", "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS",
    "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "ULTRACEMCO.NS", "UPL.NS",
    "VEDL.NS", "WIPRO.NS", "ZOMATO.NS", "ZYDUSLIFE.NS",
    # Extended list to cover ~100
    "ABB.NS", "ACC.NS", "ALKEM.NS", "AMBUJACEM.NS", "ASTRAL.NS",
    "AUROPHARMA.NS", "BALKRISIND.NS", "BANKINDIA.NS", "BATAINDIA.NS", "BHEL.NS",
    "BIOCON.NS", "CADILAHC.NS", "CASTROLIND.NS", "CUMMINSIND.NS", "ESCORTS.NS",
    "EXIDEIND.NS", "FEDERALBNK.NS", "GLAND.NS", "GODREJCP.NS", "GODREJPROP.NS",
    "HAVELLS.NS", "HINDPETRO.NS", "HINDZINC.NS", "IBULHSGFIN.NS", "IDFCFIRSTB.NS",
    "IGL.NS", "INDHOTEL.NS", "INDIGO.NS", "IPCALAB.NS", "JUBLFOOD.NS",
    "LICHSGFIN.NS", "LUPIN.NS", "M&MFIN.NS", "MANAPPURAM.NS", "MFSL.NS",
    "MGL.NS", "MOTHERSON.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NAUKRI.NS",
    "NAVINFLUOR.NS", "PAGEIND.NS", "PAYTM.NS", "PEL.NS", "PFC.NS",
    "PGHH.NS", "PIIND.NS", "POLYCAB.NS", "RAMCOCEM.NS", "RBLBANK.NS",
    "SCHAEFFLER.NS", "SKFINDIA.NS", "SRTRANSFIN.NS", "SUMICHEM.NS", "SYNGENE.NS",
    "TATACHEM.NS", "TATACOMM.NS", "TATAELXSI.NS", "TIINDIA.NS", "UNIONBANK.NS",
    "VBL.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "YESBANK.NS",
]


# ---------------------------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------------------------
def cache_path(ticker, suffix="prices"):
    safe = ticker.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{suffix}.csv")


def download_prices(ticker, start, end):
    path = cache_path(ticker, "prices")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        # Refresh if too stale
        if df.index[-1].strftime("%Y-%m-%d") >= (date.today() - timedelta(days=5)).strftime("%Y-%m-%d"):
            return df
    print(f"    Downloading {ticker} ...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.to_csv(path)
    return df


def get_fundamentals(ticker, price_series):
    """
    Fetch latest fundamentals from yfinance and anchor them to a current price.
    Returns estimated EPS, Book value per share, and annual dividend based on
    the latest reported ratios. These act as constant fundamental anchors so
    the value metric varies with price history.
    """
    path = cache_path(ticker, "info")
    if os.path.exists(path):
        info = pd.read_csv(path).to_dict("records")[0]
    else:
        try:
            info = yf.Ticker(ticker).info
        except Exception as e:
            print(f"      Could not fetch info for {ticker}: {e}")
            info = {}
        pd.DataFrame([info]).to_csv(path, index=False)

    if price_series.empty:
        return None

    current_price = float(price_series.iloc[-1])

    pe = info.get("trailingPE") or info.get("forwardPE")
    pb = info.get("priceToBook")
    div_yield = info.get("dividendYield")
    if div_yield is not None and div_yield > 1.0:
        div_yield = div_yield / 100.0

    eps = current_price / pe if pe and pe > 0 else None
    book_per_share = current_price / pb if pb and pb > 0 else None
    dividend = current_price * div_yield if div_yield is not None else 0.0

    # Need at least one value anchor to compute a value score
    if eps is None and book_per_share is None and dividend == 0:
        return None

    return {
        "current_price": current_price,
        "eps": eps,
        "book_per_share": book_per_share,
        "dividend": dividend,
    }


# ---------------------------------------------------------------------------
# VALUE SIGNAL
# ---------------------------------------------------------------------------
def build_value_history(price_data, fundamentals):
    """
    Build a daily panel of per-stock historical value-score percentiles.

    Because yfinance does not give historical fundamentals, we anchor to the
    latest reported EPS, book value, and dividend. We then reconstruct
    historical earnings yield, book-to-market, and dividend yield using the
    historical price series:
        earnings_yield = EPS / Price
        book_to_market = Book / Price
        dividend_yield = Dividend / Price

    For each stock we form a composite value score (z-scored components) and
    compute its rolling percentile over its own history. This matches the paper's
    idea that an asset is "deep value" when its expected return / valuation is
    in the extreme upper tail of its own history.
    """
    all_dates = pd.date_range(
        start=min(df.index.min() for df in price_data.values() if not df.empty),
        end=max(df.index.max() for df in price_data.values() if not df.empty),
        freq="D",
    )

    closes = pd.DataFrame({t: df["Close"] for t, df in price_data.items()}).reindex(all_dates)

    # Build daily value-metric panels
    ey_panel = pd.DataFrame(index=all_dates, columns=fundamentals.keys(), dtype=float)
    bm_panel = pd.DataFrame(index=all_dates, columns=fundamentals.keys(), dtype=float)
    dy_panel = pd.DataFrame(index=all_dates, columns=fundamentals.keys(), dtype=float)

    for ticker, f in fundamentals.items():
        if f is None:
            continue
        price = closes[ticker]
        if f["eps"] is not None:
            ey_panel[ticker] = f["eps"] / price
        if f["book_per_share"] is not None:
            bm_panel[ticker] = f["book_per_share"] / price
        if f["dividend"] is not None and f["dividend"] > 0:
            dy_panel[ticker] = f["dividend"] / price

    # Cross-sectionally z-score each metric each day, then average available metrics
    value_score_panel = pd.DataFrame(index=all_dates, columns=fundamentals.keys(), dtype=float)
    for dt in all_dates:
        row_z = []
        for panel in [ey_panel, bm_panel, dy_panel]:
            vals = panel.loc[dt].dropna()
            if len(vals) < 5:
                continue
            mean = vals.mean()
            std = vals.std()
            if std <= 0:
                continue
            z = (panel.loc[dt] - mean) / std
            row_z.append(z)
        if row_z:
            value_score_panel.loc[dt] = pd.concat(row_z, axis=1).mean(axis=1, skipna=True)

    # Per-stock rolling percentile of its own value score
    min_periods = int(252 * MIN_HISTORY_YEARS)
    window = int(252 * PERCENTILE_LOOKBACK_YEARS)

    percentile_panel = value_score_panel.rolling(
        window=window, min_periods=min_periods
    ).rank(pct=True)
    return percentile_panel


# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------
def run_backtest(price_data, percentile_panel, rebalance_dates, initial_capital=INITIAL_CAPITAL):
    """
    Long-only equal-weight deep value portfolio with hysteresis.
    State machine per stock:
      - not held: enter if pct >= IN
      - held    : exit if pct <= OUT  OR  pct >= ValueTrap
    Equal weight across held stocks at each rebalance.
    Execution at open on rebalance days using signals from previous close.
    Portfolio is marked-to-market every trading day.
    """
    opens = pd.DataFrame({t: df["Open"] for t, df in price_data.items()}).sort_index()
    closes = pd.DataFrame({t: df["Close"] for t, df in price_data.items()}).sort_index()
    percentile_panel = percentile_panel.reindex(closes.index)

    rebalance_set = set(rebalance_dates)

    held = set()
    capital = float(initial_capital)
    holdings = {}  # ticker -> shares
    equity_curve = []
    trades = []

    for i, today in enumerate(closes.index):
        if today in rebalance_set and i > 0:
            # Signal known at previous close
            signal_date = closes.index[i - 1]
            pct_row = percentile_panel.loc[signal_date]

            new_held = set()
            for ticker in pct_row.index:
                p = pct_row[ticker]
                if pd.isna(p):
                    continue
                if ticker in held:
                    if OUT_THRESHOLD < p < VALUETRAP_THRESHOLD:
                        new_held.add(ticker)
                else:
                    if p >= IN_THRESHOLD:
                        new_held.add(ticker)

            open_prices = opens.loc[today]

            target_tickers = [t for t in new_held if not pd.isna(open_prices.get(t))]

            # Sell positions that are no longer in target
            for ticker in list(holdings.keys()):
                if ticker not in target_tickers:
                    price = open_prices.get(ticker)
                    if not pd.isna(price):
                        capital += holdings[ticker] * price * (1 - COST)
                        trades.append({"date": today, "ticker": ticker, "action": "SELL", "price": price})
                    del holdings[ticker]

            # Buy only newly selected target positions (existing positions drift)
            new_entries = [t for t in target_tickers if t not in holdings]
            if new_entries:
                alloc_per_stock = capital / len(new_entries)
                for ticker in new_entries:
                    price = open_prices[ticker]
                    if pd.isna(price):
                        continue
                    shares = alloc_per_stock * (1 - COST) / price
                    holdings[ticker] = shares
                    capital -= alloc_per_stock
                    trades.append({"date": today, "ticker": ticker, "action": "BUY", "price": price})

            held = set(holdings.keys())

        # Mark-to-market at close
        close_prices = closes.loc[today]
        port_value = capital
        for ticker, shares in holdings.items():
            price = close_prices.get(ticker)
            if not pd.isna(price):
                port_value += shares * price
        equity_curve.append({"date": today, "equity": port_value})

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    return equity_df, trades


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def compute_metrics(equity_df, period_name):
    if equity_df.empty or len(equity_df) < 2:
        return {}
    rets = equity_df["equity"].pct_change().dropna()
    if rets.empty:
        return {}

    total_ret = equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1
    years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan

    # Max drawdown
    peak = equity_df["equity"].cummax()
    dd = (equity_df["equity"] - peak) / peak
    max_dd = dd.min()

    # Sharpe (quarterly rebalanced, but we'll use daily returns annualised)
    ann_vol = rets.std() * np.sqrt(252)
    ann_ret = rets.mean() * 252
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    return {
        "total_return": total_ret,
        "cagr": cagr,
        "max_dd": max_dd,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "avg_daily_ret": rets.mean(),
    }


def compute_benchmark_metrics(bench, start, end):
    bench = bench.loc[start:end].copy()
    if bench.empty:
        return {}
    rets = bench["Close"].pct_change().dropna()
    total_ret = bench["Close"].iloc[-1] / bench["Close"].iloc[0] - 1
    years = (bench.index[-1] - bench.index[0]).days / 365.25
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan
    peak = bench["Close"].cummax()
    dd = (bench["Close"] - peak) / peak
    max_dd = dd.min()
    ann_vol = rets.std() * np.sqrt(252)
    ann_ret = rets.mean() * 252
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return {
        "total_return": total_ret,
        "cagr": cagr,
        "max_dd": max_dd,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Indian Long-Only Deep Value Strategy")
    print("=" * 70)
    print(f"Full period: {START_DATE} to {END_DATE}")
    print(f"In-Sample : {IS_START} to {IS_END}")
    print(f"Out-of-Sample : {OOS_START} to {END_DATE}")
    print(f"Universe: NIFTY 100 stocks")
    print(f"Thresholds: IN={IN_THRESHOLD:.0%}, OUT={OUT_THRESHOLD:.0%}, ValueTrap={VALUETRAP_THRESHOLD:.0%}")
    print(f"Rebalancing: {REBALANCE_FREQ}")
    print(f"Cost: {COST_BPS:.2f}% one-way")
    print("=" * 70)

    # Download stock prices
    print("\nDownloading stock data...")
    price_data = {}
    valid_tickers = []
    for i, ticker in enumerate(NIFTY100_TICKERS, 1):
        print(f"  [{i:3d}/{len(NIFTY100_TICKERS)}] {ticker}")
        try:
            df = download_prices(ticker, START_DATE, END_DATE)
            if len(df) > 100:
                price_data[ticker] = df
                valid_tickers.append(ticker)
        except Exception as e:
            print(f"    Failed: {e}")
    print(f"\nValid stocks with data: {len(valid_tickers)}")

    # Download fundamentals
    print("\nDownloading fundamentals (this may be slow)...")
    fundamentals = {}
    for i, ticker in enumerate(valid_tickers, 1):
        print(f"  [{i:3d}/{len(valid_tickers)}] {ticker}")
        f = get_fundamentals(ticker, price_data[ticker]["Close"])
        if f is not None:
            fundamentals[ticker] = f
    print(f"Stocks with fundamentals: {len(fundamentals)}")

    # Build value percentile panel
    print("\nBuilding value signal panel...")
    percentile_panel = build_value_history(price_data, fundamentals)
    if percentile_panel is None or percentile_panel.empty:
        print("Could not build value signal. Exiting.")
        return

    # Download benchmark
    print("\nDownloading benchmark data (^CNX100)...")
    bench = yf.download("^CNX100", start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
    if isinstance(bench.columns, pd.MultiIndex):
        bench.columns = bench.columns.get_level_values(0)
    bench = bench[["Close"]].dropna()

    # Generate rebalance dates (quarter-end) and align to next trading day
    closes = pd.DataFrame({t: df["Close"] for t, df in price_data.items()}).sort_index()
    all_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    rebalance_dates = pd.Series(index=all_dates).resample(REBALANCE_FREQ).last().index
    trading_days = closes.index
    rebalance_dates = [trading_days[trading_days >= d][0] for d in rebalance_dates
                       if any(d <= df.index.max() for df in price_data.values())]

    # Run full backtest
    print("\nRunning backtest...")
    equity_full, trades_full = run_backtest(price_data, percentile_panel, rebalance_dates)

    # Split IS / OOS
    is_equity = equity_full.loc[IS_START:IS_END]
    oos_equity = equity_full.loc[OOS_START:END_DATE]

    is_metrics = compute_metrics(is_equity, "IS")
    oos_metrics = compute_metrics(oos_equity, "OOS")
    full_metrics = compute_metrics(equity_full, "Full")

    is_bench = compute_benchmark_metrics(bench, IS_START, IS_END)
    oos_bench = compute_benchmark_metrics(bench, OOS_START, END_DATE)
    full_bench = compute_benchmark_metrics(bench, START_DATE, END_DATE)

    # Trades count
    trades_df = pd.DataFrame(trades_full)
    is_trades = trades_df[(trades_df["date"] >= IS_START) & (trades_df["date"] <= IS_END)] if not trades_df.empty else pd.DataFrame()
    oos_trades = trades_df[trades_df["date"] >= OOS_START] if not trades_df.empty else pd.DataFrame()

    # Report
    print("\n" + "=" * 70)
    print("INDIAN LONG-ONLY DEEP VALUE STRATEGY REPORT")
    print("=" * 70)
    print(f"Full period: {equity_full.index[0].strftime('%Y-%m-%d')} to {equity_full.index[-1].strftime('%Y-%m-%d')}")
    print(f"In-Sample: {IS_START} to {IS_END}")
    print(f"Out-of-Sample: {OOS_START} to {END_DATE}")
    print(f"Thresholds: IN={IN_THRESHOLD:.0%}, OUT={OUT_THRESHOLD:.0%}, ValueTrap={VALUETRAP_THRESHOLD:.0%}")
    print(f"Rebalancing: {REBALANCE_FREQ}")
    print(f"Cost: {COST_BPS:.2f}% one-way")
    print("Note: yfinance fundamentals are not point-in-time; results are approximate.")
    print("=" * 70)

    def print_section(name, m, bench_m, n_trades):
        print(f"\n{name} PERFORMANCE")
        print("-" * 50)
        if m:
            print(f"Total return        : {m['total_return']*100:>8.2f}%")
            print(f"CAGR                : {m['cagr']*100:>8.2f}%")
            print(f"Max drawdown        : {m['max_dd']*100:>8.2f}%")
            print(f"Ann. volatility     : {m['ann_vol']*100:>8.2f}%")
            print(f"Sharpe (ann.)       : {m['sharpe']:>8.2f}")
            print(f"Trades              : {n_trades:>8d}")
        else:
            print("No data")
        print(f"\nBENCHMARK (^CNX100)")
        print("-" * 50)
        if bench_m:
            print(f"Total return        : {bench_m['total_return']*100:>8.2f}%")
            print(f"CAGR                : {bench_m['cagr']*100:>8.2f}%")
            print(f"Max drawdown        : {bench_m['max_dd']*100:>8.2f}%")
            print(f"Ann. volatility     : {bench_m['ann_vol']*100:>8.2f}%")
            print(f"Sharpe (ann.)       : {bench_m['sharpe']:>8.2f}")
            print(f"Excess CAGR         : {(m['cagr'] - bench_m['cagr'])*100 if m else np.nan:>8.2f}%")
        else:
            print("No data")

    print_section("FULL PERIOD", full_metrics, full_bench, len(trades_df))
    print_section("IN-SAMPLE", is_metrics, is_bench, len(is_trades))
    print_section("OUT-OF-SAMPLE", oos_metrics, oos_bench, len(oos_trades))

    # Save outputs
    equity_full.to_csv(os.path.join(os.path.dirname(__file__), "deep_value_equity_curve.csv"))
    trades_df.to_csv(os.path.join(os.path.dirname(__file__), "deep_value_trades.csv"), index=False)
    print("\nSaved: deep_value_equity_curve.csv, deep_value_trades.csv")


if __name__ == "__main__":
    main()
