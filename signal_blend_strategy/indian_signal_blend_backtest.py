"""
Indian Long-Only Signal-Blending Strategy

Combines three equity signals on the NIFTY 100 universe into one composite score,
then selects the top N stocks and rebalances quarterly.

Signals:
  1. Momentum    : 12-month return skipping the most recent month.
  2. Quality     : composite of profitability (ROE, ROA, margin),
                   growth (revenue, earnings), and safety (beta, vol, debt/equity).
  3. Deep Value  : rolling historical percentile of a value score
                   (earnings yield, book-to-market, dividend yield).

Blend: equal weight by default (configurable via SIGNAL_WEIGHTS).

Caveats:
  - yfinance fundamentals are current-ish, not point-in-time.
  - Universe is current NIFTY 100 constituents (survivorship bias).
"""

import os
import warnings
from datetime import date, timedelta

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

TOP_N = 10
REBALANCE_FREQ = "QE"  # quarter-end
COST_BPS = 0.10
COST = COST_BPS / 100.0
INITIAL_CAPITAL = 1_000_000

# Equal-weight blend. Adjust to test other combinations.
SIGNAL_WEIGHTS = {
    "momentum": 1.0 / 3.0,
    "quality": 1.0 / 3.0,
    "deep_value": 1.0 / 3.0,
}

# Deep value thresholds
DV_IN = 0.80
DV_OUT = 0.50
DV_TRAP = 0.90
DV_LOOKBACK_YEARS = 5
DV_MIN_HISTORY_YEARS = 3

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
        if df.index[-1].strftime("%Y-%m-%d") >= (date.today() - timedelta(days=5)).strftime("%Y-%m-%d"):
            return df
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.to_csv(path)
    return df


def get_fundamentals(ticker):
    """Fetch latest fundamentals from yfinance."""
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

    pe = info.get("trailingPE") or info.get("forwardPE")
    pb = info.get("priceToBook")
    div_yield = info.get("dividendYield")
    if div_yield is not None and div_yield > 1.0:
        div_yield = div_yield / 100.0

    return {
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "op_margin": info.get("operatingMargins"),
        "rev_growth": info.get("revenueGrowth"),
        "earn_growth": info.get("earningsGrowth"),
        "beta": info.get("beta"),
        "debt_to_equity": info.get("debtToEquity"),
        "pe": pe,
        "pb": pb,
        "div_yield": div_yield,
        "current_price": info.get("currentPrice") or info.get("previousClose"),
    }


# ---------------------------------------------------------------------------
# SIGNALS
# ---------------------------------------------------------------------------
def compute_momentum_score(price_data):
    """12-month return skipping the most recent month. Daily panel."""
    closes = pd.DataFrame({t: df["Close"] for t, df in price_data.items()}).sort_index()
    # 12-month return skipping 1 month: (price[t-1m] / price[t-12m]) - 1
    p1m = closes.shift(21)
    p12m = closes.shift(252)
    mom = p1m / p12m - 1.0
    return mom


def compute_quality_score(fundamentals, price_data):
    """Static cross-sectional quality z-score. Replicated daily."""
    df = pd.DataFrame(fundamentals).T
    # Profitability
    df["roe_z"] = z_score(df["roe"])
    df["roa_z"] = z_score(df["roa"])
    df["opm_z"] = z_score(df["op_margin"])
    # Growth
    df["rev_z"] = z_score(df["rev_growth"])
    df["earn_z"] = z_score(df["earn_growth"])
    # Safety (lower is better, so reverse)
    df["beta_z"] = -z_score(df["beta"])
    df["de_z"] = -z_score(df["debt_to_equity"])

    # Estimate volatility from price history and reverse
    vols = {}
    for t in df.index:
        rets = price_data[t]["Close"].pct_change().dropna()
        vols[t] = rets.std() * np.sqrt(252) if len(rets) > 50 else np.nan
    df["vol"] = pd.Series(vols)
    df["vol_z"] = -z_score(df["vol"])

    cols = ["roe_z", "roa_z", "opm_z", "rev_z", "earn_z", "beta_z", "de_z", "vol_z"]
    available = [c for c in cols if df[c].notna().sum() >= 5]
    if not available:
        return pd.Series(dtype=float)
    score = df[available].mean(axis=1, skipna=True)
    return score.dropna()


def compute_deep_value_score(price_data, fundamentals):
    """Per-stock rolling percentile of a dynamic value score."""
    closes = pd.DataFrame({t: df["Close"] for t, df in price_data.items()}).sort_index()

    ey_panel = pd.DataFrame(index=closes.index, columns=fundamentals.keys(), dtype=float)
    bm_panel = pd.DataFrame(index=closes.index, columns=fundamentals.keys(), dtype=float)
    dy_panel = pd.DataFrame(index=closes.index, columns=fundamentals.keys(), dtype=float)

    for ticker, f in fundamentals.items():
        if f is None:
            continue
        price = closes[ticker]
        current_price = f.get("current_price")
        pe = f.get("pe")
        pb = f.get("pb")
        div_yield = f.get("div_yield")

        if current_price and pe and pe > 0:
            eps = current_price / pe
            ey_panel[ticker] = eps / price
        if current_price and pb and pb > 0:
            book = current_price / pb
            bm_panel[ticker] = book / price
        if current_price and div_yield and div_yield > 0:
            dividend = current_price * div_yield
            dy_panel[ticker] = dividend / price

    value_score = pd.DataFrame(index=closes.index, columns=fundamentals.keys(), dtype=float)
    for dt in closes.index:
        row_z = []
        for panel in [ey_panel, bm_panel, dy_panel]:
            vals = panel.loc[dt].dropna()
            if len(vals) < 5:
                continue
            z = (panel.loc[dt] - vals.mean()) / vals.std()
            row_z.append(z)
        if row_z:
            value_score.loc[dt] = pd.concat(row_z, axis=1).mean(axis=1, skipna=True)

    min_periods = int(252 * DV_MIN_HISTORY_YEARS)
    window = int(252 * DV_LOOKBACK_YEARS)
    dv_percentile = value_score.rolling(window=window, min_periods=min_periods).rank(pct=True)
    return dv_percentile


def z_score(s):
    s = s.replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if len(valid) < 5 or valid.std() == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - valid.mean()) / valid.std()


# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------
def run_backtest(price_data, signal_panel, rebalance_dates, initial_capital=INITIAL_CAPITAL, top_n=TOP_N):
    opens = pd.DataFrame({t: df["Open"] for t, df in price_data.items()}).sort_index()
    closes = pd.DataFrame({t: df["Close"] for t, df in price_data.items()}).sort_index()
    signal_panel = signal_panel.reindex(closes.index)

    rebalance_set = set(rebalance_dates)
    capital = float(initial_capital)
    holdings = {}
    equity_curve = []
    trades = []

    for i, today in enumerate(closes.index):
        if today in rebalance_set and i > 0:
            signal_date = closes.index[i - 1]
            score_row = signal_panel.loc[signal_date].dropna()

            # Select top N
            if len(score_row) >= top_n:
                target_tickers = score_row.sort_values(ascending=False).head(top_n).index.tolist()
            else:
                target_tickers = score_row.sort_values(ascending=False).index.tolist()

            open_prices = opens.loc[today]
            target_tickers = [t for t in target_tickers if not pd.isna(open_prices.get(t))]

            # Sell exits
            for ticker in list(holdings.keys()):
                if ticker not in target_tickers:
                    price = open_prices.get(ticker)
                    if not pd.isna(price):
                        capital += holdings[ticker] * price * (1 - COST)
                        trades.append({"date": today, "ticker": ticker, "action": "SELL", "price": price})
                    del holdings[ticker]

            # Buy new entries
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

        # Daily mark-to-market
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
def compute_metrics(equity_df):
    if equity_df.empty or len(equity_df) < 2:
        return {}
    rets = equity_df["equity"].pct_change().dropna()
    if rets.empty:
        return {}
    total_ret = equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1
    years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan
    peak = equity_df["equity"].cummax()
    dd = (equity_df["equity"] - peak) / peak
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
    print("Indian Long-Only Signal-Blending Strategy")
    print("=" * 70)
    print(f"Full period: {START_DATE} to {END_DATE}")
    print(f"In-Sample : {IS_START} to {IS_END}")
    print(f"Out-of-Sample : {OOS_START} to {END_DATE}")
    print(f"Universe: NIFTY 100 stocks")
    print(f"Signals: Momentum + Quality + Deep Value")
    print(f"Signal weights: {SIGNAL_WEIGHTS}")
    print(f"Top N: {TOP_N}, Rebalancing: {REBALANCE_FREQ}")
    print(f"Cost: {COST_BPS:.2f}% one-way")
    print("=" * 70)

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

    print("\nDownloading fundamentals...")
    fundamentals = {}
    for i, ticker in enumerate(valid_tickers, 1):
        print(f"  [{i:3d}/{len(valid_tickers)}] {ticker}")
        f = get_fundamentals(ticker)
        fundamentals[ticker] = f
    print(f"Fundamental records: {len(fundamentals)}")

    print("\nComputing signals...")
    print("  - Momentum")
    mom = compute_momentum_score(price_data)

    print("  - Quality")
    qual = compute_quality_score(fundamentals, price_data)

    print("  - Deep Value")
    dv = compute_deep_value_score(price_data, fundamentals)

    # Build blended signal panel
    print("\nBuilding blended signal panel...")
    all_dates = mom.index

    # Momentum: cross-sectional rank 0-1
    mom_rank = mom.rank(axis=1, pct=True)

    # Quality: static score, replicated daily
    qual_rank = pd.DataFrame(index=all_dates, columns=qual.index)
    for t in qual.index:
        qual_rank[t] = qual[t]
    qual_rank = qual_rank.rank(axis=1, pct=True)

    # Deep value: already 0-1 percentile; higher = cheaper. We want higher = better.
    dv_rank = dv  # already percentile

    # Align
    mom_rank = mom_rank.reindex(all_dates)
    qual_rank = qual_rank.reindex(all_dates)
    dv_rank = dv_rank.reindex(all_dates)

    blended = (
        SIGNAL_WEIGHTS["momentum"] * mom_rank.fillna(0.5)
        + SIGNAL_WEIGHTS["quality"] * qual_rank.fillna(0.5)
        + SIGNAL_WEIGHTS["deep_value"] * dv_rank.fillna(0.5)
    )

    # Generate rebalance dates
    closes = pd.DataFrame({t: df["Close"] for t, df in price_data.items()}).sort_index()
    all_cal_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    rebalance_dates = pd.Series(index=all_cal_dates).resample(REBALANCE_FREQ).last().index
    trading_days = closes.index
    rebalance_dates = [trading_days[trading_days >= d][0] for d in rebalance_dates
                       if any(d <= df.index.max() for df in price_data.values())]

    print("\nRunning backtest...")
    equity_full, trades_full = run_backtest(price_data, blended, rebalance_dates)

    is_equity = equity_full.loc[IS_START:IS_END]
    oos_equity = equity_full.loc[OOS_START:END_DATE]

    is_metrics = compute_metrics(is_equity)
    oos_metrics = compute_metrics(oos_equity)
    full_metrics = compute_metrics(equity_full)

    bench = yf.download("^CNX100", start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
    if isinstance(bench.columns, pd.MultiIndex):
        bench.columns = bench.columns.get_level_values(0)
    bench = bench[["Close"]].dropna()

    is_bench = compute_benchmark_metrics(bench, IS_START, IS_END)
    oos_bench = compute_benchmark_metrics(bench, OOS_START, END_DATE)
    full_bench = compute_benchmark_metrics(bench, START_DATE, END_DATE)

    trades_df = pd.DataFrame(trades_full)
    is_trades = trades_df[(trades_df["date"] >= IS_START) & (trades_df["date"] <= IS_END)] if not trades_df.empty else pd.DataFrame()
    oos_trades = trades_df[trades_df["date"] >= OOS_START] if not trades_df.empty else pd.DataFrame()

    print("\n" + "=" * 70)
    print("INDIAN LONG-ONLY SIGNAL-BLENDING STRATEGY REPORT")
    print("=" * 70)
    print(f"Full period: {equity_full.index[0].strftime('%Y-%m-%d')} to {equity_full.index[-1].strftime('%Y-%m-%d')}")
    print(f"In-Sample: {IS_START} to {IS_END}")
    print(f"Out-of-Sample: {OOS_START} to {END_DATE}")
    print(f"Signals: Momentum {SIGNAL_WEIGHTS['momentum']:.0%} | Quality {SIGNAL_WEIGHTS['quality']:.0%} | Deep Value {SIGNAL_WEIGHTS['deep_value']:.0%}")
    print(f"Top N: {TOP_N}, Rebalancing: {REBALANCE_FREQ}, Cost: {COST_BPS:.2f}% one-way")
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

    equity_full.to_csv(os.path.join(os.path.dirname(__file__), "signal_blend_equity_curve.csv"))
    trades_df.to_csv(os.path.join(os.path.dirname(__file__), "signal_blend_trades.csv"), index=False)
    print("\nSaved: signal_blend_equity_curve.csv, signal_blend_trades.csv")


if __name__ == "__main__":
    main()
