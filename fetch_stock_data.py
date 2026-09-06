"""
Pulls 5 years of historical daily price data from Yahoo Finance (yfinance)
for a fixed basket of large-cap stocks and saves the combined result to a
single CSV file.

Usage:
    python fetch_stock_data.py [--period 5y] [--output stock_data_5y.csv]
"""

import argparse
import sys

import pandas as pd
import yfinance as yf

# Ticker -> company name, as requested.
TICKERS = {
    "META": "Meta Platforms Inc Class A",
    "GOOGL": "Alphabet Inc Class A",
    "T": "AT&T Inc",
    "AMZN": "Amazon.com Inc",
    "TSLA": "Tesla Inc",
    "HD": "Home Depot Inc",
    "WMT": "Walmart Inc",
    "COST": "Costco Wholesale Corp",
    "KO": "Coca-Cola Co",
    "XOM": "Exxon Mobil Corp",
    "CVX": "Chevron Corp",
    "COP": "ConocoPhillips",
    "JPM": "JPMorgan Chase & Co",
    "BRK-B": "Berkshire Hathaway Inc Class B",
    "V": "Visa Inc Class A",
    "LLY": "Eli Lilly & Co",
    "JNJ": "Johnson & Johnson",
    "ABBV": "AbbVie Inc",
    "CAT": "Caterpillar Inc",
    "GE": "General Electric Co",
    "RTX": "RTX Corp",
    "LIN": "Linde PLC",
    "NEM": "Newmont Corp",
    "FCX": "Freeport-McMoRan Inc",
    "WELL": "Welltower Inc",
    "PLD": "Prologis Inc",
    "EQIX": "Equinix Inc",
    "NVDA": "Nvidia Corp",
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft Corp",
    "NEE": "NextEra Energy Inc",
    "SO": "Southern Co",
    "DUK": "Duke Energy Corp",
}


def fetch_all(tickers: dict, period: str = "5y") -> pd.DataFrame:
    """Download `period` of daily OHLCV data for every ticker and return a
    single tidy (long-format) DataFrame with one row per ticker/date."""

    symbols = list(tickers.keys())
    print(f"Downloading {period} of daily data for {len(symbols)} tickers...")

    raw = yf.download(
        symbols,
        period=period,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=True,
    )

    if raw.empty:
        raise RuntimeError("yfinance returned no data - check tickers/network.")

    frames = []
    for symbol in symbols:
        try:
            df = raw[symbol].copy()
        except KeyError:
            print(f"  WARNING: no data returned for {symbol}, skipping.")
            continue

        df = df.dropna(how="all")
        if df.empty:
            print(f"  WARNING: empty data for {symbol}, skipping.")
            continue

        df = df.reset_index()
        df["Ticker"] = symbol
        df["Company"] = tickers[symbol]
        frames.append(df)

    if not frames:
        raise RuntimeError("No data was successfully downloaded for any ticker.")

    combined = pd.concat(frames, ignore_index=True)

    # Column order: identifying columns first, then OHLCV.
    ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in combined.columns]
    combined = combined[["Date", "Ticker", "Company"] + ohlcv_cols]
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="5y", help="yfinance period string (default: 5y)")
    parser.add_argument("--output", default="stock_data_5y.csv", help="output CSV path")
    args = parser.parse_args()

    try:
        data = fetch_all(TICKERS, period=args.period)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    data.to_csv(args.output, index=False)

    n_tickers = data["Ticker"].nunique()
    print(f"\nSaved {len(data):,} rows covering {n_tickers}/{len(TICKERS)} tickers to '{args.output}'")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")


if __name__ == "__main__":
    main()
