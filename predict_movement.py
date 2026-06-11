"""
S&P 500 Movement Predictor
==========================
Predicts next-day S&P 500 (SPY) direction (UP / DOWN) using supervised
classifiers trained with expanding-window walk-forward validation, so every
reported prediction is strictly out-of-sample (no look-ahead bias).

Pipeline:
  1. Load SPY OHLCV data (yfinance, a local CSV, or synthetic data for testing).
  2. Engineer the same style of features used by the regime experiments
     (returns, volatility, trend, drawdown, RSI, higher moments, volume).
  3. Build the target: sign of the NEXT day's log return.
  4. Walk-forward train/predict with Logistic Regression and
     Gradient Boosting, retraining every `step_days`.
  5. Evaluate directional accuracy vs the always-up baseline and backtest a
     long/cash strategy on the predictions via BacktestEngine.
  6. Print the prediction for the next trading day.

Usage:
  python predict_movement.py                          # fetch SPY via yfinance
  python predict_movement.py --csv spy.csv            # use a local OHLCV CSV
  python predict_movement.py --synthetic              # offline smoke test
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backtest_engine import BacktestEngine

FEATURE_COLS = [
    'Log_Return', 'Return_SMA_5', 'Volatility', 'Trend_Ratio',
    'Log_Drawdown', 'RSI', 'BB_Width', 'Rolling_Skew_63', 'Rolling_Kurt_63',
]


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Downloads OHLCV data via yfinance."""
    import yfinance as yf

    df = yf.download([ticker], start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=['Close'])
    if df.empty:
        raise RuntimeError(
            f"No data returned for {ticker}. If you are offline, pass "
            f"--csv <file> with Date/Close columns, or --synthetic."
        )
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Loads OHLCV data from a CSV with at least Date and Close columns."""
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    if 'Close' not in df.columns:
        raise ValueError(f"CSV {path} must contain a 'Close' column.")
    return df.sort_index().dropna(subset=['Close'])


def make_synthetic_prices(n_days: int = 4000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a regime-switching price series (calm bull / volatile bear) so
    the full pipeline can be exercised without network access.
    """
    rng = np.random.default_rng(seed)
    regime = 0  # 0 = bull, 1 = bear
    rets = np.zeros(n_days)
    for t in range(n_days):
        if regime == 0:
            rets[t] = rng.normal(0.0005, 0.008)
            if rng.random() < 0.005:
                regime = 1
        else:
            rets[t] = rng.normal(-0.0008, 0.022)
            if rng.random() < 0.02:
                regime = 0
    prices = 100 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range('2008-01-01', periods=n_days)
    volume = rng.lognormal(18, 0.3, n_days)
    return pd.DataFrame({'Close': prices, 'Volume': volume}, index=idx)


# ─────────────────────────────────────────────
# Feature engineering (self-contained, no pandas_ta dependency)
# ─────────────────────────────────────────────
def compute_rsi(price: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = price.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, min_periods=length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def engineer_features(df: pd.DataFrame, vol_window: int = 21,
                      ma_fast: int = 50, ma_slow: int = 200) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={'Close': 'Price'}, inplace=True)

    df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
    df['Return_SMA_5'] = df['Log_Return'].rolling(window=5).mean()
    df['Volatility'] = df['Log_Return'].rolling(window=vol_window).std() * np.sqrt(252)

    df['Trend_Ratio'] = (df['Price'].rolling(window=ma_fast).mean()
                         / df['Price'].rolling(window=ma_slow).mean())

    rolling_max = df['Price'].rolling(window=252, min_periods=1).max()
    df['Log_Drawdown'] = np.log(df['Price'] / rolling_max)

    df['RSI'] = compute_rsi(df['Price'], length=14)

    # Bollinger Band width (vol squeeze detector)
    mid = df['Price'].rolling(window=20).mean()
    sd = df['Price'].rolling(window=20).std()
    df['BB_Width'] = (4 * sd) / mid * 100

    df['Rolling_Skew_63'] = df['Log_Return'].rolling(window=63).skew()
    df['Rolling_Kurt_63'] = df['Log_Return'].rolling(window=63).kurt()

    if 'Volume' in df.columns:
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(window=50).mean()

    # Target: 1 if TOMORROW's return is positive, else 0.
    # shift(-1) means the last row has no target — that row is what we
    # ultimately predict for the next trading day.
    df['Target'] = (df['Log_Return'].shift(-1) > 0).astype(float)
    df.loc[df['Log_Return'].shift(-1).isna(), 'Target'] = np.nan

    return df


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
def build_models(random_state: int = 42) -> dict:
    return {
        'Logistic': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, C=0.1,
                                       random_state=random_state)),
        ]),
        'GradBoost': HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200,
            l2_regularization=1.0, random_state=random_state),
    }


def walk_forward_predict(features: pd.DataFrame, feature_cols: list,
                         model_name: str, initial_train_days: int = 1260,
                         step_days: int = 21) -> pd.DataFrame:
    """
    Expanding-window walk-forward: fit on [0, t), predict [t, t + step),
    advance. Returns a DataFrame with out-of-sample probabilities and labels.
    """
    data = features.dropna(subset=feature_cols + ['Target'])
    X = data[feature_cols].values
    y = data['Target'].values

    total = len(data)
    proba = pd.Series(index=data.index, dtype=float)

    train_end = initial_train_days
    while train_end < total:
        test_end = min(train_end + step_days, total)
        model = build_models()[model_name]
        model.fit(X[:train_end], y[:train_end])
        p = model.predict_proba(X[train_end:test_end])[:, 1]
        proba.iloc[train_end:test_end] = p
        train_end += step_days

    out = pd.DataFrame({
        'Proba_Up': proba,
        'Target': data['Target'],
        'Price': data['Price'],
    }).dropna(subset=['Proba_Up'])
    out['Pred_Up'] = (out['Proba_Up'] >= 0.5).astype(int)
    return out


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate(oos: pd.DataFrame) -> dict:
    acc = (oos['Pred_Up'] == oos['Target']).mean()
    baseline = oos['Target'].mean()  # accuracy of always predicting UP

    up = oos[oos['Pred_Up'] == 1]
    down = oos[oos['Pred_Up'] == 0]
    hit_up = (up['Target'] == 1).mean() if len(up) else np.nan
    hit_down = (down['Target'] == 0).mean() if len(down) else np.nan

    # Backtest a long/cash strategy: predicted UP -> long (label 0),
    # predicted DOWN -> cash (label 2). Reuses the repo's BacktestEngine,
    # which already shifts signals by one day to avoid look-ahead.
    bt_df = oos.copy()
    bt_df['Regime_Pred'] = np.where(bt_df['Pred_Up'] == 1, 0, 2)
    engine = BacktestEngine(bt_df, price_col='Price', regime_col='Regime_Pred')
    bt = engine.run_backtest()

    return {
        'Accuracy': acc,
        'Baseline_AlwaysUp': baseline,
        'Edge_vs_Baseline': acc - baseline,
        'Hit_Rate_Up_Calls': hit_up,
        'Hit_Rate_Down_Calls': hit_down,
        'N_Days': len(oos),
        'Pct_Days_Long': (oos['Pred_Up'] == 1).mean(),
        'CAGR_Strategy': bt['CAGR_Strategy'],
        'CAGR_BnH': bt['CAGR_BnH'],
        'Sharpe_Strategy': bt['Sharpe_Strategy'],
        'Sharpe_BnH': bt['Sharpe_BnH'],
        'MaxDD_Strategy': bt['Max_Drawdown_Strategy'],
        'MaxDD_BnH': bt['Max_Drawdown_BnH'],
    }


def predict_next_day(features: pd.DataFrame, feature_cols: list) -> dict:
    """
    Trains each model on ALL available labeled history and predicts the
    direction of the next trading day from the latest feature row.
    """
    labeled = features.dropna(subset=feature_cols + ['Target'])
    latest = features.dropna(subset=feature_cols).iloc[[-1]]

    results = {}
    for name, model in build_models().items():
        model.fit(labeled[feature_cols].values, labeled['Target'].values)
        p_up = model.predict_proba(latest[feature_cols].values)[0, 1]
        results[name] = p_up
    results['Ensemble'] = float(np.mean(list(results.values())))
    results['as_of'] = latest.index[-1]
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Predict next-day S&P 500 movement.')
    parser.add_argument('--ticker', default='SPY')
    parser.add_argument('--start', default='2000-01-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--csv', default=None, help='Local OHLCV CSV (Date, Close[, Volume])')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic regime-switching data (offline smoke test)')
    parser.add_argument('--initial-train-days', type=int, default=1260)
    parser.add_argument('--step-days', type=int, default=21)
    args = parser.parse_args()

    print('--- S&P 500 Next-Day Movement Predictor ---')

    if args.synthetic:
        print('\n1. Generating synthetic regime-switching data (offline mode)...')
        raw = make_synthetic_prices()
    elif args.csv:
        print(f'\n1. Loading data from {args.csv}...')
        raw = load_csv(args.csv)
    else:
        print(f'\n1. Fetching {args.ticker} data via yfinance...')
        raw = load_prices(args.ticker, args.start, args.end)
    print(f'   {len(raw)} rows, {raw.index[0].date()} to {raw.index[-1].date()}')

    print('\n2. Engineering features...')
    features = engineer_features(raw)
    feature_cols = [c for c in FEATURE_COLS if c in features.columns]
    if 'Vol_Ratio' in features.columns:
        feature_cols.append('Vol_Ratio')
    n_usable = len(features.dropna(subset=feature_cols + ['Target']))
    print(f'   Features: {feature_cols}')
    print(f'   Usable labeled rows: {n_usable}')

    min_required = args.initial_train_days + args.step_days
    if n_usable < min_required:
        sys.exit(f'Not enough data: need at least {min_required} usable rows, got {n_usable}.')

    print('\n3. Walk-forward out-of-sample evaluation '
          f'(initial train {args.initial_train_days}d, retrain every {args.step_days}d)...')
    summary = {}
    for name in build_models():
        print(f'   Evaluating {name}...')
        oos = walk_forward_predict(features, feature_cols, name,
                                   args.initial_train_days, args.step_days)
        summary[name] = evaluate(oos)

    df_metrics = pd.DataFrame(summary).T
    pct_cols = ['Accuracy', 'Baseline_AlwaysUp', 'Edge_vs_Baseline',
                'Hit_Rate_Up_Calls', 'Hit_Rate_Down_Calls', 'Pct_Days_Long',
                'CAGR_Strategy', 'CAGR_BnH', 'MaxDD_Strategy', 'MaxDD_BnH']
    fmt = df_metrics.copy()
    for col in fmt.columns:
        if col in pct_cols:
            fmt[col] = fmt[col].apply(lambda x: f'{x:.2%}')
        elif col == 'N_Days':
            fmt[col] = fmt[col].astype(int)
        else:
            fmt[col] = fmt[col].apply(lambda x: f'{x:.2f}')

    print('\n--- OUT-OF-SAMPLE RESULTS ---\n')
    print(fmt.to_string())

    print('\n4. Next trading day prediction (trained on full history):')
    nd = predict_next_day(features, feature_cols)
    as_of = nd.pop('as_of')
    print(f'   As of close on {as_of.date()}:')
    for name, p_up in nd.items():
        direction = 'UP' if p_up >= 0.5 else 'DOWN'
        print(f'   {name:>10}: {direction}  (P(up) = {p_up:.1%})')

    print('\nNote: next-day direction is close to a coin flip even for good models;'
          '\nan edge of 1-3% over the always-up baseline is meaningful. Nothing here'
          '\nis investment advice.')


if __name__ == '__main__':
    main()
