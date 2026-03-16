import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, df: pd.DataFrame, price_col: str, regime_col: str):
        """
        Initializes the backtesting engine.
        
        :param df: DataFrame containing price and regime data.
        :param price_col: Name of the column containing the asset's price.
        :param regime_col: Name of the column containing the predicted regimes.
        """
        self.df = df.copy()
        self.price_col = price_col
        self.regime_col = regime_col
        
        # Calculate standard buy and hold returns
        self.df['BnH_Return'] = np.log(self.df[self.price_col] / self.df[self.price_col].shift(1))
        
    def _map_regimes_to_signals(self):
        """
        Maps pre-aligned regime labels to trading signals.
        Labels must already be semantically ordered by the walk-forward state-alignment step:
          0 = Bull   (Low Volatility)  → 100% Long
          1 = Transition               → 100% Long
          2 = Crisis  (High Volatility) → Cash (risk-free)
        
        This mapping is STATIC — it does NOT use any OOS data to re-sort labels,
        preventing look-ahead bias in the signal assignment.
        """
        # Static trusted mapping — labels are pre-aligned in walk_forward_validation()
        SIGNAL_MAP = {0: 1.0, 1: 1.0, 2: 0.0}
        
        self.df['Signal'] = self.df[self.regime_col].map(SIGNAL_MAP).fillna(1.0)
        
        # Shift signal by 1 day to prevent look-ahead bias:
        # Trade today based on yesterday's predicted regime only.
        self.df['Signal'] = self.df['Signal'].shift(1)
        
    def run_backtest(self) -> dict:
        """
        Runs the vectorized backtest and returns performance metrics.
        """
        self._map_regimes_to_signals()
        
        # Assume a 4% annualized risk-free return on Cash (T-Bills)
        daily_rf_rate = 0.04 / 252
        
        # Calculate strategy returns: Market Return if Long, Risk-Free Rate if Cash
        self.df['Strategy_Return'] = np.where(self.df['Signal'] == 1.0, self.df['BnH_Return'], daily_rf_rate)
        
        # Drop NaN values created by shifting
        self.df.dropna(inplace=True)
        
        # Calculate cumulative returns
        self.df['Cumulative_BnH'] = np.exp(self.df['BnH_Return'].cumsum())
        self.df['Cumulative_Strategy'] = np.exp(self.df['Strategy_Return'].cumsum())
        
        # --- Performance Metrics ---
        metrics = {}
        
        # 1. Total Return
        metrics['Total_Return_BnH'] = self.df['Cumulative_BnH'].iloc[-1] - 1
        metrics['Total_Return_Strategy'] = self.df['Cumulative_Strategy'].iloc[-1] - 1
        
        # 2. Annualized Return (CAGR)
        days = (self.df.index[-1] - self.df.index[0]).days
        years = days / 365.25
        metrics['CAGR_BnH'] = (self.df['Cumulative_BnH'].iloc[-1]) ** (1/years) - 1
        metrics['CAGR_Strategy'] = (self.df['Cumulative_Strategy'].iloc[-1]) ** (1/years) - 1
        
        # 3. Annualized Volatility
        metrics['Vol_BnH'] = self.df['BnH_Return'].std() * np.sqrt(252)
        metrics['Vol_Strategy'] = self.df['Strategy_Return'].std() * np.sqrt(252)
        
        # 4. Sharpe Ratio (properly subtracts 4% annualized risk-free rate from numerator)
        rf_annual = 0.04
        metrics['Sharpe_BnH'] = (metrics['CAGR_BnH'] - rf_annual) / metrics['Vol_BnH'] if metrics['Vol_BnH'] != 0 else 0
        metrics['Sharpe_Strategy'] = (metrics['CAGR_Strategy'] - rf_annual) / metrics['Vol_Strategy'] if metrics['Vol_Strategy'] != 0 else 0
        
        # 5. Maximum Drawdown
        roll_max_bnh = self.df['Cumulative_BnH'].cummax()
        drawdown_bnh = self.df['Cumulative_BnH'] / roll_max_bnh - 1.0
        metrics['Max_Drawdown_BnH'] = drawdown_bnh.min()
        
        roll_max_strat = self.df['Cumulative_Strategy'].cummax()
        drawdown_strat = self.df['Cumulative_Strategy'] / roll_max_strat - 1.0
        metrics['Max_Drawdown_Strategy'] = drawdown_strat.min()
        
        return metrics

if __name__ == "__main__":
    from data_loader import DataLoader
    from models import HMMRegimeDetector
    
    # 1. Load Data
    loader = DataLoader(['SPY'], '2000-01-01', '2023-12-31')
    loader.fetch_data()
    features = loader.engineer_features('SPY')
    
    # 2. Split into in-sample train and out-of-sample test
    model_features = features[['Return_SMA_5', 'Volatility']].dropna()
    split = int(len(model_features) * 0.7)
    X_train = model_features.iloc[:split]
    X_test  = model_features.iloc[split:]
    
    # 3. Fit ONLY on train, predict on unseen test
    print("\nFitting HMM Model on train split...")
    hmm = HMMRegimeDetector(n_regimes=3)
    hmm.fit(X_train)
    features.loc[X_test.index, 'Regime_HMM'] = hmm.predict(X_test).values
    features.dropna(subset=['Regime_HMM'], inplace=True)
    
    # 4. Run Backtest on out-of-sample slice only
    print("\nRunning Backtest on HMM Regimes (out-of-sample only)...")
    engine = BacktestEngine(features.loc[X_test.index], price_col='Price', regime_col='Regime_HMM')
    results = engine.run_backtest()
    
    print("\n--- Backtest Results (HMM Strategy vs. Buy & Hold) ---")
    for key, value in results.items():
        if "Total" in key or "Drawdown" in key or "CAGR" in key or "Vol" in key:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value:.2f}")
