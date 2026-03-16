import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from data_loader import DataLoader
from models import KMeansRegimeDetector, GMMRegimeDetector, HMMRegimeDetector, MarkovSwitchingRegimeDetector, MSGARCHRegimeDetector
from backtest_engine import BacktestEngine

def plot_regimes(df: pd.DataFrame, price_col: str, regime_col: str, title: str, filename: str):
    """
    Plots the asset price overlaid with colors representing pre-aligned regime labels.
    Labels are expected to be semantically ordered: 0=Bull, 1=Transition, 2=Crisis.
    """
    plt.figure(figsize=(14, 7))
    
    # Static mapping — trusts pre-aligned labels from walk-forward state alignment.
    # No groupby on OOS data; colours are deterministic.
    regime_names = {0: 'Bull (Low Vol)', 1: 'Transition (Med Vol)', 2: 'Crisis (High Vol)'}
    state_colors = {'Bull (Low Vol)': 'green', 'Transition (Med Vol)': 'orange', 'Crisis (High Vol)': 'red'}
    
    plt.plot(df.index, df[price_col], color='black', alpha=0.3, label='Price')
    
    for regime in sorted(df[regime_col].dropna().unique()):
        idx = df[df[regime_col] == regime].index
        state_name = regime_names.get(int(regime), f'Regime {int(regime)}')
        color = state_colors.get(state_name, 'gray')
        plt.scatter(idx, df.loc[idx, price_col], color=color, label=state_name, s=10, alpha=0.6)
        
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_cumulative_returns(df: pd.DataFrame, title: str, filename: str):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Cumulative_BnH'], label='Buy & Hold', color='gray', alpha=0.7)
    plt.plot(df.index, df['Cumulative_Strategy'], label='Regime Strategy', color='blue')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def walk_forward_validation(features: pd.DataFrame, model_features_cols: list, model_class, initial_train_days: int = 1260, step_days: int = 21):
    """
    Performs expanding window walk-forward validation to eliminate look-ahead bias.
    
    :param initial_train_days: Days to use for generating the very first model (e.g. 5 years).
    :param step_days: How often to retrain the model (e.g. 21 days = roughly 1 trading month).
    """
    out_of_sample_predictions = pd.Series(index=features.index, dtype=float)
    
    total_days = len(features)
    current_train_end = initial_train_days
    
    print(f"      Initial Train set: {current_train_end} days.")
    print(f"      Stepping forward every {step_days} days to retrain.")

    while current_train_end < total_days:
        # Define Training and Test boundaries
        train_start = 0 # Expanding window (always start at T=0)
        train_end = current_train_end
        test_start = current_train_end
        test_end = min(current_train_end + step_days, total_days)
        
        # Split Data
        X_train = features.iloc[train_start:train_end][model_features_cols]
        X_test = features.iloc[test_start:test_end][model_features_cols]
        
        # Fit Model ONLY on Training Data
        model = model_class(n_regimes=3)
        model.fit(X_train)
        
        # Predict ONLY on Unseen Test Data
        # Fix: For sequential models (HMM/MSM), predicting the whole test block at once uses
        # Viterbi look-ahead bias (future test data used to predict current test data).
        # We must predict step-by-step using history up to that step.
        if model_class in [HMMRegimeDetector, MarkovSwitchingRegimeDetector]:
            preds_list = []
            for i in range(len(X_test)):
                # History from train_start up to the current test day (inclusive)
                X_history = features.iloc[train_start : test_start + i + 1][model_features_cols]
                # Predict on history and take only the last day's state (no look-ahead)
                pred = model.predict(X_history).iloc[-1]
                preds_list.append(pred)
            preds = pd.Series(preds_list, index=X_test.index)
        else:
            # For non-sequential models (KMeans, GMM, MSGARCH), batch prediction is safe and fast
            preds = model.predict(X_test)
        
        # --- CRITICAL BUG FIX: STATE ALIGNMENT ---
        # HMM states instantiate randomly. 'State 0' could be Bull in one window and Bear in another.
        # We must semantically map the predicted raw state back to risk profiles (0=Low Vol, 2=High Vol)
        # using the training set's cluster distribution before storing.
        
        train_preds = getattr(model, 'train_labels_', None)
        if train_preds is None:
            train_preds = model.predict(X_train)
            
        df_train = X_train.copy()
        df_train['Raw_Label'] = train_preds
        
        # Sort train clusters by their average volatility
        vol_means = df_train.groupby('Raw_Label')['Volatility'].mean().sort_values()
        
        # Map: Original Raw Label -> Relative Rank (0=Lowest Vol (Bull), 2=Highest Vol (Crisis))
        label_mapping = {vol_means.index[i]: i for i in range(len(vol_means))}
        
        # Map test predictions consistently and store
        mapped_preds = pd.Series(preds).map(label_mapping).values
        out_of_sample_predictions.iloc[test_start:test_end] = mapped_preds
        
        # Move window forward
        current_train_end += step_days

    # We drop the initial training period because we have no out-of-sample predictions for it
    return out_of_sample_predictions.dropna()

def run_experiment():
    print("--- Rigorous Out-of-Sample Market Regimes Experiment ---")
    
    # 1. Load Data
    print("\n1. Fetching S&P 500 Data (2000 - 2023)...")
    loader = DataLoader(['SPY'], '2000-01-01', '2023-12-31')
    loader.fetch_data()
    features = loader.engineer_features('SPY')
    
    # Reduced complexity to prevent curse of dimensionality in K-Means/HMM
    model_feat_cols = ['Return_SMA_5', 'Volatility']
    
    # Clean drops
    features.dropna(subset=model_feat_cols, inplace=True)
    
    metrics_summary = {}

    models = {
        'K-Means':  KMeansRegimeDetector,
        'GMM':      GMMRegimeDetector,
        'HMM':      HMMRegimeDetector,
        'MSM':      MarkovSwitchingRegimeDetector,
        'MS-GARCH': MSGARCHRegimeDetector
    }

    # 2. Run rigorous out-of-sample backtest
    print("\n2. Executing Expanding Window Walk-Forward Validation...")
    
    # We will slice the main 'features' dataframe to only include out-of-sample dates
    # Assuming initial burn-in is 1260 trading days (5 years)
    burn_in_days = 1260 
    out_of_sample_features = features.iloc[burn_in_days:].copy()
    
    for model_name, model_class in models.items():
        print(f"   Processing {model_name}...")
        col_name = f'Regime_{model_name}'
        
        # Get strictly out-of-sample predictions
        oos_preds = walk_forward_validation(features, model_feat_cols, model_class, initial_train_days=burn_in_days, step_days=21)
        
        # Merge predictions into our out-of-sample dataset
        out_of_sample_features[col_name] = oos_preds
        
        # Drop any NaN regimes (boundary rows between walk-forward windows)
        valid = out_of_sample_features.dropna(subset=[col_name])
        
        # Plot Regimes (only on out-of-sample data)
        plot_regimes(valid, 'Price', col_name, f'OOS S&P 500 Regimes by {model_name}', f'regime_plot_{model_name.lower()}.png')
        
        # Run Backtest (only on out-of-sample data)
        engine = BacktestEngine(valid, price_col='Price', regime_col=col_name)
        results = engine.run_backtest()
        
        plot_title = f'{model_name} Out-of-Sample Strategy vs Buy & Hold'
        plot_cumulative_returns(engine.df, plot_title, f'backtest_{model_name.lower()}.png')
        
        metrics_summary[model_name] = results
        
    print("\n--- FINAL TRUE OUT-OF-SAMPLE PERFORMANCE METRICS ---")
    df_metrics = pd.DataFrame(metrics_summary).T
    
    columns_to_print = ['CAGR_BnH', 'CAGR_Strategy', 'Sharpe_BnH', 'Sharpe_Strategy', 'Max_Drawdown_BnH', 'Max_Drawdown_Strategy']
    df_metrics = df_metrics[columns_to_print]
    
    for col in df_metrics.columns:
        if 'Sharpe' not in col:
            df_metrics[col] = df_metrics[col].apply(lambda x: f"{x:.2%}")
        else:
            df_metrics[col] = df_metrics[col].apply(lambda x: f"{x:.2f}")
            
    print("\n")
    print(df_metrics.to_string())
    print("\nWalk-forward optimizations and charts saved correctly.")

if __name__ == "__main__":
    run_experiment()
