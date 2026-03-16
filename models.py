import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class BaseRegimeDetector:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.scaler = RobustScaler()
        self.model = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame):
        """Fits the scaler and model to training data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        if hasattr(self.model, 'labels_'):
            self.train_labels_ = self.model.labels_
        elif hasattr(self.model, 'predict'):
            self.train_labels_ = self.model.predict(X_scaled)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predicts regime for new unseen test data using fitted scaler and model."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        X_scaled = self.scaler.transform(X)
        regimes = self.model.predict(X_scaled)
        return pd.Series(regimes, index=X.index)


# ─────────────────────────────────────────────
# 1. K-Means Clustering
# ─────────────────────────────────────────────
class KMeansRegimeDetector(BaseRegimeDetector):
    def __init__(self, n_regimes=3, random_state=42):
        super().__init__(n_regimes)
        self.model = KMeans(n_clusters=n_regimes, random_state=random_state, n_init='auto')


# ─────────────────────────────────────────────
# 2. Gaussian Mixture Model
# ─────────────────────────────────────────────
class GMMRegimeDetector(BaseRegimeDetector):
    def __init__(self, n_regimes=3, random_state=42):
        super().__init__(n_regimes)
        self.model = GaussianMixture(n_components=n_regimes, random_state=random_state)


# ─────────────────────────────────────────────
# 3. Hidden Markov Model
# ─────────────────────────────────────────────
class HMMRegimeDetector(BaseRegimeDetector):
    def __init__(self, n_regimes=3, random_state=42):
        super().__init__(n_regimes)
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            random_state=random_state,
            n_iter=100
        )


# ─────────────────────────────────────────────
# 4. Markov-Switching Regression Model (MSM)
#    Fits statsmodels MarkovRegression (switching mean + variance) on training data.
#    For OOS prediction, extracts the learned parameters and runs a Viterbi decoder
#    manually — fully rigorous, no OOS data used during signal generation.
# ─────────────────────────────────────────────
class MarkovSwitchingRegimeDetector:
    def __init__(self, n_regimes=3, random_state=42):
        self.n_regimes = n_regimes
        self.scaler = RobustScaler()
        self._is_fitted = False
        # Learned parameters extracted from training fit
        self._trans_matrix = None   # k x k transition probability matrix
        self._regime_means = None   # k means
        self._regime_stds  = None   # k std devs

    def fit(self, X: pd.DataFrame):
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        X_scaled = self.scaler.fit_transform(X)
        endog = X_scaled[:, 0]

        model = MarkovRegression(
            endog,
            k_regimes=self.n_regimes,
            trend='c',
            switching_variance=True
        )
        # Increase maxiter and search_reps for better convergence in walk-forward
        res = model.fit(disp=False, maxiter=1000, search_reps=20)

        # Extract transition matrix — squeeze from (k, k, 1) to (k, k)
        self._trans_matrix = res.regime_transition.squeeze()

        # Extract regime means and std devs from named params
        param_names = res.model.param_names
        params      = res.params

        means, stds = [], []
        for k in range(self.n_regimes):
            const_key  = f'const[{k}]'
            sigma2_key = f'sigma2[{k}]'
            mean_val   = params[param_names.index(const_key)]  if const_key  in param_names else 0.0
            sigma2_val = params[param_names.index(sigma2_key)] if sigma2_key in param_names else 1.0
            means.append(mean_val)
            stds.append(np.sqrt(max(abs(sigma2_val), 1e-8)))

        self._regime_means = np.array(means)
        self._regime_stds  = np.array(stds)
        self._is_fitted = True
        self.train_labels_ = self.predict(X).values
        return self

    def _gaussian_loglik(self, obs: float) -> np.ndarray:
        """Log-likelihood of obs under each regime's Gaussian."""
        return -0.5 * ((obs - self._regime_means) / self._regime_stds) ** 2 \
               - np.log(self._regime_stds)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Viterbi decoding using training-learned params — fully OOS, no leakage."""
        if not self._is_fitted:
            raise ValueError("Must call fit() before predict()")

        X_scaled = self.scaler.transform(X)
        obs_seq   = X_scaled[:, 0]
        T = len(obs_seq)
        k = self.n_regimes

        log_trans = np.log(np.maximum(self._trans_matrix, 1e-300))

        # Viterbi forward pass
        delta = np.full((T, k), -np.inf)
        psi   = np.zeros((T, k), dtype=int)

        # Uniform prior for initial state
        delta[0] = self._gaussian_loglik(obs_seq[0])

        for t in range(1, T):
            loglik = self._gaussian_loglik(obs_seq[t])
            for j in range(k):
                trans_scores = delta[t-1] + log_trans[:, j]
                psi[t, j]   = np.argmax(trans_scores)
                delta[t, j] = trans_scores[psi[t, j]] + loglik[j]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return pd.Series(states, index=X.index)



# ─────────────────────────────────────────────
# 5. MS-GARCH (Markov-Switching GARCH)
#    Fits a GARCH(1,1) on training returns to extract conditional volatility.
#    Uses KMeans to cluster those vols into regimes.
#    For OOS, applies GARCH recursion with fixed training params — no refitting.
#    This correctly separates the fitting and prediction steps without OOS leakage.
# ─────────────────────────────────────────────
class MSGARCHRegimeDetector:
    def __init__(self, n_regimes=3, random_state=42):
        self.n_regimes = n_regimes
        self.scaler = RobustScaler()
        self._garch_params = None
        self._last_train_var = None
        self._last_train_resid = None
        self._vol_kmeans = KMeans(n_clusters=n_regimes, n_init='auto', random_state=random_state)
        self._is_fitted = False

    def fit(self, X: pd.DataFrame):
        from arch import arch_model

        # Scale returns to % for GARCH numerical stability
        returns = X['Return_SMA_5'].values * 100

        am = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal', rescale=False)
        res = am.fit(disp='off')

        # Store GARCH(1,1) parameters: omega, alpha, beta
        self._garch_params = {
            'omega': res.params['omega'],
            'alpha': res.params['alpha[1]'],
            'beta':  res.params['beta[1]']
        }

        # Store the last training variance and residual for OOS GARCH recursion
        self._last_train_var    = (res.conditional_volatility[-1]) ** 2
        self._last_train_resid  = res.resid[-1]

        # Cluster training conditional volatility into regimes using KMeans
        train_cond_vol = res.conditional_volatility.reshape(-1, 1)
        self._vol_kmeans.fit(train_cond_vol)
        self.train_labels_ = self._vol_kmeans.labels_

        self._is_fitted = True
        return self

    def _garch_filter(self, returns: np.ndarray) -> np.ndarray:
        """
        Applies GARCH(1,1) recursion to new OOS returns using fixed training parameters.
        This is the key mechanism that prevents look-ahead bias: we never refit to test data.
        """
        omega = self._garch_params['omega']
        alpha = self._garch_params['alpha']
        beta  = self._garch_params['beta']

        n = len(returns)
        var = np.zeros(n)

        # The first OOS variance is seeded from the last training state
        var[0] = omega + alpha * self._last_train_resid**2 + beta * self._last_train_var

        for t in range(1, n):
            var[t] = omega + alpha * returns[t-1]**2 + beta * var[t-1]

        return np.sqrt(np.maximum(var, 1e-8))  # Return conditional standard deviation

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._is_fitted:
            raise ValueError("Must call fit() before predict()")

        returns = X['Return_SMA_5'].values * 100
        cond_vol = self._garch_filter(returns).reshape(-1, 1)

        # Classify OOS vols against training regime centroids
        raw_labels = self._vol_kmeans.predict(cond_vol)
        return pd.Series(raw_labels, index=X.index)
