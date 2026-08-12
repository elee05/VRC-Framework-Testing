"""
3-state Gaussian HMM on SPX daily log-returns (2000-present)
==============================================================

Two implementations are provided:
  1. `fit_hmmlearn()`   - production-grade, uses the `hmmlearn` library
  2. `GaussianHMMScratch` - a from-scratch Baum-Welch/EM implementation,
     useful for understanding the mechanics or if you can't install hmmlearn.

Usage
-----
    python spx_hmm.py --csv spx.csv          # your own OHLC/close CSV
    python spx_hmm.py --n-states 3           # override state count
    python spx_hmm.py --scan-states          # BIC scan over 2..6 states

Input CSV format expected: two columns, `Date` and `Close` (or `Adj Close`).
If you don't have a CSV handy, see the `download_spx()` stub below - network
access to Yahoo/Stooq is NOT available in this sandbox, so fetch that on your
own machine and point --csv at the result. Any long daily-close series works
(SPX, SPY, ^GSPC export from Yahoo Finance, stooq.com, FRED, etc).
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Data loading / prep
# ----------------------------------------------------------------------

def load_prices(csv_path: str, start: str = "2000-01-01") -> pd.Series:
    df = pd.read_csv(csv_path)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    price_col = None
    for cand in ["Adj Close", "Close", "close", "adj_close", "Price"]:
        if cand in df.columns:
            price_col = cand
            break
    if price_col is None:
        raise ValueError(f"Couldn't find a close-price column in {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    s = df[price_col].astype(float)
    s = s[s.index >= start]
    return s.dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def download_spx_stub():
    """
    Placeholder showing how you'd pull data on your own machine
    (this sandbox has no network access to finance APIs).

    import yfinance as yf
    px = yf.download("^GSPC", start="2000-01-01")["Adj Close"]
    px.to_csv("spx.csv")
    """
    raise NotImplementedError("Run this on your own machine, save to CSV, then use --csv")


# ----------------------------------------------------------------------
# 2. hmmlearn version
# ----------------------------------------------------------------------

def fit_hmmlearn(returns: pd.Series, n_states: int = 3, n_iter: int = 1000,
                  random_state: int = 42):
    from hmmlearn.hmm import GaussianHMM

    X = returns.values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",   # diag == full in 1D; "full" also fine
        n_iter=n_iter,
        tol=1e-6,
        random_state=random_state,
        init_params="stmc",       # let it init start-prob, transmat, means, covars
    )
    model.fit(X)

    # Relabel states by ascending volatility (state 0 = calmest)
    order = np.argsort(model.covars_.flatten())
    means = model.means_.flatten()[order]
    stds = np.sqrt(model.covars_.flatten())[order]
    transmat = model.transmat_[np.ix_(order, order)]
    startprob = model.startprob_[order]

    # relabel the fitted model in place so predict()/predict_proba() come out ordered
    # (covars_ getter returns shape (K,1,1) but the setter wants (K, n_dim) -> reshape)
    model.means_ = model.means_[order]
    model.covars_ = model.covars_[order].reshape(n_states, -1)
    model.transmat_ = transmat
    model.startprob_ = startprob

    states = model.predict(X)                 # Viterbi path
    state_probs = model.predict_proba(X)       # smoothed marginals
    loglik = model.score(X)

    n_params = n_states * n_states - 1 + n_states * 2  # transmat(off-diag) + mean+var per state, roughly
    bic = -2 * loglik + n_params * np.log(len(X))

    summary = pd.DataFrame({
        "mean_daily": means,
        "std_daily": stds,
        "ann_return_%": means * 252 * 100,
        "ann_vol_%": stds * np.sqrt(252) * 100,
    })

    return {
        "model": model,
        "states": states,
        "state_probs": state_probs,
        "loglik": loglik,
        "bic": bic,
        "summary": summary,
        "transmat": transmat,
    }


def scan_n_states(returns: pd.Series, k_range=range(2, 7)):
    rows = []
    for k in k_range:
        res = fit_hmmlearn(returns, n_states=k)
        rows.append({"n_states": k, "loglik": res["loglik"], "bic": res["bic"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 3. From-scratch Gaussian HMM (Baum-Welch / EM)
# ----------------------------------------------------------------------

class GaussianHMMScratch:
    """
    Minimal 1-D Gaussian HMM fit via EM (Baum-Welch), for pedagogical use.
    Not optimized for speed/numerical robustness the way hmmlearn is -
    for real work prefer fit_hmmlearn().
    """

    def __init__(self, n_states=3, n_iter=200, tol=1e-6, random_state=42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

    def _init_params(self, x):
        k = self.n_states
        # init means by quantiles of the data, stds by overall std
        qs = np.linspace(0.1, 0.9, k)
        self.mu = np.quantile(x, qs)
        self.sigma2 = np.full(k, np.var(x))
        self.pi = np.full(k, 1.0 / k)
        A = self.rng.dirichlet(np.ones(k) * 5, size=k)  # mildly sticky init
        self.A = A

    @staticmethod
    def _gauss_pdf(x, mu, sigma2):
        return (1.0 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-0.5 * (x - mu) ** 2 / sigma2)

    def _emission_matrix(self, x):
        T = len(x)
        B = np.zeros((T, self.n_states))
        for j in range(self.n_states):
            B[:, j] = self._gauss_pdf(x, self.mu[j], self.sigma2[j])
        return np.clip(B, 1e-300, None)

    def _forward_backward(self, B):
        T, K = B.shape
        alpha = np.zeros((T, K))
        c = np.zeros(T)  # scaling factors for numerical stability

        alpha[0] = self.pi * B[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]

        beta = np.zeros((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / c[t + 1]

        loglik = np.sum(np.log(c))
        return alpha, beta, c, loglik

    def fit(self, x):
        x = np.asarray(x).flatten()
        self._init_params(x)
        prev_ll = -np.inf

        for it in range(self.n_iter):
            B = self._emission_matrix(x)
            alpha, beta, c, loglik = self._forward_backward(B)

            gamma = alpha * beta          # (T, K), already normalized (sums to 1 per row)
            gamma /= gamma.sum(axis=1, keepdims=True)

            T, K = B.shape
            xi_sum = np.zeros((K, K))
            for t in range(T - 1):
                num = (alpha[t][:, None] * self.A * B[t + 1][None, :] * beta[t + 1][None, :])
                xi_sum += num / c[t + 1]

            # M-step
            self.pi = gamma[0]
            self.A = xi_sum / xi_sum.sum(axis=1, keepdims=True)
            Nk = gamma.sum(axis=0)
            self.mu = (gamma * x[:, None]).sum(axis=0) / Nk
            self.sigma2 = (gamma * (x[:, None] - self.mu[None, :]) ** 2).sum(axis=0) / Nk
            self.sigma2 = np.clip(self.sigma2, 1e-10, None)

            if abs(loglik - prev_ll) < self.tol:
                break
            prev_ll = loglik

        self.loglik_ = loglik
        # sort states by variance ascending for interpretability
        order = np.argsort(self.sigma2)
        self.mu, self.sigma2 = self.mu[order], self.sigma2[order]
        self.A = self.A[np.ix_(order, order)]
        self.pi = self.pi[order]
        return self

    def predict_proba(self, x):
        x = np.asarray(x).flatten()
        B = self._emission_matrix(x)
        alpha, beta, c, _ = self._forward_backward(B)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def predict(self, x):
        """Viterbi decoding of the most likely state sequence."""
        x = np.asarray(x).flatten()
        B = self._emission_matrix(x)
        T, K = B.shape
        logA = np.log(self.A + 1e-300)
        logB = np.log(B)
        logpi = np.log(self.pi + 1e-300)

        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)
        delta[0] = logpi + logB[0]
        for t in range(1, T):
            for j in range(K):
                scores = delta[t - 1] + logA[:, j]
                psi[t, j] = np.argmax(scores)
                delta[t, j] = scores[psi[t, j]] + logB[t, j]

        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states


# ----------------------------------------------------------------------
# 4. Plotting
# ----------------------------------------------------------------------

def plot_regimes(prices, returns, states, out_path="spx_hmm_regimes.png", n_states=3):
    aligned_prices = prices.reindex(returns.index)
    colors = plt.cm.viridis(np.linspace(0, 1, n_states))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(aligned_prices.index, aligned_prices.values, color="black", lw=0.8)
    ax1.set_yscale("log")
    ax1.set_title("SPX with HMM-decoded regimes (state 0 = lowest vol)")

    for s in range(n_states):
        mask = states == s
        ax1.scatter(aligned_prices.index[mask], aligned_prices.values[mask],
                     color=colors[s], s=4, label=f"State {s}")
    ax1.legend(loc="upper left")

    ax2.plot(returns.index, states, drawstyle="steps-post", color="gray", lw=0.7)
    ax2.set_ylabel("State")
    ax2.set_yticks(range(n_states))

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved plot -> {out_path}")


# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="CSV with Date + Close columns")
    ap.add_argument("--start", type=str, default="2000-01-01")
    ap.add_argument("--n-states", type=int, default=3)
    ap.add_argument("--scratch", action="store_true", help="use from-scratch EM instead of hmmlearn")
    ap.add_argument("--scan-states", action="store_true", help="BIC scan over 2..6 states")
    args = ap.parse_args()

    prices = load_prices(args.csv, start=args.start)
    rets = log_returns(prices)
    print(f"Loaded {len(prices)} prices, {len(rets)} daily log-returns "
          f"from {rets.index.min().date()} to {rets.index.max().date()}")

    if args.scan_states:
        scan = scan_n_states(rets)
        print("\nBIC scan (lower is better):")
        print(scan.to_string(index=False))
        return

    if args.scratch:
        model = GaussianHMMScratch(n_states=args.n_states).fit(rets.values)
        states = model.predict(rets.values)
        print("\n--- From-scratch EM results ---")
        print("Means (daily):", model.mu)
        print("Std devs (daily):", np.sqrt(model.sigma2))
        print("Transition matrix:\n", model.A)
        print(f"Log-likelihood: {model.loglik_:.2f}")
    else:
        res = fit_hmmlearn(rets, n_states=args.n_states)
        states = res["states"]
        print("\n--- hmmlearn results ---")
        print(res["summary"].to_string())
        print("\nTransition matrix (rows sum to 1):")
        print(pd.DataFrame(res["transmat"]).round(3).to_string())
        print(f"\nLog-likelihood: {res['loglik']:.2f}   BIC: {res['bic']:.2f}")

        # persistence / expected regime duration = 1 / (1 - p_stay)
        durations = 1.0 / (1.0 - np.diag(res["transmat"]))
        print("\nExpected regime duration (days):", np.round(durations, 1))

    plot_regimes(prices, rets, states, n_states=args.n_states)


if __name__ == "__main__":
    main()
