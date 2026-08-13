from scipy.stats import norm, chi2
import numpy as np
import pandas as pd

# implementation of likelihood score for each particle using 3 components

# (i) the live SPX and single-stock options chain (the VRC options surface likelihood
# (ii) Garman-Klass realized vol estimators with multiple lookback windows(5, 10, 22 days), and )
# (iii) the VIX futures term structure slope



# return likelihood commonent(how well do the observed returns correspond to the supposed regime)
# \phi(r_k - (m_i - 1/2 \sigma^2_i)\delta / \sigma_i \sqrt{\delta})
#   r_k = observed log return
#   (m_i - 1/2 \sigma^2_i) = expected return; m_i = expected return, \sigma^2_i  =expected variance
#   \phi = norm pdf

def return_likelihood(r, mu_i, sigma_i, dt):
    # input to standard normal pdf: (1/ \sqrt{2\pi}) * e^{-z^2 / 2}
    z = r - ((mu_i - 1/2 * sigma_i^2)/sigma_i*np.sqrt(dt))*dt
    return (1/np.sqrt(2*np.pi)) * np.e^(-z^2/2)




# rolling volatility likelihood component:
# how consistent the recent realized volatility (measured via the Garman-Klass estimator )
# is with each regime's volatility level (a chi-squared comparison, which is the right distribution for a variance estimate).
# Garman-Klass estimator uses high/low/open/close prices rather than just closing prices) 
def garman_klass_variance(open_, high, low, close):
    # Per-bar Garman-Klass variance estimate.
    # Returns: array of per-bar variance estimates
    
    open_ = np.asarray(open_, dtype=float)
    high  = np.asarray(high, dtype=float)
    low   = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    gk_var = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
    return gk_var


def rolling_gk_realized_variance(df, M, price_cols=('Open','High','Low','Close')):
    # Rolling realized variance (sigma_hat_k^2) over M preceding periods, using the Garman-Klass per-bar estimator.

    # df: DataFrame with OHLC columns
    # M: rolling window length
    # Returns: pd.Series of rolling realized variance, aligned to df.index
    
    o_col, h_col, l_col, c_col = price_cols
    per_bar_var = garman_klass_variance(df[o_col], df[h_col], df[l_col], df[c_col])
    per_bar_var = pd.Series(per_bar_var, index=df.index)

    # rolling mean of the M preceding bars (excludes current bar k, per your ℓ_k formula)
    rolling_var = per_bar_var.shift(1).rolling(window=M).mean()
    return rolling_var
def rol_vol():
    pass
