from scipy.stats import norm, chi2
import numpy as np
import pandas as pd

# implementation of likelihood score for each particle using 3 components

# (i) the live SPX and single-stock options chain (the VRC options surface likelihood
# (ii) Garman-Klass realized vol estimators with multiple lookback windows(5, 10, 22 days), and )
# (iii) the VIX futures term structure slope



# ! return likelihood commonent(how well do the observed returns correspond to the supposed regime)
# \phi(r_k - (m_i - 1/2 \sigma^2_i)\delta / \sigma_i \sqrt{\delta})
#   r_k = observed log return
#   (m_i - 1/2 \sigma^2_i) = expected return; m_i = expected return, \sigma^2_i  =expected variance
#   \phi = norm pdf

def return_likelihood(r, mu_i, sigma_i, dt):
    # input to standard normal pdf: (1/ \sqrt{2\pi}) * e^{-z^2 / 2}
    z = r - ((mu_i - 1/2 * sigma_i^2)/sigma_i*np.sqrt(dt))*dt
    return (1/np.sqrt(2*np.pi)) * np.e^(-z^2/2)




# ! rolling volatility likelihood component:
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

# def rolling_var_for_input(df, M, price_cols=('Open','High','Low','Close')):



def rolling_vol_likelihood(sigma_hat_sq, sigma_i_sq, M):
    """
    Rolling volatility likelihood term f_chi((M-1)*sigma_hat^2 / sigma_i^2; M-1)

    Parameters
    ----------
    sigma_hat_sq : float or np.ndarray
        Realized variance over the M preceding periods (Garman-Klass estimate)
    sigma_i_sq : float
        Hypothesized variance under regime i
    M : int
        Window length (number of preceding periods)

    Returns
    -------
    float or np.ndarray
        Chi-squared density value(s) -- the likelihood weight
    """
    dof = M - 1
    x = (dof * sigma_hat_sq) / sigma_i_sq
    return chi2.pdf(x, df=dof)

# chi function, need
#   estimated garma klass vol
#   expected vol
#   time period of estimated vol

# Notes
# Zero/negative range guard. If High == Low (illiquid bar, or bad data), ln(H/L)=0, which is fine. But if data has 
# H<L or non-positive prices, you'll get NaNs/errors 

# Annualization. If you want annualized vol rather than per-period variance, multiply by the number of periods per year before taking the square root: 
# 𝜎^ann = sqrt(\hat{sigma^2_k} * periods/year). Whether you need this depends on how sigma_i(regime vol) is scaled in the rest of  VRC model 
# — they need to be on the same time basis for the ratio (M-1)\hat{sigma^2_k}/sigma^2_i to make sense.

# Overnight gaps. Vanilla GK assumes no overnight jump (i.e., that O_t trades continuously from C_{t-1}). 
# If using daily bars with meaningful overnight gaps (equities, not FX), consider the Yang-Zhang extension, which adds an overnight-return term and is robust to opening jumps

# Window alignment with M−1 dof. .shift(1) above excludes the current bar from the rolling window, matching "preceding periods. 




# ! Options-surface likelihood 
# compares the actual observed ATM implied vol and skew in the options market to what PRISM's own pricing model (from Section 3's PDE system) 
# predicts those should look like if the market were in each given regime. 
# If real observed IV/skew closely matches what "crisis regime" would imply, 
# that regime gets more weight(even before the underlying price has actually moved dramatically.)