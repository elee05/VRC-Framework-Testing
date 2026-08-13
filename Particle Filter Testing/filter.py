from scipy.stats import norm, chi2
import numpy as np

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
