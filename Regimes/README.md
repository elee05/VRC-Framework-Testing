# Regime Detection
The paper assumes that the market exists in 1 of 3 volatility regimes. While there is a general consensus that the market will lie in about 3 ranges of volatility, the paper provides no evidence for the exact numbers it uses: 10.2%, 22.1%, 44.8%
# This Folder
To validate this claim, this folder looks at historical returns of the SPX and identifies 3 levels of volatiliy \ Still assuming a market behavior following a 3-state Hidden Markov Model and the following:
- The system is in some hidden state $s_t \in {0,1,2}$
- Conditional on that state, the observed log-return is drawn from a Normal distribution: $r_t \~ N(\mu_{s_t},\sigma_{s_t}^2)$