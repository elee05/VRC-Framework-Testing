# Regime Detection
The paper assumes that the market exists in 1 of 3 volatility regimes. While there is a general consensus that the market will lie in about 3 ranges of volatility, the paper provides no evidence for the exact numbers it uses: 10.2%, 22.1%, 44.8%
# This Folder
- Two implementations: hmmlearn library and from-scratch Baum-Welch/EM implementation
- Learns 
    - each intitial state's probability, $\pi$
    - transition matrix, A
    - the per state expected value and variance: $(\mu_i, \sigma_i^2)$
- Input CSV format expected: two columns, `Date` and `Close` (or `Adj Close`).
If you don't have a CSV handy, see the `download_spx()` stub below - network
access to Yahoo/Stooq is NOT available in this sandbox, so fetch that on your
own machine and point --csv at the result. Any long daily-close series works
(SPX, SPY, ^GSPC export from Yahoo Finance, stooq.com, FRED, etc).
### Usage
- python spx_hmm.py --csv spx.csv          
- python spx_hmm.py --n-states 3           # override state count
- python spx_hmm.py --scan-states          # BIC scan over 2..6 states
# Theory
To validate the paper's initial claim, this folder looks at historical returns of the SPX and identifies 3 levels of volatiliy \ Still assuming a market behavior following a 3-state Hidden Markov Model and the following:
- The system is in some hidden state $s_t \in {0,1,2}$
- Conditional on that state, the observed log-return is drawn from a Normal distribution: $r_t \sim N(\mu_{s_t},\sigma_{s_t}^2)$
- The state itself evolves as a Markov chain: $A_{ij} = P(s_{t+1}=j | s_t = i)$

### Baum-Welch Algorithm 
- variation of Expected Maximization for finding unknown parameters of Hidden Markov Model. An implementation is written in the spx_hmm.py file alongside the hmmlearn implementation.

#### E-Step
- $\alpha_t(i) = $ probability of the observations up to time $t$ and being in state $i$ at time $t$ (computed left-to-right, recursively: alpha[t] = (alpha[t-1] @ A) * B[t]