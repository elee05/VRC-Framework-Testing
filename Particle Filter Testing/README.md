# PRISM: A Probabilistic Regime-Integrated Scalping Model for Derivatives Arbitrage:
- Section 9 of the mentioned paper describes how PRISM figures out, in real time, which volatility regime the market is currently in(using prices and options data)
- Since you can't observe the regime directly, you estimate a probability distribution over which regime you're likely in, and update that distribution as new data arrives. A "particle filter" is a standard technique for this: instead of tracking one guess, you track thousands of hypothetical "particles" (guesses), each representing a possible regime path, and weight them by how well they explain what's actually been observed. Over time, particles that predicted the data well get more weight; ones that predicted poorly get discarded. This is a well-established method (Sequential Importance Resampling, or SIR). Here is an attemped replication.
- PRISM is unique in that, beyond return data, it uses:
    - Return likelihood: how consistent today's price return is with each regime's assumed drift/volatility (a normal-distribution comparison).
    - Rolling realized-vol likelihood: how consistent the recent realized volatility (measured via the Garman-Klass estimator, a standard technique using high/low/open/close prices rather than just closing prices) is with each regime's volatility level (a chi-squared comparison, which is the right distribution for a variance estimate).
    - Options-surface likelihood: It compares the actual observed ATM implied vol and skew in the options market to what PRISM's own pricing model (from Section 3's PDE system) predicts those should look like if the market were in each given regime. If real observed IV/skew closely matches what "crisis regime" would imply, that regime gets more weight, even before the underlying price has actually moved dramatically.

### Algorithm:
- Initialize: start with 20,000 "particles" (guesses about the regime), all equally weighted.
- Propagate: at each new time step, let each particle transition to a new regime guess according to the Markov chain's transition probabilities (matrix exponential of the generator matrix Λ).
- Weight: score each particle by how well its regime guess explains what was actually observed (equation 51 — the three-signal likelihood).
- Normalize: rescale weights to sum to 1, forming a probability distribution.
- Aggregate: sum up the weights of all particles currently guessing "regime i" to get the overall probability the market is in regime i (this is the "posterior" π̂).
- Compute entropy: measure how spread-out/uncertain that distribution is (entropy — high when probabilities are spread across regimes, low when concentrated on one).
- Resample: if too few particles are carrying meaningful weight (a common particle-filter degeneracy problem — most particles' weights collapse toward zero over time), redraw a fresh set of particles concentrated where the current weight actually is. This keeps the filter numerically healthy.
- Broadcast: send the updated regime probabilities out to feed the bandwidth formula and Kelly sizing calculations elsewhere in the system.

### File Structure
- generator.py - simulates paths based on preset volatilities for testing