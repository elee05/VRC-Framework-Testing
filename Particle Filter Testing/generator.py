import numpy as np

# calibrated 3-regime generator matrix (per year)
lambda_12, lambda_21 = 3.2, 9.8
lambda_13, lambda_31 = 0.4, 6.1
lambda_23, lambda_32 = 1.1, 4.7

Lambda = np.array([
    [-(lambda_12+lambda_13), lambda_12, lambda_13],
    [lambda_21, -(lambda_21+lambda_23), lambda_23],
    [lambda_31, lambda_32, -(lambda_31+lambda_32)]
])

sigmas = np.array([0.102, 0.221, 0.448])  # calm, normal, crisis

def simulate_true_path(T_days, dt=1/252, seed=None):
    rng = np.random.default_rng(seed)
    n_steps = int(T_days)
    regimes = np.zeros(n_steps, dtype=int)
    regimes[0] = 0
    P = scipy.linalg.expm(Lambda * dt)  # transition matrix over dt
    for t in range(1, n_steps):
        regimes[t] = rng.choice(3, p=P[regimes[t-1]])
    returns = rng.normal(0, sigmas[regimes]*np.sqrt(dt))
    return regimes, returns