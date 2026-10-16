"""Finite-sample one-sided lower confidence bound for the mean of bounded variables,
by betting (Waudby-Smith and Ramdas, JRSSB 2024, predictable plug-in hedged capital).

For x_1..x_n in [0, 1] independent with means mu_i, and mu = mean(mu_i), the set
{m : K_n(m) < 1/alpha}, K_n(m) = prod_i (1 + lambda_i (x_i - m)), with lambda_i
predictable in [0, 1/m), contains mu with probability >= 1 - alpha (Ville's inequality,
since K_t(mu) is a nonnegative supermartingale when the x_i have means mu_i averaging to
at least mu in the order they arrive; we use a uniformly random order).
"""
import math
import random


def lcb(xs, alpha, grid=2000, seed=0):
    xs = xs[:]
    random.Random(seed).shuffle(xs)
    n = len(xs)
    c = 0.5
    best = 0.0
    for gi in range(1, grid):
        m = gi / grid
        K, s1, s2 = 1.0, 0.5, 0.25                       # running mean / second moment priors
        rejected = False
        for i, x in enumerate(xs, 1):
            mu_hat = s1 / i
            var_hat = max(1e-4, s2 / i - mu_hat ** 2)
            lam = min(math.sqrt(2 * math.log(1 / alpha) / (n * var_hat)), c / m)
            K *= 1 + lam * (x - m)
            if K >= 1 / alpha:
                rejected = True
                break
            s1 += x
            s2 += x * x
        if rejected:
            best = m                                     # m is below the mean with confidence
        else:
            break
    return best
