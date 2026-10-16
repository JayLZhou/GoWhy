"""Finite-sample certification bounds for the fork-point replay estimator.

Sign-split KL bound (new). With one replay per affected episode and outcomes in {0,1},
D_i = a_i (Vhat_i - Y_i) is in {-1, 0, +1}: +1 only if the logged episode failed and the
replay succeeded ("rescued"), -1 only if it succeeded and the replay failed ("broken"),
and 0 for every unaffected episode. Hence
    Delta_hat = (n_plus - n_minus) / N,   n_plus = #rescued, n_minus = #broken,
and Delta = (|F| / N) (u - v) with u, v the mean rescue and break probabilities over the
affected episodes F (independent across tasks; heterogeneous, Poisson-binomial). The
Chernoff-Hoeffding KL bound holds for Poisson-binomial counts (Hoeffding 1956: a
Poisson-binomial is more concentrated than the binomial with the same mean), so with
probability >= 1 - alpha,
    u >= u_L = inf{p <= n_plus/|F| : |F| kl(n_plus/|F|, p) <= log(2/alpha)}
    v <= v_U = sup{p >= n_minus/|F| : |F| kl(n_minus/|F|, p) <= log(2/alpha)}
    Delta >= (|F| / N) (u_L - v_U).
"""
import math


def kl(a, b):
    b = min(max(b, 1e-12), 1 - 1e-12)
    t = 0.0
    if a > 0:
        t += a * math.log(a / b)
    if a < 1:
        t += (1 - a) * math.log((1 - a) / (1 - b))
    return t


def kl_lower(k, n, eps):
    a = k / n
    lo, hi = 0.0, a
    for _ in range(100):
        mid = (lo + hi) / 2
        if n * kl(a, mid) > eps:
            lo = mid
        else:
            hi = mid
    return hi


def kl_upper(k, n, eps):
    a = k / n
    lo, hi = a, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if n * kl(a, mid) > eps:
            hi = mid
        else:
            lo = mid
    return lo


def signsplit_lcb(n_plus, n_minus, n_affected, N, alpha):
    eps = math.log(2 / alpha)
    uL = kl_lower(n_plus, n_affected, eps)
    vU = kl_upper(n_minus, n_affected, eps)
    return n_affected / N * (uL - vU)
