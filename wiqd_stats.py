from typing import Dict, List, Tuple
import math
from math import comb
import numpy as np
from tqdm.auto import tqdm


def mannwhitney_u_p(x, y):
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    combined = x + y
    if min(combined) == max(combined):
        return (float("nan"), float("nan"))
    data = [(v, 0) for v in x] + [(v, 1) for v in y]
    data.sort(key=lambda t: t[0])
    R = [0.0] * (n1 + n2)
    i = 0
    while i < len(data):
        j = i
        while j < len(data) and data[j][0] == data[i][0]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            R[k] = rank
        i = j
    R1 = sum(R[:n1])
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)
    i = 0
    T = 0
    while i < len(data):
        j = i
        while j < len(data) and data[j][0] == data[i][0]:
            j += 1
        t = j - i
        if t > 1:
            T += t * (t * t - 1)
        i = j
    mu = n1 * n2 / 2.0
    sigma2 = (
        n1 * n2 * (n1 + n2 + 1) / 12.0
        - (n1 * n2 * T) / (12.0 * (n1 + n2) * (n1 + n2 - 1))
        if (n1 + n2) > 1
        else 0.0
    )
    sigma = (sigma2**0.5) if sigma2 > 0 else float("nan")
    if not (sigma == sigma) or sigma <= 0:
        return (U, float("nan"))
    z = (U - mu + 0.5) / sigma
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / (2.0**0.5))))
    return (U, max(0.0, min(1.0, p)))


def ranks_avg(values):
    enumerated = list(enumerate(values))
    enumerated.sort(key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(enumerated):
        j = i
        while j < len(enumerated) and enumerated[j][1] == enumerated[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[enumerated[k][0]] = avg_rank
        i = j
    return ranks


def mannwhitney_U_only(x, y):
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    values = x + y
    ranks = ranks_avg(values)
    n1 = len(x)
    R1 = sum(ranks[:n1])
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * len(y) - U1
    return min(U1, U2)


def perm_pvalue_U(x, y, iters=5000, rng_seed=123, progress_desc=None):

    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    pooled = x + y
    U_obs = mannwhitney_U_only(x, y)
    U_obs_min = min(U_obs, n1 * n2 - U_obs)
    cnt = 0
    iterable = range(iters)

    iterable = tqdm(iterable, desc=progress_desc) if progress_desc else iterable

    rng = np.random.default_rng(rng_seed)
    arr = np.array(pooled, dtype=float)

    for _ in iterable:
        rng.shuffle(arr)
        xp = arr[:n1].tolist()
        yp = arr[n1:].tolist()
        U_perm = mannwhitney_U_only(xp, yp)
        if min(U_perm, n1 * n2 - U_perm) <= U_obs_min:
            cnt += 1

    return (cnt + 1) / (iters + 1)


def cliffs_delta(x, y):
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    gt = lt = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                gt += 1
            elif xi < yj:
                lt += 1
    n = n1 * n2
    return (gt - lt) / n if n else float("nan")


def cohens_d_and_g(x, y):
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return (float("nan"), float("nan"))
    mx = sum(x) / n1
    my = sum(y) / n2
    sx = (sum((v - mx) ** 2 for v in x) / (n1 - 1)) ** 0.5
    sy = (sum((v - my) ** 2 for v in y) / (n2 - 1)) ** 0.5
    sp2 = (
        ((n1 - 1) * sx * sx + (n2 - 1) * sy * sy) / (n1 + n2 - 2)
        if (n1 + n2 - 2) > 0
        else float("nan")
    )
    if not (sp2 == sp2) or sp2 <= 0:
        return (float("nan"), float("nan"))
    d = (my - mx) / (sp2**0.5)
    J = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0)) if (n1 + n2) > 2 else 1.0
    g = J * d
    return d, g


def fisher_exact(a, b, c, d):

    n = a + b + c + d
    if n == 0:
        return float("nan")
    row1 = a + b
    col1 = a + c
    if row1 == 0 or col1 == 0 or row1 == n or col1 == n:
        return float("nan")

    def pmf(x):
        return comb(col1, x) * comb(n - col1, row1 - x) / comb(n, row1)

    obs = pmf(a)
    p = 0.0
    lo = max(0, row1 - (n - col1))
    hi = min(row1, col1)
    for x in range(lo, hi + 1):
        px = pmf(x)
        if px <= obs + 1e-15:
            p += px
    return max(0.0, min(1.0, p))


def bh_fdr(pvals: Dict[str, float]) -> Dict[str, float]:
    valid = [(k, v) for k, v in pvals.items() if isinstance(v, float) and v == v]
    m = len(valid)
    if m == 0:
        return {k: float("nan") for k in pvals.keys()}
    valid.sort(key=lambda kv: kv[1])
    qs = [0.0] * m
    for i, (_, p) in enumerate(valid):
        qs[i] = p * m / (i + 1)
    for i in range(m - 2, -1, -1):
        qs[i] = min(qs[i], qs[i + 1])
    out = {}
    for i, (k, _) in enumerate(valid):
        out[k] = min(qs[i], 1.0)
    for k in pvals.keys():
        if k not in out:
            out[k] = float("nan")
    return out


def fmt_p(p: float) -> str:
    if not (isinstance(p, float) and p == p):
        return "NA"
    if p <= 0:
        return "<1e-300"
    if p < 1e-12:
        return "<1e-12"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.3g}"


fmt_q = fmt_p
