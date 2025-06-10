"""This is a copy of the ksum.py file from kditransform."""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.special import factorial


def norm_const_K(betas: np.ndarray) -> np.floating:
    factorial_terms = np.array(
        [betas[k - 1] * factorial(k - 1) for k in range(1, len(betas) + 1)]
    )
    return 2 * np.sum(factorial_terms)


def betas_for_order(order: int) -> np.ndarray:
    unnorm_betas = 1 / factorial(np.arange(order + 1))
    return unnorm_betas / norm_const_K(unnorm_betas)


def roughness_K(betas: np.ndarray) -> np.floating:
    beta_use = betas / norm_const_K(betas)
    betakj = np.outer(beta_use, beta_use)
    n = len(betas)
    kpj = np.log2(np.outer(2 ** np.arange(n), 2 ** np.arange(n))).astype(np.int64)
    return np.sum(betakj / (2**kpj) * factorial(kpj))


def var_K(betas: np.ndarray) -> np.floating:
    beta_use = betas / norm_const_K(betas)
    factorial_terms = np.array(
        [beta_use[k - 1] * factorial(k + 1) for k in range(1, len(betas) + 1)]
    )
    return 2 * np.sum(factorial_terms)


def h_Gauss_to_K(h: float | np.floating, betas: np.ndarray) -> np.floating:
    """Converts bandwidth of Gaussian kernel to that of poly-exp kernel."""
    return h * (roughness_K(betas) / (var_K(betas) ** 2) * 2 * np.sqrt(np.pi)) ** 0.2


def norm_const_K_fast(betas: np.ndarray) -> np.floating:
    (N,) = betas.shape
    return 2.0 * np.sum(betas * factorial(np.arange(N)))


def betas_for_order_fast(order: int) -> np.ndarray:
    N = order + 1
    return (1.0 / (2.0 * N)) / factorial(np.arange(N))


def roughness_K_fast(betas: np.ndarray) -> np.floating:
    # betas should already be normalized
    (N,) = betas.shape
    betakj = np.outer(betas, betas)
    powers = np.arange(N)
    kpj = np.add.outer(powers, powers)
    return np.sum(betakj / (2**kpj) * factorial(kpj))


def var_K_fast(betas: np.ndarray) -> np.floating:
    # betas should already be normalized
    (N,) = betas.shape
    return 2.0 * np.sum(betas * factorial(np.arange(2, N + 2)))


def h_Gauss_to_K_factor_fast(betas: np.ndarray) -> np.floating:
    return (
        roughness_K_fast(betas) / (var_K_fast(betas) ** 2) * 2 * np.sqrt(np.pi)
    ) ** 0.2


def h_Gauss_to_K_fast(h: float | np.floating, betas: np.ndarray) -> np.floating:
    return h * h_Gauss_to_K_factor_fast(betas)


# @njit(error_model="numpy")
@njit
def ksum_numba(  # noqa: C901, PLR0912, RUF100
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray,
    h: float,
    betas: np.ndarray,
    output: np.ndarray,
    counts: np.ndarray,
    coefs: np.ndarray,
    Ly: np.ndarray,
    Ry: np.ndarray,
):
    """Implements kernel density estimation with poly-exponential kernel.

    See "Fast exact evaluation of univariate kernel sums" (Hofmeyr, 2019)
    and https://github.com/DavidHofmeyr/FKSUM.
    """
    n = x.shape[0]
    n_eval = x_eval.shape[0]
    order = betas.shape[0] - 1
    output[:] = 0.0
    counts[:] = 0
    coefs[:] = 0.0
    Ly[:, :] = 0.0
    Ry[:, :] = 0.0

    for i in range(order + 1):
        Ly[i, 0] = np.power(-x[0], i) * y[0]
    for i in range(1, n):
        for j in range(order + 1):
            Ly[j, i] = (
                np.power(-x[i], j) * y[i] + np.exp((x[i - 1] - x[i]) / h) * Ly[j, i - 1]
            )
            Ry[j, n - i - 1] = np.exp((x[n - i - 1] - x[n - i]) / h) * (
                np.power(x[n - i], j) * y[n - i] + Ry[j, n - i]
            )

    count = 0
    for i in range(n_eval):
        if x_eval[i] >= x[n - 1]:
            counts[i] = n
        else:
            while count < n and x[count] <= x_eval[i]:
                count += 1
            counts[i] = count

    for orddo in range(order + 1):
        coefs[0] = 1
        coefs[orddo] = 1
        if orddo > 1:
            num = 1.0
            for j in range(2, orddo + 1):
                num *= j
            denom1 = 1.0
            denom2 = num / orddo
            for i in range(2, orddo + 1):
                coefs[i - 1] = num / (denom1 * denom2)
                denom1 *= i
                denom2 /= orddo - i + 1
        denom = np.power(h, orddo)

        ix = 0
        for i in range(n_eval):
            ix = np.round(counts[i])
            if ix == 0:
                exp_mult = np.exp((x_eval[i] - x[0]) / h)
                output[i] += (
                    betas[orddo]
                    * np.power(x[0] - x_eval[i], orddo)
                    / denom
                    * exp_mult
                    * y[0]
                )
                for j in range(orddo + 1):
                    output[i] += (
                        betas[orddo]
                        * coefs[j]
                        * np.power(-x_eval[i], orddo - j)
                        * Ry[j, 0]
                        / denom
                        * exp_mult
                    )
            else:
                exp_mult = np.exp((x[ix - 1] - x_eval[i]) / h)
                for j in range(orddo + 1):
                    output[i] += (
                        betas[orddo]
                        * coefs[j]
                        * (
                            np.power(x_eval[i], orddo - j) * Ly[j, ix - 1] * exp_mult
                            + np.power(-x_eval[i], orddo - j)
                            * Ry[j, ix - 1]
                            / max(exp_mult, 1e-300)
                        )
                        / denom
                    )
