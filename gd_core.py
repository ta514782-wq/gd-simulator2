"""Optimizers and convergence tools for GradAI.

Pure NumPy: no Streamlit here, so every function can be unit-tested.
A 'function bundle' fn (see mathengine.build_function) provides
fn["f"](p), fn["g"](p) (gradient) and optionally fn["H"](p) (Hessian).
"""
import numpy as np

TOL = 1e-6          # default tolerance
BLOWUP = 1e8        # declare divergence when ||x|| exceeds this

METHODS = ["Gradient descent", "Momentum", "Adam"]
STOP_RULES = {
    "Gradient norm < tolerance": "gradient",
    "Step size < tolerance": "step",
    "Either": "either",
}


def optimize(fn, x0, lr, steps, method="Gradient descent", beta=0.9,
             tol=TOL, stop="gradient"):
    """Run x_{k+1} = x_k - lr * (update direction), from scratch.

    Returns (path, info). path = array of iterates x_0, x_1, ...
    info = dict(diverged: bool, reason: str).
    reason is one of: 'gradient tolerance', 'step tolerance',
    'maximum iterations', 'divergence'.
    """
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)            # momentum buffer
    m = np.zeros_like(x)            # Adam first moment
    s = np.zeros_like(x)            # Adam second moment
    path = [x.copy()]
    reason = "maximum iterations"
    with np.errstate(all="ignore"):
        for k in range(1, int(steps) + 1):
            g = fn["g"](x)
            if not np.all(np.isfinite(g)):
                reason = "divergence"
                break
            if stop in ("gradient", "either") and np.linalg.norm(g) < tol:
                reason = "gradient tolerance"
                break
            if method == "Gradient descent":
                x_new = x - lr * g
            elif method == "Momentum":
                v = beta * v + g
                x_new = x - lr * v
            else:  # Adam
                m = 0.9 * m + 0.1 * g
                s = 0.999 * s + 0.001 * g * g
                mh = m / (1 - 0.9 ** k)
                sh = s / (1 - 0.999 ** k)
                x_new = x - lr * mh / (np.sqrt(sh) + 1e-8)
            if (not np.all(np.isfinite(x_new))) or np.linalg.norm(x_new) > BLOWUP:
                reason = "divergence"
                break
            step_norm = np.linalg.norm(x_new - x)
            x = x_new
            path.append(x.copy())
            if stop in ("step", "either") and step_norm < tol:
                reason = "step tolerance"
                break
    return np.array(path), dict(diverged=(reason == "divergence"), reason=reason)


def metrics(fn, path):
    """f(x_k), ||grad f(x_k)||, ||x_k - x*|| (None if x* unknown), step sizes."""
    fvals = np.array([fn["f"](p) for p in path])
    grads = np.array([fn["g"](p) for p in path])
    gnorms = np.linalg.norm(grads, axis=1)
    errs = None
    if fn.get("xstar") is not None:
        errs = np.linalg.norm(path - fn["xstar"], axis=1)
    steps = np.r_[np.nan, np.linalg.norm(np.diff(path, axis=0), axis=1)]
    return fvals, grads, gnorms, errs, steps


def hessian(fn, x, h=1e-5):
    """Analytic Hessian if available, else central differences of the gradient."""
    if "H" in fn:
        return (lambda M: (M + M.T) / 2)(fn["H"](x))
    x = np.array(x, dtype=float)
    n = len(x)
    M = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        M[:, i] = (fn["g"](x + e) - fn["g"](x - e)) / (2 * h)
    return (M + M.T) / 2


def local_lipschitz(fn, x):
    """Largest |eigenvalue| of the Hessian at x (local Lipschitz constant of
    grad f). Plain gradient descent is stable for lr < 2/L."""
    return float(np.max(np.abs(np.linalg.eigvalsh(hessian(fn, x)))))


def theoretical_rate(fn, lr):
    """rho = max_i |1 - lr * lambda_i|, lambda_i = eigenvalues of Hessian(x*).
    Plain gradient descent converges locally iff rho < 1. None if x* unknown."""
    if fn.get("xstar") is None:
        return None
    lam = np.linalg.eigvalsh(hessian(fn, fn["xstar"]))
    return float(np.max(np.abs(1 - lr * lam)))


def observed_rate(errs, window=20):
    """Geometric-mean ratio e_{k+1}/e_k over the last `window` steps."""
    if errs is None or len(errs) < 3:
        return None
    e = errs[-(window + 1):]
    e = e[e > 1e-300]
    if len(e) < 2:
        return None
    return float(np.exp(np.mean(np.diff(np.log(e)))))


def status(info):
    if info["diverged"]:
        return "Diverged"
    if info["reason"] in ("gradient tolerance", "step tolerance"):
        return "Converged"
    return "Max iterations reached"


def iterations_to_tol(fn, path, tol=TOL):
    for k, p in enumerate(path):
        if np.linalg.norm(fn["g"](p)) < tol:
            return k
    return None


def validate_settings(lr, steps, tol):
    """Return an error message (str) if the settings are invalid, else None."""
    if not np.isfinite(lr) or lr <= 0:
        return "The learning rate must be a positive number (alpha > 0)."
    if lr > 1000:
        return "The learning rate is unreasonably large (alpha > 1000)."
    if steps < 1:
        return "Maximum iterations must be at least 1."
    if not np.isfinite(tol) or tol <= 0:
        return "The tolerance must be a positive number."
    return None
