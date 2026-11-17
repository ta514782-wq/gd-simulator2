"""Mathematical engine for GradAI (Stage 1).

* Safe parsing of user-typed functions f(x) or f(x, y) with SymPy
  (a strict character/name whitelist is applied BEFORE parsing).
* Symbolic gradient and Hessian, fast NumPy versions via lambdify.
* Critical-point search (multi-start Newton) and second-derivative
  classification (minimum / maximum / saddle / inconclusive).
* Light checks of the Real Analysis assumptions (continuity and
  differentiability).

No Streamlit code lives here, so everything can be unit-tested.
"""
import re

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, convert_xor, implicit_multiplication,
)

X, Y = sp.symbols("x y", real=True)

MAX_LEN = 200
ALLOWED_NAMES = {
    "x", "y", "pi", "e", "sin", "cos", "tan", "exp", "log", "ln", "sqrt",
    "abs", "sinh", "cosh", "tanh", "asin", "acos", "atan",
}
_NAMESPACE = {
    "x": X, "y": Y, "pi": sp.pi, "e": sp.E,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
    "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
}
_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication)


class ExpressionError(ValueError):
    """Raised when the user's function text is invalid or unsafe."""


# --------------------------------------------------------------- presets
# expr, plotting domain ((xmin, xmax), (ymin, ymax) or None), start, learning rate
PRESETS = {
    "x^2": dict(expr="x**2", dom=((-12, 12), None), start=[10.0], lr=0.1),
    "x^3 - 4x": dict(expr="x**3 - 4*x", dom=((-3, 3), None), start=[2.0], lr=0.05),
    "x^4 - 3x^2 + x (two minima)": dict(expr="x**4 - 3*x**2 + x",
                                       dom=((-2.5, 2.5), None), start=[2.0], lr=0.02),
    "sin(x)": dict(expr="sin(x)", dom=((-7, 7), None), start=[1.0], lr=0.2),
    "exp(x) (no minimum)": dict(expr="exp(x)", dom=((-5, 3), None), start=[2.0], lr=0.5),
    "x^2 + y^2": dict(expr="x**2 + y**2", dom=((-5, 5), (-5, 5)), start=[4.0, 3.0], lr=0.1),
    "x^2 + 2y^2": dict(expr="x**2 + 2*y**2", dom=((-5, 5), (-5, 5)), start=[4.0, 3.0], lr=0.1),
    "x^2 - y^2 (saddle)": dict(expr="x**2 - y**2", dom=((-4, 4), (-4, 4)), start=[3.0, 0.1], lr=0.1),
    "x^2 + xy + y^2": dict(expr="x**2 + x*y + y**2", dom=((-5, 5), (-5, 5)), start=[4.0, 3.0], lr=0.2),
    "x^2 + 5y^2 (elongated bowl)": dict(expr="x**2 + 5*y**2", dom=((-5, 5), (-4, 4)), start=[4.0, 3.0], lr=0.1),
    "Rosenbrock valley": dict(expr="(1-x)**2 + 100*(y-x**2)**2", dom=((-2, 2), (-1, 3)),
                              start=[-1.2, 1.0], lr=0.001),
    "Himmelblau (4 minima)": dict(expr="(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
                                  dom=((-5, 5), (-5, 5)), start=[0.0, 0.0], lr=0.01),
}


# --------------------------------------------------------------- parsing
def parse_function(text):
    """Turn user text into a SymPy expression in x (and optionally y).

    Security: only digits, the letters of whitelisted names, + - * / ^ ( ) .
    and spaces are accepted; every identifier must be whitelisted; no
    attribute access, no commas, no underscores. Raises ExpressionError.
    """
    s = (text or "").strip()
    if not s:
        raise ExpressionError("Please type a function, for example x**2 + 3*sin(x).")
    if len(s) > MAX_LEN:
        raise ExpressionError(f"Expression is too long (max {MAX_LEN} characters).")
    if not re.fullmatch(r"[0-9A-Za-z+\-*/^().\s]+", s):
        raise ExpressionError(
            "Only numbers, x, y, + - * / ^ ( ) and the functions "
            "sin, cos, tan, exp, log, sqrt, abs are allowed.")
    if re.search(r"[A-Za-z)]\s*\.", s) or re.search(r"\.\s*[A-Za-z(]", s):
        raise ExpressionError("Invalid use of '.' in the expression.")
    for name in re.findall(r"[A-Za-z]+", s):
        if name not in ALLOWED_NAMES:
            raise ExpressionError(
                f"'{name}' is not allowed. Use x and y, write products with * "
                "(for example x*y), and choose functions from: sin, cos, tan, "
                "exp, log, sqrt, abs, sinh, cosh, tanh, asin, acos, atan.")
    try:
        expr = parse_expr(s, local_dict=dict(_NAMESPACE),
                          transformations=_TRANSFORMS, evaluate=True)
    except Exception as err:  # SyntaxError, TypeError, ...
        raise ExpressionError(f"Could not understand the expression ({type(err).__name__}). "
                              "Check brackets and operators.") from None
    if not isinstance(expr, sp.Expr):
        raise ExpressionError("That is not a mathematical expression.")
    if expr.has(sp.I) or expr.has(sp.zoo) or expr.has(sp.nan) or expr.has(sp.oo):
        raise ExpressionError("The expression is not a real-valued function.")
    free = expr.free_symbols
    if not free:
        raise ExpressionError("The function must depend on x (or x and y).")
    if free - {X, Y}:
        raise ExpressionError("Use only the variables x and y.")
    if Y in free and X not in free:
        raise ExpressionError("For a one-variable function please use x, not y.")
    return expr


def build_function(expr):
    """Create the numeric 'function bundle' used by the optimizers.

    Keys: dim, expr, grad, hess (symbolic); f, g, H (numeric, take one
    point p as array-like); f_grid (numeric, takes x, y arrays).
    """
    dim = 2 if Y in expr.free_symbols else 1
    vars_ = (X, Y)[:dim]
    grad = [sp.diff(expr, v) for v in vars_]
    hess = sp.hessian(expr, vars_)
    f_num = sp.lambdify(vars_, expr, "numpy")
    g_num = sp.lambdify(vars_, grad, "numpy")
    h_num = sp.lambdify(vars_, hess.tolist(), "numpy")

    def f(p):
        return float(np.asarray(f_num(*np.asarray(p, float)), dtype=float))

    def g(p):
        return np.array([float(np.asarray(c, dtype=float))
                         for c in g_num(*np.asarray(p, float))])

    def H(p):
        return np.array(h_num(*np.asarray(p, float)), dtype=float).reshape(dim, dim)

    def f_grid(*args):
        out = f_num(*args)
        return out * np.ones_like(args[0], dtype=float)

    return dict(dim=dim, expr=expr, grad=grad, hess=hess, f=f, g=g, H=H,
                f_grid=f_grid, xstar=None)


# ------------------------------------------------- assumptions (Real Analysis)
def assumption_checks(expr):
    """Human-readable warnings about continuity / differentiability."""
    notes = []
    if expr.has(sp.Abs):
        notes.append("Contains |.|: f is continuous but NOT differentiable at the "
                     "points where the inside is 0, so the gradient does not exist there.")
    if expr.has(sp.log):
        notes.append("Contains log: f is only defined for positive arguments; "
                     "iterates that leave the domain give invalid values.")
    if expr.has(sp.tan):
        notes.append("Contains tan: f has vertical asymptotes (discontinuities); "
                     "gradient descent can jump across them.")
    if expr.has(sp.sqrt) or any(isinstance(p, sp.Pow) and p.exp.is_Rational
                                and not p.exp.is_Integer for p in expr.atoms(sp.Pow)):
        notes.append("Contains a root: f may be non-differentiable at the boundary "
                     "of its domain.")
    if any(isinstance(p, sp.Pow) and p.exp.is_negative for p in expr.atoms(sp.Pow)):
        notes.append("Contains a division by a variable quantity: f has singularities "
                     "where the denominator is 0.")
    return notes


# ---------------------------------------------------- classification / search
def classify_point(fn, p, tol=1e-7):
    """Second-derivative test (Hessian eigenvalues) at p."""
    eig = np.linalg.eigvalsh((lambda H: (H + H.T) / 2)(fn["H"](p)))
    if np.all(eig > tol):
        return "local minimum"
    if np.all(eig < -tol):
        return "local maximum"
    if np.any(eig > tol) and np.any(eig < -tol):
        return "saddle point"
    return "inconclusive (Hessian is singular)"


def find_critical_points(fn, dom, tol=1e-9, max_iter=80):
    """Multi-start Newton search for solutions of grad f = 0 inside dom.

    Returns a list of dicts with point, f(point), type. This is a numerical
    search: it can miss critical points and says nothing outside dom.
    """
    dim = fn["dim"]
    (xa, xb), ydom = dom
    if dim == 1:
        starts = [np.array([s]) for s in np.linspace(xa, xb, 41)]
        box = [(xa, xb)]
    else:
        ya, yb = ydom
        starts = [np.array([sx, sy]) for sx in np.linspace(xa, xb, 9)
                  for sy in np.linspace(ya, yb, 9)]
        box = [(xa, xb), (ya, yb)]
    margin = [0.01 * (hi - lo) for lo, hi in box]
    found = []
    with np.errstate(all="ignore"):
        for s in starts:
            x = s.astype(float)
            ok = False
            for _ in range(max_iter):
                try:
                    g = fn["g"](x)
                except Exception:
                    break
                if not np.all(np.isfinite(g)):
                    break
                if np.linalg.norm(g) < tol:
                    ok = True
                    break
                step = np.linalg.pinv(fn["H"](x)) @ g
                n = np.linalg.norm(step)
                if not np.isfinite(n):
                    break
                if n > 2.0:
                    step = step * (2.0 / n)
                x = x - step
            if not ok:
                continue
            if any(x[i] < box[i][0] - margin[i] or x[i] > box[i][1] + margin[i]
                   for i in range(dim)):
                continue
            if all(np.linalg.norm(x - q["point"]) > 1e-5 for q in found):
                found.append(dict(point=x.copy()))
    out = []
    for q in sorted(found, key=lambda d: tuple(d["point"])):
        p = q["point"]
        out.append(dict(point=p, value=fn["f"](p), kind=classify_point(fn, p)))
    return out


def with_minimizer(fn, dom):
    """Return a copy of fn with fn['xstar'] set ONLY when the numerical search
    finds exactly one critical point in dom and it is a local minimum."""
    new = dict(fn)
    pts = find_critical_points(fn, dom)
    if len(pts) == 1 and pts[0]["kind"] == "local minimum":
        new["xstar"] = pts[0]["point"]
    else:
        new["xstar"] = None
    new["critical_points"] = pts
    return new
