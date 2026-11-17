"""Unit tests for GradAI Stage 1.  Run:  python test_engine.py   (or: pytest)"""
import numpy as np
import sympy as sp

import gd_core as core
import mathengine as me


def fn_of(text, dom=((-5, 5), (-5, 5))):
    expr = me.parse_function(text)
    fn = me.build_function(expr)
    return me.with_minimizer(fn, dom)


# ------------------------------------------------------------- parsing
def test_valid_expressions():
    for s in ["x**2", "x^2", "2x", "x**3 - 4*x", "sin(x)", "exp(x)",
              "x**2 + y**2", "(1-x)**2 + 100*(y-x**2)**2", "3(x+1)", "ln(x)"]:
        assert isinstance(me.parse_function(s), sp.Expr), s


def test_invalid_and_unsafe_expressions():
    bad = ["", "   ", "__import__('os')", "x.__class__", "import os", "x**2; 1",
           "open('f')", "z + 1", "5", "y**2", "1/0", "x**", "(((", "lambda: 1",
           "x" * 300, "x, y", "a*x", "xy", "[x]"]
    for s in bad:
        try:
            me.parse_function(s)
        except me.ExpressionError:
            continue
        raise AssertionError(f"should have been rejected: {s!r}")


def test_derivative_and_gradient():
    f1 = me.build_function(me.parse_function("x**2"))
    assert sp.simplify(f1["grad"][0] - 2 * me.X) == 0
    f2 = me.build_function(me.parse_function("x**2 + 2*y**2"))
    assert np.allclose(f2["g"]([3.0, 1.0]), [6.0, 4.0])
    assert np.allclose(f2["H"]([0, 0]), [[2, 0], [0, 4]])


# ------------------------------------------- Tests A, B, C from the spec
def test_A_convergence():
    fn = fn_of("x**2", ((-12, 12), None))
    path, info = core.optimize(fn, [10.0], 0.1, 2000)
    assert core.status(info) == "Converged"
    assert abs(path[-1][0]) < 1e-5
    n = len(path) - 1
    # exact sequence x_n = (1 - 2*alpha)^n * x_0
    assert np.isclose(path[-1][0], (1 - 0.2) ** n * 10.0)


def test_B_slower_convergence():
    fn = fn_of("x**2", ((-12, 12), None))
    pa, _ = core.optimize(fn, [10.0], 0.1, 5000)
    pb, ib = core.optimize(fn, [10.0], 0.01, 5000)
    assert core.status(ib) == "Converged"
    assert len(pb) > len(pa)


def test_C_divergence():
    fn = fn_of("x**2", ((-12, 12), None))
    path, info = core.optimize(fn, [10.0], 1.1, 5000)
    assert core.status(info) == "Diverged"


def test_oscillation_at_alpha_one():
    fn = fn_of("x**2", ((-12, 12), None))
    path, info = core.optimize(fn, [10.0], 1.0, 50)
    assert core.status(info) == "Max iterations reached"
    assert np.allclose(np.abs(path[:, 0]), 10.0)          # x_k = (-1)^k * 10


# ------------------------------------------------------- multivariable
def test_multivariable_convergence():
    fn = fn_of("x**2 + 2*y**2")
    path, info = core.optimize(fn, [4.0, 3.0], 0.1, 2000)
    assert core.status(info) == "Converged"
    assert np.allclose(path[-1], [0, 0], atol=1e-5)


def test_rate_theory_matches_observation():
    fn = fn_of("x**2 + 5*y**2")
    path, info = core.optimize(fn, [4.0, 3.0], 0.1, 60, tol=1e-30)
    _, _, _, errs, _ = core.metrics(fn, path)
    rho = core.theoretical_rate(fn, 0.1)
    assert np.isclose(rho, 0.8)
    assert abs(core.observed_rate(errs) - rho) < 1e-6


def test_lipschitz_bound_separates_convergence_and_divergence():
    fn = fn_of("x**2 + 5*y**2")
    L = core.local_lipschitz(fn, [0, 0])
    assert np.isclose(L, 10.0)
    _, ok = core.optimize(fn, [4, 3], 0.9 * 2 / L, 3000)
    _, bad = core.optimize(fn, [4, 3], 1.1 * 2 / L, 3000)
    assert core.status(ok) == "Converged" and core.status(bad) == "Diverged"


# ----------------------------------------------------- critical points
def test_cubic_critical_points():
    fn = me.build_function(me.parse_function("x**3 - 4*x"))
    pts = me.find_critical_points(fn, ((-3, 3), None))
    xs = sorted((round(float(p["point"][0]), 4), p["kind"]) for p in pts)
    assert xs == [(-1.1547, "local maximum"), (1.1547, "local minimum")]


def test_saddle_function():
    fn = me.build_function(me.parse_function("x**2 - y**2"))
    pts = me.find_critical_points(fn, ((-4, 4), (-4, 4)))
    assert len(pts) == 1 and pts[0]["kind"] == "saddle point"


def test_himmelblau_nine_critical_points():
    fn = me.build_function(me.parse_function(me.PRESETS["Himmelblau (4 minima)"]["expr"]))
    pts = me.find_critical_points(fn, ((-5, 5), (-5, 5)))
    kinds = [p["kind"] for p in pts]
    assert kinds.count("local minimum") == 4
    assert kinds.count("saddle point") == 4
    assert kinds.count("local maximum") == 1


def test_unique_minimizer_only_when_unique():
    assert fn_of("x**2 + y**2")["xstar"] is not None
    assert fn_of("x**3 - 4*x", ((-3, 3), None))["xstar"] is None


# ------------------------------------------------------ safeguards
def test_invalid_settings():
    assert core.validate_settings(0, 10, 1e-6)
    assert core.validate_settings(-0.1, 10, 1e-6)
    assert core.validate_settings(float("nan"), 10, 1e-6)
    assert core.validate_settings(0.1, 0, 1e-6)
    assert core.validate_settings(0.1, 10, 0)
    assert core.validate_settings(0.1, 10, 1e-6) is None


def test_max_iterations_limit():
    fn = fn_of("x**2", ((-12, 12), None))
    path, info = core.optimize(fn, [10.0], 0.001, 5)
    assert info["reason"] == "maximum iterations" and len(path) == 6


def test_step_tolerance_rule():
    fn = fn_of("x**2", ((-12, 12), None))
    _, info = core.optimize(fn, [10.0], 0.1, 5000, tol=1e-6, stop="step")
    assert info["reason"] == "step tolerance"


def test_numerical_overflow_is_caught():
    fn = fn_of("exp(x)", ((-5, 3), None))
    path, info = core.optimize(fn, [800.0], 0.1, 10)
    assert info["diverged"] and np.all(np.isfinite(path))


def test_assumption_checks():
    assert me.assumption_checks(me.parse_function("abs(x)"))
    assert me.assumption_checks(me.parse_function("log(x)"))
    assert not me.assumption_checks(me.parse_function("x**2 + 1"))


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, f in tests:
        try:
            f()
            print("PASS", name)
        except Exception as err:  # noqa
            failed += 1
            print("FAIL", name, "->", repr(err))
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
