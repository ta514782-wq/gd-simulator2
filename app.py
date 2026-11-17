"""GradAI: Interactive Gradient Descent and Optimization Laboratory
MATH 303 Real Analysis I (Stage 1: Function Explorer + Simulator)

Run locally:   streamlit run app.py
"""
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

import gd_core as core
import mathengine as me

st.set_page_config(page_title="GradAI", page_icon="\U0001F4C9", layout="wide")

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.10);
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 12px;
    padding: 10px 14px;
}
</style>
""", unsafe_allow_html=True)

PAGES = ["Function Explorer", "Gradient Descent Simulator",
         "Real Analysis Theory", "Test Cases"]
WIDGET_PREFIXES = ("x0_", "y0_", "lr_", "n_", "tol_", "stop_", "opt_", "beta_")


@st.cache_resource(show_spinner=False)
def get_fn(expr_text):
    """Parse + build once per expression text (errors are not cached)."""
    expr = me.parse_function(expr_text)
    return me.build_function(expr)


def reset_controls():
    for k in list(st.session_state.keys()):
        if k.startswith(WIDGET_PREFIXES):
            del st.session_state[k]


# ============================================================== sidebar
st.sidebar.title("GradAI")
st.sidebar.caption("Gradient descent laboratory | MATH 303 Real Analysis I")
page = st.sidebar.radio("Module", PAGES)

st.sidebar.divider()
choice = st.sidebar.selectbox("Function f", list(me.PRESETS) + ["Custom function..."])

if choice == "Custom function...":
    expr_text = st.sidebar.text_input(
        "f(x) or f(x, y)", value="x**2 + 3*sin(x)",
        help="Use x and y, * for products (x*y), ** or ^ for powers. "
             "Functions: sin cos tan exp log sqrt abs.")
    cx = st.sidebar.columns(2)
    xa = cx[0].number_input("x min", value=-5.0, key="cxa")
    xb = cx[1].number_input("x max", value=5.0, key="cxb")
    cy = st.sidebar.columns(2)
    ya = cy[0].number_input("y min", value=-5.0, key="cya")
    yb = cy[1].number_input("y max", value=5.0, key="cyb")
    custom = True
else:
    preset = me.PRESETS[choice]
    expr_text = preset["expr"]
    custom = False

try:
    fn0 = get_fn(expr_text)
except me.ExpressionError as err:
    st.sidebar.error(str(err))
    st.error(f"Invalid function: {err}")
    st.stop()

dim = fn0["dim"]
if custom:
    if not (xa < xb) or (dim == 2 and not (ya < yb)):
        st.error("Each plotting range needs min < max.")
        st.stop()
    dom = ((xa, xb), (ya, yb) if dim == 2 else None)
    d_start = [2.0] if dim == 1 else [2.0, 2.0]
    d_lr = 0.1
else:
    dom = preset["dom"]
    d_start = preset["start"]
    d_lr = preset["lr"]

fn = me.with_minimizer(fn0, dom)
tag = hashlib.md5(expr_text.encode()).hexdigest()[:8]
fvars = "x" if dim == 1 else "x, y"


# ============================================================== helpers
def show_function_header():
    st.latex(rf"f({fvars}) = {sp.latex(fn['expr'])}")
    for note in me.assumption_checks(fn["expr"]):
        st.warning(note)


def grid_for_plot(n=140):
    (xa_, xb_), ydom = dom
    xs = np.linspace(xa_, xb_, n)
    if dim == 1:
        return xs, None, fn["f_grid"](xs)
    ys = np.linspace(ydom[0], ydom[1], n)
    XX, YY = np.meshgrid(xs, ys)
    return xs, ys, fn["f_grid"](XX, YY)


def fmt_point(p):
    return "(" + ", ".join(f"{v:.5g}" for v in p) + ")"


# ============================================================== page 1
def page_explorer():
    st.header("Function Explorer")
    st.caption("Symbolic derivatives, values, graphs, and stationary points.")
    show_function_header()

    st.subheader("Derivatives")
    if dim == 1:
        st.latex(rf"f'(x) = {sp.latex(fn['grad'][0])}")
        st.latex(rf"f''(x) = {sp.latex(fn['hess'][0, 0])}")
    else:
        st.latex(rf"\frac{{\partial f}}{{\partial x}} = {sp.latex(fn['grad'][0])}"
                 rf"\qquad \frac{{\partial f}}{{\partial y}} = {sp.latex(fn['grad'][1])}")
        st.latex(r"\nabla f = \left(\frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y}\right)"
                 rf"\quad H = {sp.latex(fn['hess'])}")
    st.caption("Derivatives are computed symbolically (exact), not by finite differences.")

    st.subheader("Evaluate at a point")
    cols = st.columns(dim)
    p = [cols[i].number_input(("x", "y")[i], value=float(d_start[i]),
                              key=f"ex_{('x', 'y')[i]}_{tag}") for i in range(dim)]
    try:
        val, gr = fn["f"](p), fn["g"](p)
        if not (np.isfinite(val) and np.all(np.isfinite(gr))):
            raise ValueError
        m = st.columns(3)
        m[0].metric("f(point)", f"{val:.6g}")
        m[1].metric("||grad f||", f"{np.linalg.norm(gr):.6g}")
        m[2].metric("grad f", fmt_point(gr))
    except Exception:
        st.error("f or its gradient is not defined at this point (outside the domain).")

    st.subheader("Graph")
    xs, ys, Z = grid_for_plot()
    if dim == 1:
        dz = fn["g"]  # derivative curve
        dvals = np.array([dz([t])[0] for t in xs])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=Z, name="f(x)", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=xs, y=dvals, name="f'(x)", line=dict(dash="dash")))
        for q in fn["critical_points"]:
            fig.add_trace(go.Scatter(x=[q["point"][0]], y=[q["value"]], mode="markers",
                                     marker=dict(size=11, symbol="diamond"),
                                     name=q["kind"], showlegend=False,
                                     hovertemplate=f"{q['kind']}<br>x=%{{x:.4f}}<extra></extra>"))
        fig.update_layout(height=430, xaxis_title="x", yaxis_title="value",
                          margin=dict(l=10, r=10, t=30, b=10))
        zfin = Z[np.isfinite(Z)]
        if zfin.size:
            lo, hi = np.percentile(zfin, [1, 99])
            pad = 0.15 * (hi - lo + 1e-9)
            fig.update_yaxes(range=[lo - pad, hi + pad])
        st.plotly_chart(fig)
        st.caption("Diamonds mark stationary points (f'(x) = 0). The dashed curve is f'(x).")
    else:
        t1, t2 = st.tabs(["3D surface", "Contour map"])
        with t1:
            fig = go.Figure(go.Surface(x=xs, y=ys, z=Z, colorscale="Viridis"))
            fig.update_layout(height=470, margin=dict(l=0, r=0, t=20, b=0),
                              scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f"))
            st.plotly_chart(fig)
        with t2:
            fig = go.Figure(go.Contour(x=xs, y=ys, z=Z, colorscale="Viridis",
                                       contours=dict(showlabels=False)))
            for q in fn["critical_points"]:
                fig.add_trace(go.Scatter(x=[q["point"][0]], y=[q["point"][1]], mode="markers",
                                         marker=dict(size=11, color="white", line=dict(width=2, color="black")),
                                         name=q["kind"], showlegend=False,
                                         hovertemplate=f"{q['kind']}<extra></extra>"))
            fig.update_layout(height=470, xaxis_title="x", yaxis_title="y",
                              margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig)
            st.caption("White dots mark critical points (grad f = 0).")

    st.subheader("Stationary (critical) points")
    pts = fn["critical_points"]
    if pts:
        st.dataframe(pd.DataFrame({
            "point": [fmt_point(q["point"]) for q in pts],
            "f(point)": [q["value"] for q in pts],
            "type": [q["kind"] for q in pts],
        }), hide_index=True)
    else:
        st.info("No critical points found inside the plotting range. "
                "(For example exp(x) has none: it keeps decreasing as x goes to -infinity.)")
    st.caption("Found by a numerical multi-start Newton search inside the plotting range, "
               "so some points could be missed. Type comes from the second-derivative test.")
    with st.expander("How the classification works"):
        st.markdown(r"""
At a critical point $\nabla f = 0$. Look at the Hessian $H$ (matrix of second partial derivatives):
- all eigenvalues $>0$: **local minimum**
- all eigenvalues $<0$: **local maximum**
- mixed signs: **saddle point** (a minimum in one direction, a maximum in another)
- some eigenvalue $=0$: the test is **inconclusive**

In one variable this is the familiar test with $f''(x)$.
""")


# ============================================================== page 2
def step_by_step(path, grads, lr):
    st.subheader("Step-by-step (first iterations)")
    n_show = min(3, len(path) - 1)
    for k in range(n_show):
        x, g, xn = path[k], grads[k], path[k + 1]
        if dim == 1:
            st.latex(rf"x_{{{k + 1}}} = x_{{{k}}} - \alpha f'(x_{{{k}}}) = "
                     rf"{x[0]:.6g} - {lr:g}\cdot({g[0]:.6g}) = {xn[0]:.6g}")
        else:
            st.latex(rf"\mathbf{{x}}_{{{k + 1}}} = \mathbf{{x}}_{{{k}}} - \alpha\nabla f(\mathbf{{x}}_{{{k}}}) = "
                     rf"({x[0]:.5g}, {x[1]:.5g}) - {lr:g}\cdot({g[0]:.5g}, {g[1]:.5g}) "
                     rf"= ({xn[0]:.5g}, {xn[1]:.5g})")


def page_simulator():
    st.header("Gradient Descent Simulator")
    show_function_header()

    st.sidebar.divider()
    st.sidebar.markdown("**Simulator settings**")
    method = st.sidebar.selectbox("Optimizer", core.METHODS, key=f"opt_{tag}")
    x0 = [st.sidebar.number_input("start x\u2080", value=float(d_start[0]), step=0.5,
                                  key=f"x0_{tag}")]
    if dim == 2:
        x0.append(st.sidebar.number_input("start y\u2080", value=float(d_start[1]), step=0.5,
                                          key=f"y0_{tag}"))
    lr = st.sidebar.number_input("Learning rate \u03b1", value=float(d_lr), min_value=-10.0,
                                 max_value=100.0, format="%.5f", step=max(d_lr / 4, 1e-5),
                                 key=f"lr_{tag}")
    steps = st.sidebar.slider("Max iterations", 1, 5000, 200, key=f"n_{tag}")
    tol = st.sidebar.number_input("Tolerance", value=1e-6, min_value=-1.0, format="%.2e",
                                  step=1e-6, key=f"tol_{tag}")
    stop_label = st.sidebar.selectbox("Stopping rule", list(core.STOP_RULES), key=f"stop_{tag}")
    st.sidebar.button("Reset settings", on_click=reset_controls)

    err = core.validate_settings(lr, steps, tol)
    if err:
        st.error(err)
        return
    try:
        if not (np.isfinite(fn["f"](x0)) and np.all(np.isfinite(fn["g"](x0)))):
            raise ValueError
    except Exception:
        st.error("f or its gradient is not defined at the starting point. Choose another start.")
        return

    path, info = core.optimize(fn, x0, lr, steps, method, tol=tol,
                               stop=core.STOP_RULES[stop_label])
    fvals, grads, gnorms, errs, stepn = core.metrics(fn, path)
    state = core.status(info)

    c = st.columns(4)
    c[0].metric("Status", state)
    c[1].metric("Iterations", len(path) - 1)
    c[2].metric("Final estimate", fmt_point(path[-1]))
    c[3].metric("f(final)", f"{fvals[-1]:.6g}")
    c = st.columns(3)
    c[0].metric("Stopped because", info["reason"])
    c[1].metric("||grad f|| at end", f"{gnorms[-1]:.3g}")
    c[2].metric("Start point", fmt_point(path[0]))

    if info["diverged"]:
        st.error("The sequence of iterates is unbounded or non-finite. Either the learning rate is "
                 "too large for this function, or f has no minimum in the direction the "
                 "iterates are moving (for example a saddle point or a function unbounded below).")
    elif state == "Converged":
        kind = me.classify_point(fn, path[-1])
        st.success(f"Stopped at {fmt_point(path[-1])}. The Hessian test there says: {kind}.")
    else:
        st.info("The iteration limit was reached before the tolerance. Increase the iterations "
                "or the learning rate, or check whether the sequence oscillates.")
    if state == "Converged" and info["reason"] == "gradient tolerance" \
            and np.linalg.norm(path[-1]) > 50:
        st.warning("The gradient is tiny but x is still far away and moving: a small gradient does "
                   "not by itself prove that the sequence x_k converges (see the Theory page).")

    # ----------------- plots
    left, right = st.columns(2)
    (xa_, xb_), ydom = dom
    if dim == 1:
        xs, _, Z = grid_for_plot(400)
        fig = go.Figure(go.Scatter(x=xs, y=Z, name="f(x)", line=dict(width=3)))
        inside = (path[:, 0] >= xa_) & (path[:, 0] <= xb_)
        fig.add_trace(go.Scatter(x=path[inside, 0], y=fvals[inside], mode="lines+markers",
                                 name="iterates", marker=dict(size=6, color="crimson"),
                                 line=dict(color="crimson", width=1)))
        fig.add_trace(go.Scatter(x=[path[0, 0]], y=[fvals[0]], mode="markers", name="start",
                                 marker=dict(size=13, color="green", symbol="square")))
        fig.add_trace(go.Scatter(x=[path[-1, 0]], y=[fvals[-1]], mode="markers", name="end",
                                 marker=dict(size=15, color="gold", symbol="star",
                                             line=dict(width=1, color="black"))))
        zf = Z[np.isfinite(Z)]
        lo, hi = np.percentile(zf, [0, 100])
        pad = 0.1 * (hi - lo + 1e-9)
        fig.update_xaxes(range=[xa_, xb_], title="x")
        fig.update_yaxes(range=[lo - pad, hi + pad], title="f(x)")
    else:
        xs, ys, Z = grid_for_plot(160)
        fig = go.Figure(go.Contour(x=xs, y=ys, z=Z, colorscale="Viridis", showscale=False,
                                   contours=dict(showlabels=False)))
        fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines+markers", name="iterates",
                                 marker=dict(size=5, color="crimson"),
                                 line=dict(color="crimson", width=1)))
        fig.add_trace(go.Scatter(x=[path[0, 0]], y=[path[0, 1]], mode="markers", name="start",
                                 marker=dict(size=12, color="lime", symbol="square",
                                             line=dict(width=1, color="black"))))
        fig.add_trace(go.Scatter(x=[path[-1, 0]], y=[path[-1, 1]], mode="markers", name="end",
                                 marker=dict(size=14, color="gold", symbol="star",
                                             line=dict(width=1, color="black"))))
        fig.update_xaxes(range=[xa_, xb_], title="x")
        fig.update_yaxes(range=[ydom[0], ydom[1]], title="y")
    fig.update_layout(height=430, title=f"{method}: path of the iterates",
                      margin=dict(l=10, r=10, t=45, b=10), legend=dict(orientation="h", y=-0.2))
    left.plotly_chart(fig)

    ks = np.arange(len(path))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ks, y=np.maximum(gnorms, 1e-16), name="||grad f(x_k)||"))
    if errs is not None:
        fig2.add_trace(go.Scatter(x=ks, y=np.maximum(errs, 1e-16), name="||x_k - x*||"))
    fig2.update_yaxes(type="log", title="log scale")
    fig2.update_xaxes(title="iteration k")
    fig2.update_layout(height=430, title="Convergence of the sequence",
                       margin=dict(l=10, r=10, t=45, b=10), legend=dict(orientation="h", y=-0.2))
    right.plotly_chart(fig2)

    fig3 = go.Figure(go.Scatter(x=ks, y=fvals, name="f(x_k)"))
    fig3.update_layout(height=300, title="Function value f(x_k)", xaxis_title="iteration k",
                       margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig3)

    # ----------------- theory vs experiment
    st.subheader("Theory vs experiment")
    L = core.local_lipschitz(fn, path[0] if info["diverged"] else path[-1])
    t = st.columns(3)
    t[0].metric("Local L (largest |eigenvalue| of H)", f"{L:.4g}")
    t[1].metric("Safe bound 2/L", f"{2 / L:.4g}" if L > 1e-12 else "no bound")
    t[2].metric("\u03b1 below the bound?", "Yes" if (L > 1e-12 and lr < 2 / L) else "No")
    if method == "Gradient descent":
        rho = core.theoretical_rate(fn, lr)
        if rho is not None:
            robs = core.observed_rate(errs)
            u = st.columns(2)
            u[0].metric("Theoretical \u03c1 = max|1 - \u03b1\u03bb|", f"{rho:.4f}")
            u[1].metric("Observed e(k+1)/e(k)", f"{robs:.4f}" if robs is not None else "n/a")
            st.caption("\u03c1 < 1 means each step shrinks the distance to the minimum by about "
                       "this factor (a contraction). The bound 2/L is a local statement near the point shown.")
        else:
            st.caption("This function does not have exactly one minimum in the range, so there "
                       "is no single contraction factor \u03c1.")
    else:
        st.caption("The closed-form \u03c1 applies to plain gradient descent only.")

    if method == "Gradient descent":
        step_by_step(path, grads, lr)

    # ----------------- table + download
    st.subheader("Iteration history")
    table = {"k": ks}
    names = ("x", "y")[:dim]
    for i, nm in enumerate(names):
        table[nm] = path[:, i]
    table["f(x_k)"] = fvals
    for i, nm in enumerate(names):
        table[f"df/d{nm}"] = grads[:, i]
    table["||grad f||"] = gnorms
    table["step size"] = stepn
    df = pd.DataFrame(table)
    st.dataframe(df, hide_index=True, height=300)
    st.download_button("Download iteration history (CSV)", df.to_csv(index=False).encode(),
                       file_name="gradai_iterations.csv", mime="text/csv")


# ============================================================== page 3
def page_theory():
    st.header("Real Analysis Theory")
    st.markdown("Each idea below is tied to what the simulator does. "
                "**Guarantees** are proved statements; **observations** come from running the code.")

    st.subheader("1. Derivatives give the direction")
    st.markdown(r"""
If $f$ is differentiable at $x$, then $f(x+h)\approx f(x)+f'(x)h$. Taking $h=-\alpha f'(x)$ gives
$f(x+h)\approx f(x)-\alpha f'(x)^2 \le f(x)$, so moving against the derivative decreases $f$ for small $\alpha$.
In several variables the same role is played by the gradient $\nabla f$ (the vector of partial derivatives).
""")
    st.subheader("2. The algorithm builds a sequence")
    st.latex(r"x_{k+1}=x_k-\alpha f'(x_k)\qquad\text{or}\qquad \mathbf{x}_{k+1}=\mathbf{x}_k-\alpha\nabla f(\mathbf{x}_k)")
    st.markdown(r"We study the sequence $(x_k)$. It **converges** to $x^*$ if for every $\varepsilon>0$ "
                r"there is $N$ with $|x_k-x^*|<\varepsilon$ for all $k\ge N$.")

    st.subheader("3. Worked example: f(x) = x\u00b2")
    st.latex(r"f'(x)=2x,\qquad x_{k+1}=(1-2\alpha)x_k\ \Rightarrow\ x_k=(1-2\alpha)^k x_0")
    st.markdown(r"""
This is a geometric sequence. **Guarantee:** $x_k\to 0$ for every starting point $x_0$ if and only if
$|1-2\alpha|<1$, i.e. $0<\alpha<1$. For $\alpha=1$ the sequence is $x_0,-x_0,x_0,\dots$ (oscillation, no limit);
for $\alpha>1$ it grows without bound (divergence).
""")
    a = st.slider("Try a learning rate \u03b1", 0.01, 1.3, 0.1, 0.01, key="th_alpha")
    x0 = st.number_input("x\u2080", value=10.0, key="th_x0")
    n = np.arange(0, 41)
    seq = (1 - 2 * a) ** n * x0
    factor = abs(1 - 2 * a)
    fig = go.Figure(go.Scatter(x=n, y=seq, mode="lines+markers"))
    fig.update_layout(height=320, xaxis_title="k", yaxis_title="x_k", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig)
    if factor < 1:
        st.success(f"|1 - 2\u03b1| = {factor:.3f} < 1: the sequence converges to 0.")
    elif np.isclose(factor, 1):
        st.warning("|1 - 2\u03b1| = 1: the sequence oscillates between x\u2080 and -x\u2080 (no limit).")
    else:
        st.error(f"|1 - 2\u03b1| = {factor:.3f} > 1: the sequence diverges.")

    st.subheader("4. Convergence is NOT automatic")
    st.markdown(r"""
Whether gradient descent converges depends on the function, the learning rate, the starting point
and the assumptions:
- **Learning rate:** if $\nabla f$ is Lipschitz with constant $L$ (and $f$ is convex), then
  $0<\alpha<2/L$ is the standard safe range. Larger $\alpha$ can oscillate or diverge.
- **Starting point:** $f(x)=x^4-3x^2+x$ has two minima; different starts reach different limits.
- **No minimum:** $f(x)=x^3-4x$ is unbounded below, so starting left of its local maximum the iterates run to $-\infty$.
  For $e^x$ the infimum $0$ is never attained: $f(x_k)\to 0$ but $x_k\to-\infty$.
- **Small gradient is not convergence of $x_k$:** $\nabla f(x_k)\to0$ does not by itself prove $x_k$ converges.
- **Saddle points:** on $x^2-y^2$ starting exactly on the x-axis converges to the saddle, but any tiny $y\neq0$ escapes.
""")
    st.subheader("5. Why continuity and differentiability matter")
    st.markdown(r"""
Gradient descent needs $f'$ (or $\nabla f$) to exist. For $f(x)=|x|$ there is no derivative at $0$.
At a discontinuity (for example near the asymptotes of $\tan x$) the local linear approximation fails,
so a step can jump to an unrelated value. The Function Explorer shows a warning when a typed function
contains such pieces.
""")
    st.subheader("6. Multivariable optimization")
    st.markdown(r"""
A **critical point** satisfies $\nabla f=0$. The Hessian $H$ classifies it: all eigenvalues positive gives a local
minimum, all negative a local maximum, mixed signs a saddle point. Near a minimum with eigenvalues
$\lambda_i$, plain gradient descent contracts the error by $\rho=\max_i|1-\alpha\lambda_i|$, and converges when $\rho<1$.
""")
    st.info("Limitation: the numerical tools here are for learning. Critical-point search only covers the plotting "
            "range and can miss points; convergence rates and the 2/L bound are local statements.")


# ============================================================== page 4
def page_tests():
    st.header("Test Cases")
    st.caption("These runs are computed live by the same code you use in the simulator.")
    cases = [
        ("A: x\u00b2, start 10, \u03b1 = 0.1 (expect convergence)", "x**2", ((-12, 12), None), [10.0], 0.1, 5000),
        ("B: x\u00b2, start 10, \u03b1 = 0.01 (expect slow convergence)", "x**2", ((-12, 12), None), [10.0], 0.01, 5000),
        ("C: x\u00b2, start 10, \u03b1 = 1.1 (expect divergence)", "x**2", ((-12, 12), None), [10.0], 1.1, 5000),
        ("D: x\u00b2, start 10, \u03b1 = 1.0 (oscillation)", "x**2", ((-12, 12), None), [10.0], 1.0, 200),
        ("E: x\u00b2 + 2y\u00b2, start (4, 3), \u03b1 = 0.1 (2D)", "x**2 + 2*y**2", ((-5, 5), (-5, 5)), [4.0, 3.0], 0.1, 5000),
        ("F: Rosenbrock, \u03b1 = 0.001 (hard valley)", me.PRESETS["Rosenbrock valley"]["expr"],
         ((-2, 2), (-1, 3)), [-1.2, 1.0], 0.001, 5000),
    ]
    rows = []
    for label, text, d, start, a, n in cases:
        f_ = me.with_minimizer(get_fn(text), d)
        p, inf = core.optimize(f_, start, a, n)
        fv, _, _, _, _ = core.metrics(f_, p)
        rho = core.theoretical_rate(f_, a)
        rows.append({
            "Test": label,
            "Result": core.status(inf),
            "Iterations": len(p) - 1,
            "Final point": fmt_point(p[-1]) if np.all(np.abs(p[-1]) < 1e6) else "unbounded",
            "Final f": f"{fv[-1]:.3g}" if np.isfinite(fv[-1]) and abs(fv[-1]) < 1e12 else "huge",
            "Theoretical \u03c1": f"{rho:.4f}" if rho is not None else "n/a",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("For x\u00b2 the theory says \u03c1 = |1 - 2\u03b1|: 0.8 (A), 0.98 (B), 1.2 (C), 1.0 (D). "
               "\u03c1 < 1 converges, \u03c1 = 1 oscillates, \u03c1 > 1 diverges.")
    st.markdown("**Input validation checks (also tested in the code):** \u03b1 = 0, negative \u03b1, "
                "tolerance \u2264 0, invalid expressions and unsafe text are all rejected with a message.")


# ============================================================== router
{"Function Explorer": page_explorer,
 "Gradient Descent Simulator": page_simulator,
 "Real Analysis Theory": page_theory,
 "Test Cases": page_tests}[page]()
