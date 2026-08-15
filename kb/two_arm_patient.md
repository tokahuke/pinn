# two_arm — the first-order patient correction

> STATUS 2026-08-15: the derivation below is sound and its payoff is real
> where the correction is applied PURELY, but the program it was built for --
> blending it into the graded law below the sampling floor -- was measured and
> REJECTED the same day. See kb/two_arm.md section 9 for the four targets tried
> and why no pointwise construction is gradeable there. What shipped is the
> self-similar clamp. Read this doc as mathematics, not as live guidance.
>
> The torch implementation (pinn/problems/two_arm/patient.py: Chebyshev
> remainder, 80 baked coefficients, ODE residual 6e-9 relative, double-backward
> safe) is DELETED, since nothing imported it. Section 4 below carries the
> coefficients and the method, so it is reconstructible from this doc alone.

The `tauhat -> 0` expansion of the two-arm premium past its startup term. Read
`kb/two_arm.md` section 8 first: notation, the similarity chart, and the proof
that the startup solution is the free-information envelope `nu` exactly.
Everything here is exact unless flagged. Implementation:
`pinn/problems/two_arm/patient.py`.

Short version. The correction `g1` exists, is a one-line quadrature after an
integrating factor, and drops the natural-units HJB residual from `O(u)` to
`O(sqrt(tauhat))` — measured 99.9x at `tauhat = 1e-3`, 999x at `1e-4`, and it
turns the degenerate vertex-commit policy into a real allocation. It is ALSO
positive, which the proven free-information bound says the true correction
cannot be (section 5). Both facts are true; they are about different norms.

## 1. The expansion and the order-e^s collection

The chart equation (two_arm section 8), on `z >= 0`, `s = log tauhat`:

    g_s + (1/2) g_zz + (z/2) g_z - (1/2) g = e^s (sqrt(g) + sqrt(g + z))^2

Write `L[g]` for the left side; `L` is the learning operator in the chart
(`L[g] = tauhat^(3/2) Lhat_ab[u]`). The startup solution `g0(z) = nu(-z, 1)`
kills `L` identically, which is what makes it an information martingale.

Expand `g = g0(z) + e^s g1(z) + O(e^(2s))`. The `s`-derivative of `e^s g1`
returns `+g1`, which flips the sign of the zeroth-order `-(1/2) g`:

    order 1:    (1/2) g0'' + (z/2) g0' - (1/2) g0 = 0          (startup, known)
    order e^s:  (1/2) g1'' + (z/2) g1' + (1/2) g1 = Q(z)

    Q(z) := ( sqrt(g0(z)) + sqrt(g0(z) + z) )^2
          = 2 g0 + z + 2 sqrt( g0 (g0 + z) )

VERIFIED as stated in the brief — the `+1/2` is not a typo, it is the whole
content of the collection. BC1 (`g_z(0, s) = -1/2` at every `s`) transcribes to
`g1'(0) = 0`. `Q > 0` everywhere: `Q(0) = 4 g0(0) = 4 phi(0) = 1.5958`,
`Q(z) -> z` as `z -> infinity`.

## 2. Homogeneous solutions — the brief's lead is wrong here

Multiply by 2: the operator is an exact derivative,

    g1'' + z g1' + g1 = (g1' + z g1)' = 2 Q

so the homogeneous equation is `y' + z y = const`:

    const = 0:  y1 = exp(-z^2/2)  (proportional to phi),  y1'(0) = 0
    const = 1:  y2 = exp(-z^2/2) int_0^z exp(w^2/2) dw,   y2'(0) = 1

`phi` does solve it (lead confirmed). The rest of that lead does not: `y2` is
the Dawson function up to scaling, `y2 = sqrt(2) F(z/sqrt(2))`, and it DECAYS
like `1/z` — measured `z y2(z)` = 1.082 at `z = 4`, 1.016 at `z = 8`. Nothing
grows like `exp(z^2/2)` here, so boundedness at infinity selects NOTHING. What
selects `y2` out is BC1, not decay.

The mode spectrum, for the record. Homogeneous solutions of the linearized PDE
of the form `e^(lambda s) psi(z)` need
`(1/2) psi'' + (z/2) psi' + (lambda - 1/2) psi = 0`; substituting
`psi = exp(-z^2/2) v` gives Hermite's equation with index `2 lambda - 2`, so the
decaying branch is `psi = exp(-z^2/2) He_(2 lambda - 2)(z)` and `psi'(0) = 0`
picks `lambda in {1, 2, 3, ...}`. Separately `lambda = 1/2, psi = 1` solves it
and satisfies BC1 but does not decay in `z`. Checked numerically to 1e-16 for
`lambda = 1/2, 1, 3/2, 2` and for `g0` at `lambda = 0`.

## 3. Solution by integrating factor

Integrate the exact derivative once; `g1'(0) = 0` sets the constant to zero:

    g1' + z g1 = 2 R(z),                    R(z) := int_0^z Q
    g1(z) = exp(-z^2/2) ( C2 + 2 int_0^z exp(w^2/2) R(w) dw ),   C2 = g1(0)

Split off the source's linear tail. `y_p = z` solves `y'' + z y' + y = 2z`
EXACTLY, so with `Q = z + q`,

    q(z) := 2 g0 + 2 sqrt( g0 (g0 + z) ) >= 0,     q ~ 2 sqrt(z g0) ~ e^(-z^2/4)
    g1(z) = z + h(z),      h' + z h = -1 + 2 S(z),     S(z) := int_0^z q
    h(z) = exp(-z^2/2) C2 + int_0^z exp(-(z^2 - w^2)/2) (2 S(w) - 1) dw

`h` is the whole numerical content and it is bounded. Constants:

    int_0^w g0 = -(w^2/2) Phi(-w) + (w/2) phi(w) + (1/2)(Phi(w) - 1/2)
    int_0^inf 2 g0 = 1/2                                      (exact)
    S(inf) = 1.8404892629554774                               (the sqrt half is
                                                               not elementary)
    P_INF := -1 + 2 S(inf) = 2.680978525910955

Far field: `h' + z h -> P_INF`, so `h ~ P_INF (1/z + 1/z^3 + 3/z^5 + ...)`,
i.e. `h = P_INF * (Dawson-type y2) + O(e^(-z^2/4))`, and

    g1(z) = z + P_INF / z + O(1/z^3)

The `1/z` series is divergent (coefficients `(2k-1)!!`), which is why the fit of
section 6 maps the whole half-line rather than matching to a truncated tail.

## 4. Which solution — the honest version

`C1` is fixed by BC1. `C2 = g1(0)` is NOT fixed by anything local: it multiplies
`exp(-z^2/2)`, which decays, so no far-field condition weaker than
exponential precision sees it. In PDE terms `C2 e^s exp(-z^2/2)` is the
`lambda = 1` mode of section 2 — a genuine homogeneous solution of the
linearized problem, decaying as `s -> -infinity` at exactly the order being
computed. Its amplitude is set by the global solution (the free boundary), not
by the inner ODE.

`patient.py` normalizes `C2 = 0`, i.e. `g1(0) = 0`: the premium ON THE RIDGE
gets no first-order correction. This is a choice, recorded as such. It costs
nothing that is graded — `C2` is an exact homogeneous solution at this order,
so every acceptance number in section 7 is `C2`-independent — and it keeps the
ridge identity `Lhat_ab[u]|_0 = 4 u|_0` (two_arm section 6) exact term by term.

The far-field condition, stated honestly: there is NO admissible-by-decay
solution. `g1 ~ z` for every choice of both constants, because the source's own
linear tail forces it. The inner expansion does not decay and is not supposed
to: `z -> infinity` at fixed `s` leaves the corridor, and the matching partner
is the free-boundary region, not zero.

## 5. The sign problem — the correction has the wrong sign, and that is real

With `C2 = 0`, `g1 >= 0` everywhere (minimum `1.4e-15` at `z = 0`, over 4801
points on `z in [0, 12]`), and `g1 ~ z` regardless of `C2`. So

    g = g0 + tauhat g1  >  g0

But the free-information envelope (kb/three_arm.md section 13, specialized to
one challenger) proves `u <= nu(-muhat, tauhat^(-1/2))`, which in the chart is
`g(z, s) <= g0(z)` at EVERY `s`. The true first correction must be `<= 0`.
No `C2` repairs this: `C2` only adds `C2 exp(-z^2/2)`, dead past `z ~ 3`, while
the violation grows like `z`. Measured excess `tauhat g1 / g0`:

    z        tauhat=1e-3   1e-4      1e-6
    0.5      1.90e-3       1.90e-4   1.90e-6
    1.0      1.54e-2       1.54e-3   1.54e-5
    1.5      7.71e-2       7.71e-3   7.71e-5
    2.0      3.56e-1       3.56e-2   3.56e-4
    2.5      1.77e+0       1.77e-1   1.77e-3

So the regular power series in `tauhat` is a correct FORMAL solution of the PDE
but is not the asymptotics of the true solution: a term the local analysis
cannot produce is missing, and it is negative and at least as large.

UNVERIFIED, scaling only — where it comes from. The true solution must vanish at
the free boundary `Z(s)`, where `g0` is already down to `~ e^s Z`. Propagate
that deficit inward: writing `g = g0 + delta` with `delta` solving the
linearized homogeneous equation, `delta = e^(s/2) E[ e^(-s_hit/2) delta_hit ]`,
and `z` drifts as `dz = (z/2) ds + dW`, so `e^(s_hit/2) ~ e^(s/2) Z/z`. With
`|delta_hit| ~ e^(s_hit) Z` this gives

    delta ~ - e^s Z(s)^2 / z ~ - tauhat * 2 log(1/tauhat) / z

negative, and one log factor ABOVE `tauhat g1`. Consistent with the direct
argument that the premium is below free information because learning takes
time: reaching the boundary from `z` costs `Delta s ~ 2 log(Z/z)` e-folds of
`tauhat`, and precision advances at `dtauhat/dt' = alpha(1 - alpha) <= 1/4` in
discount-scaled time `t' = rho t`, so that costs `Delta t' >= 4 Delta tauhat` of
discounting.

What this means in practice: `tauhat g1` fixes the EQUATION (section 7) and does
not improve the sup-norm approximation the envelope measures. Do not use it as
an upper bound, and do not wire it into an architecture whose invariant is
`u <= envelope` without a guard — at `tauhat = 1e-3, z = 2` it is 36% over.

## 6. Implementation

`g1(z) = z + h(z)`, `h` by Clenshaw on a Chebyshev series in

    y = (z - MAP) / (z + MAP),      MAP = 3.0

which carries `[0, infinity)` onto `[-1, 1]`, so the fit covers the whole
half-line and `h(infinity) = 0` is a node rather than an extrapolation. 80
coefficients, generated offline from an RK4 solve of the section-3 system
(`S' = q`, `h' = -z h - 1 + 2 S`, both from 0, step 2.5e-4) evaluated at the
Chebyshev nodes, with `h = P_INF * Dawson` used for nodes past `z = 13` where
`q < 1e-16`. Forward integration is stable: the homogeneous solution decays.

Max `|h - fit|` over `z in (0, 100]` against the reference solve:

    coefficients   MAP=1.5   2.0       2.5       3.0       4.0
    32             9.9e-6    5.4e-6    3.0e-6    2.6e-6    2.5e-6
    48             7.8e-8    2.7e-8    1.1e-8    7.1e-9    2.6e-9
    64             6.5e-10   8.0e-11   3.5e-11   1.2e-11   5.1e-12
    80             --        7.6e-13   1.9e-13   1.2e-13   6.3e-14

Coefficients fall ~0.69 per order; 96 gives 5.6e-14, the float64 floor of the
reference solve, so 80 is where the fit stops paying. `MAP` between 2 and 4 is
within a factor 10 at fixed degree and irrelevant at 80.

Alternatives rejected:

- Closed form: `int sqrt(g0 (g0 + z))` has no elementary antiderivative (the
  `2 g0` half does, section 3), so `S` blocks it.
- Nested Gauss-Legendre in torch (the semi-closed option): two levels, ~1000
  evaluations of `q` per point, and the outer kernel `exp(-z^2(1 - t^2)/2)`
  needs the node count to grow with `z`. Too much machinery for a function of
  one variable.
- Split `h = P_INF * D(z) + r` with `D` the Dawson function: exact tail, but
  torch has no Dawson and the standard rational approximations are PIECEWISE —
  a jump in the second derivative, which is precisely what the loss
  differentiates.
- Chebyshev on a truncated `[0, Z]` plus an asymptotic continuation: needs a
  blend, hence either a kink or another tuned scale.
- Table or spline lookup: not twice differentiable at the knots.

## 7. Verification (all asserted in the module self-check)

`poetry run python -m pinn.problems.two_arm.patient`, float64, derivatives by
`torch.autograd` with `create_graph=True` (which is also the double-backward
proof; the self-check backpropagates through `g1''`).

ODE residual `(1/2) g1'' + (z/2) g1' + (1/2) g1 - Q`, relative to `Q`, on 241
points over `z in [0, 6]`:

    max 6.1e-9, at z = 0 exactly (Chebyshev derivative error is worst at the
    mapped endpoint); 1.7e-11 over the rest of the grid.

Boundary and normalization:

    g1'(0)  = -1.2e-11        (BC1)
    g1(0)   =  1.4e-15        (the C2 = 0 normalization)
    g1''(0) =  3.1915382626   against 2 Q(0) = 3.1915382432, 6e-9 relative

Far field: `z h(z)` = `P_INF` to 2e-3 at `z = 50, 200, 1000`.

The payoff. Both premia are run through `model.DimensionlessValueFunction.
hamiltonian` and graded exactly as `loss.pde_loss` does — natural units, the
chart's `tauhat^(3/2)` divided back out. `u_bare = nu(-muhat, tauhat^(-1/2))`,
`u_corrected = u_bare + sqrt(tauhat) g1(z)`. RMS over 161 points:

    z <= 2            bare        corrected    ratio
    tauhat 1e-3       5.278e+0    5.283e-2      99.9
    tauhat 5e-4       7.465e+0    3.736e-2     199.8
    tauhat 1e-4       1.669e+1    1.671e-2     999.0

    z <= 4            bare        corrected    ratio
    tauhat 1e-3       3.766e+0    9.604e-2      39.2
    tauhat 5e-4       5.326e+0    6.791e-2      78.4
    tauhat 1e-4       1.191e+1    3.037e-2     392.2

The scalings are exact: bare `~ tauhat^(-1/2)` (it IS the premium — `L` vanishes
on `g0`, the maximization collapses to `alpha = 1`, and the residual is `u`),
corrected `~ tauhat^(+1/2)`, so the ratio is `~ 1/tauhat`, one order per decade.
The `z <= 4` column is lower only because that grid runs past the free boundary;
see section 8.

Policy, at `muhat = 0` (natural units):

    tauhat    l_ab bare   l_ab corrected   4 u(0)     alpha bare   alpha corrected
    1e-2      2.8e-14     15.9577          15.9577    0.000        0.500
    1e-3      0.0e+00     50.4627          50.4627    0.000        0.500
    1e-4      0.0e+00     159.5769         159.5769   0.000        0.500

The bare `l_ab` is zero to float64 because `nu` is an information martingale;
the Hamiltonian is then LINEAR in `alpha` and the max sits on a vertex — commit
on zero evidence, which is the failure this exercise was aimed at. The corrected
`l_ab` reproduces the ridge identity `Lhat_ab[u]|_0 = 4 u|_0` to all digits
printed, and the policy at `tauhat = 1e-3` runs 0.500, 0.653, 0.783, 0.878,
0.939 at `z = 0, 0.5, 1, 1.5, 2` against a flat 1.000 for the bare envelope.

## 8. Validity range, and the free boundary as a bonus

The expansion holds while `tauhat g1 << g0`, i.e. while `tauhat z << phi(z)/z^2`.
Two measurements of where it ends, against the free-boundary locations measured
on trained nets (two_arm section 6: `muhat_b` 3.783 at `tauhat = 1e-1`, 21.13 at
`1e-2`, i.e. `z_b = muhat_b sqrt(tauhat)` = 1.196 and 2.113):

    tauhat    residual crossover   Phi^-1(1 - tauhat)   measured z_b
    1e-1      0.883                1.282                1.196
    1e-2      1.587                2.326                2.113
    1e-3      2.330                3.090                --
    5e-4      2.540                3.291                --
    1e-4      2.994                3.719                --

Column 2 is where the corrected residual stops beating the bare one pointwise.
Column 3 is smooth pasting at first order: `g_z = 0` reads
`Phi(-z) = tauhat g1'(z)`, and `g1' -> 1`, so `z_b ~ Phi^-1(1 - tauhat)` — i.e.
`Z(s) = Phi^-1(1 - e^s)`, growing like `sqrt(2 log(1/tauhat))`. The two bracket
the measured boundary at both decades where a measurement exists. Solving the
pasting condition with the true `g1'` instead of 1 gives 0.858 and 2.332 — no
better, which is expected: at the boundary the expansion has already broken
down (`tauhat g1` is several times `g0` there), so this is an ESTIMATE with a
~10-25% bracket, not a derivation. Value matching (`g = 0`) is not satisfied
there at all.

Beyond the crossover the correction is worse than nothing: it is a premium of
size `tauhat muhat` in a region where the true premium is exponentially small,
and the residual it leaves is `sqrt(tauhat) z` while the bare one is
`g0(z)/sqrt(tauhat)`. Anything that uses `patient.premium` outside the corridor
needs its own cut-off, and the cut-off is the free boundary, which is exactly
the object the whole two_arm problem is about.
