# Three-Armed Bayesian Allocation (ABC) — HJB Problem

Status: part 1 (observation model + belief dynamics), transcribed from a prior
session 2026-08-04 and re-verified in-session. HJB, symmetry reduction,
dimensionless form, and boundary structure still to come.

## 0. Notation

Follows docs/two_arm.md conventions (ASCII, uppercase dimensional, hats for
dimensionless once they exist). Arms: control `a`, treatments `b`, `c`. Traffic
split `alpha = (alpha_a, alpha_b, alpha_c)`, nonnegative, summing to 1; write
`alpha_bc = (alpha_b, alpha_c)^T`. Contrasts (what we actually care about):

    theta_b = mu_b - mu_a,    theta_c = mu_c - mu_a

## 1. Observation model

Over dt, arm totals are `A ~ N(alpha_a mu_a dt, alpha_a sigma^2 dt)` etc. The
contrast estimates `y_b = B/alpha_b - A/alpha_a`, `y_c = C/alpha_c - A/alpha_a`
are unbiased for `(theta_b, theta_c)` with per-unit-time noise covariance

    Sigma_obs(alpha) = sigma^2 [ 1/alpha_a + 1/alpha_b        1/alpha_a          ]
                               [ 1/alpha_a                    1/alpha_a + 1/alpha_c ]

The shared control arm is the correlation.

## 2. Information rate

Inverting (cross terms collapse via
`1/(alpha_a alpha_b) + 1/(alpha_a alpha_c) + 1/(alpha_b alpha_c)
= 1/(alpha_a alpha_b alpha_c)`, using `sum alpha = 1`):

    G(alpha) = Sigma_obs^{-1}
             = (1/sigma^2) [ alpha_b (1 - alpha_b)    -alpha_b alpha_c      ]
                           [ -alpha_b alpha_c          alpha_c (1 - alpha_c) ]
             = (1/sigma^2) ( diag(alpha_bc) - alpha_bc alpha_bc^T )

the multinomial covariance matrix. `alpha_a` has vanished from the entries but
not the geometry:

    det G = alpha_a alpha_b alpha_c / sigma^4

so information degenerates on the ENTIRE boundary of the simplex: starve any
arm, including control, and one direction of learning dies. (Verified by direct
inversion in-session. Consistency: `alpha_c = 0` collapses `G` to the scalar
`alpha_b (1 - alpha_b)/sigma^2` — exactly the two-arm learning rate.)

## 3. Belief state and dynamics

Gaussian posterior on `(theta_b, theta_c)`: mean `m = (m_b, m_c)`, precision

    T = [ tau_bb  tau_bc ]
        [ tau_bc  tau_cc ]

Kalman-Bucy filtering:

    dT/dt = G(alpha)                    (deterministic)
    dm    = martingale with covariance rate  M(alpha, T) := T^{-1} G(alpha) T^{-1}

Five states: `(m_b, m_c, tau_bb, tau_bc, tau_cc)`. Control on the simplex

    Delta = { alpha_b, alpha_c >= 0,  alpha_b + alpha_c <= 1 }

(`alpha_a = 1 - alpha_b - alpha_c` implied.)

## 4. Fixed-policy PDE

Compact form, for a given policy `alpha(m, T)` (reward flow `alpha_b m_b +
alpha_c m_c`; drift from dT/dt = G; diffusion from the mean's covariance rate):

    rho V = alpha_b m_b + alpha_c m_c
          + (1/2) tr( M(alpha, T) D2_m V ) + sum_ij G_ij(alpha) dV/dT_ij

with `M = T^{-1} G T^{-1} = (adj T) G (adj T) / (det T)^2`,
`adj T = [tau_cc, -tau_bc; -tau_bc, tau_bb]`, `D = det T = tau_bb tau_cc - tau_bc^2`.

Fully explicit (writing `gb = alpha_b (1 - alpha_b)`, `gc = alpha_c (1 - alpha_c)`,
`g = alpha_b alpha_c`):

    rho V = alpha_b m_b + alpha_c m_c

      + [ gb tau_cc^2 + 2 g tau_bc tau_cc + gc tau_bc^2 ]
        / (2 sigma^2 D^2)                                   * d2V/dm_b^2

      - [ gb tau_bc tau_cc + g (tau_bb tau_cc + tau_bc^2) + gc tau_bb tau_bc ]
        / (sigma^2 D^2)                                     * d2V/dm_b dm_c

      + [ gb tau_bc^2 + 2 g tau_bb tau_bc + gc tau_bb^2 ]
        / (2 sigma^2 D^2)                                   * d2V/dm_c^2

      + (gb / sigma^2) dV/dtau_bb
      - (g  / sigma^2) dV/dtau_bc
      + (gc / sigma^2) dV/dtau_cc

Structure worth registering (all verified in-session against M = SGS/D^2):

- No first-order terms in `m` — the posterior mean is a martingale; all
  m-dependence flows through the reward and the second derivatives.
- The three m-diffusion numerators are the quadratic form `(adj T) G (adj T)`:
  three shared ingredients `gb, g, gc` paired with tau-monomials. The `b <-> c`
  relabel maps the first diffusion numerator to the third and fixes the middle
  one — the relabeling symmetry visible at the PDE level.
- Two-arm collapse: `alpha_c = 0`, diagonal prior in c => the b-block reduces to
  `gb/(2 sigma^2 tau_bb^2) V_mbmb + gb/sigma^2 V_taubb`, i.e. docs/two_arm.md
  section 2 exactly.
- The HJB is this with `max over alpha in Delta` of the right side.

## 5. Exploration premium (derived in-session 2026-08-04; exact)

Commit value (attainable: any pure alpha makes G = 0, so beliefs freeze):

    C(m) = max(0, m_b, m_c) / rho

Define the exploration premium `U := V - C`. The generator annihilates C off
its kink set (no T-dependence kills the drift term; piecewise-linearity in m
kills the diffusion term; there is no first-order m-term by the martingale
property). Substituting V = C + U into the section-4 PDE:

    rho U = -R(alpha, m)
          + (1/2) tr( M(alpha, T) D2_m U ) + sum_ij G_ij(alpha) dU/dT_ij

    R(alpha, m) := max(0, m_b, m_c) - (alpha_b m_b + alpha_c m_c)  >= 0

R is the instantaneous commit-regret of the split alpha (a convex combination
of (0, m_b, m_c) never beats the max; R = 0 iff all traffic on a currently
best arm). U bleeds regret and earns nothing else: every unit of premium is
paid for by the learning terms. Same coefficients as section 4, V -> U, reward
replaced by -R.

Structure:

- `U >= 0`; `U = 0` on the commit regions. Two-arm collapse: alpha_c = 0,
  m_b > 0 gives R = (1 - alpha_b) m_b — docs/two_arm.md exactly.
- Kink set (the BC1 generalization): U inherits minus C's gradient jumps on
  the three argmax-tie ridges {m_b = m_c > 0}, {m_b = 0 >= m_c},
  {m_c = 0 >= m_b}, meeting at the triple point m = 0. V is smooth across
  them; the kinks live in C.
- Relabel symmetry: U invariant under
  (m_b, m_c, tau_bb, tau_bc, tau_cc) -> (m_c, m_b, tau_cc, tau_bc, tau_bb).

## 6. Symmetry: what we get and what we don't (2026-08-04, sympy-checked)

- Relabeling the three arms (6 possible relabels, control included) leaves the
  problem unchanged. In contrast coordinates each relabel is a small linear
  change of (m, T) plus a renaming of the controls, and the premium U is
  EXACTLY the same before and after. Consequence: it is enough to solve on one
  sixth of the state space (say "control looks best, b looks second best") and
  copy everywhere else by relabeling. The three kink surfaces of section 5 are
  also copies of one relation.
- No continuous symmetry exists. Checked by direct computation: no smooth
  change of variables removes a state. The problem really is 5-dimensional.

## 7. Better coordinates: information per pair (2026-08-04, sympy-checked)

- Learning happens pairwise. Each pair of arms (a-b, a-c, b-c) accumulates
  its own pairwise precision, filled at rate alpha_x * alpha_y / sigma^2 when
  running split alpha. Track the pairwise precisions I = (I_ab, I_ac, I_bc)
  instead of the precision matrix:
  - T is prior plus a fixed linear function of I (nothing lost).
  - The reachable states are exactly {I_ab, I_ac, I_bc >= 0} — a box corner,
    easy to sample correctly. In tau-language the same set is a skewed cone
    that is easy to sample wrongly (about 4x wasted volume).
  - Arm relabels just shuffle the three I's. The hidden symmetry becomes
    visible bookkeeping.
  - The learning part of the PDE splits into three pieces, one per pair,
    each shaped like the two-arm learning term.
- The best-split problem at each point (max over the allowed alphas) has seven
  cases — inside, three edges, three corners — each in closed form. Keep the
  max explicit, as in two_arm.

## 8. Known limits to anchor the solve

- Commit regions: U = 0 exactly.
- One arm hopeless (its mean -> -infinity): the problem becomes the SOLVED
  two-arm problem for the remaining pair, using that pair's effective
  precision. By symmetry this covers all three "one arm is dead" limits.
  VERIFY the convergence rate before using as a hard constraint; safe as
  far-field boundary data.
- One contrast perfectly known (its precision -> infinity): also collapses to
  two-arm-shaped problems. VERIFY the exact form.

## 9. Literature check

The PDE-limit approach to bandit experiments is established (Adusumilli,
Econometrica 2025, arXiv:2112.06363; Wager and Xu, Diffusion Asymptotics for
Sequential Experiments). Nobody appears to treat the shared-control
correlation or to reduce the 5-D state; no index/Gittins shortcut exists here
(the shared control makes arms dependent).

## 10. The inner max: seven candidates (derived in-session 2026-08-04; exact)

Write b = alpha_b, c = alpha_c. Pairwise learning numbers at a state, one per
pair, with directions v_ab = e_b, v_ac = e_c, v_bc = e_b - e_c and
w = T^{-1} v:

    L_v[U] = (1/sigma^2) [ (1/2) w^T (D2_m U) w + (v-form on dU/dT) ]

    v-form on dU/dT:  a-b: dU/dtau_bb
                      a-c: dU/dtau_cc
                      b-c: dU/dtau_bb + dU/dtau_cc - dU/dtau_bc

Substituting alpha_a = 1 - b - c into the section-5 equation and dropping the
-max(0, m_b, m_c) constant (cannot move the argmax), the Hamiltonian is a
quadratic on the triangle:

    H(b, c) = b (m_b + L_ab) + c (m_c + L_ac)
            - L_ab b^2 - L_ac c^2 + (L_bc - L_ab - L_ac) bc

The max of a quadratic over a triangle sits at an interior stationary point,
an edge's stationary point, or a vertex — nowhere else. Seven candidates, all
closed form:

    1.   interior: solve grad H = 0 (2x2 linear; safe-denominator its det
         4 L_ab L_ac - (L_bc - L_ab - L_ac)^2); feasible iff b, c >= 0, b + c <= 1
    2-4. edges — each one IS the two-arm clamped vertex:
         c = 0:      b* = clamp( (m_b + L_ab) / (2 L_ab), 0, 1 )
         b = 0:      c* = clamp( (m_c + L_ac) / (2 L_ac), 0, 1 )
         b + c = 1:  s* = clamp( 1/2 + (m_b - m_c) / (2 L_bc), 0, 1 ),
                     (b, c) = (s*, 1 - s*)
    5-7. vertices: H(0,0) = 0, H(1,0) = m_b, H(0,1) = m_c

Implementation rule (the two-arm lesson, generalized): do NOT case-split on
concavity — evaluate H at every feasible candidate (infeasible -> -inf) and
take the max. Saddles, convex edges, and degenerate L's are then handled
automatically, and the whole thing vectorizes: 7 candidates x 1 evaluation x
1 max per collocation point.

Checks: the three edge formulas are the three two-arm subproblems (the a-b
edge collapse of section 4 verified earlier); arm relabels permute the
candidate list, so relabeling inputs must leave max H unchanged — a free
implementation test.

## 11. Units (resolved 2026-08-04)

Working with rho = sigma = 1 IS the dimensionless form: two parameters, two
independent units (time, mean-units), so no dimensionless group survives
(same counting as two_arm, and every arm pair scales exactly like the
two-arm problem it collapses to). The code's rho = sigma = 1 convention loses
no generality.

Readout dictionary for a real problem (only needed at deployment/benchmark,
never during training):

    mhat_i     = m_i / (sigma sqrt(rho))     # both posterior means
    tauhat_xy  = rho sigma^2 tau_xy          # all three precision entries
    v_real     = (sigma / sqrt(rho)) v_hat   # values back to real units

The prior enters as the starting point (mhat_0, That_0), not as a PDE
parameter -- as in two_arm.

## 12. The wedge and its wall conditions (derived in-session 2026-08-04; exact)

Fundamental wedge: {m_c <= m_b <= 0} ("control best, b second"). Inside it the
commit envelope is 0, so v = u; the six relabels tile the rest of the space.
Two walls, different characters. Notation: D- is the wedge-side normal
derivative dU/dm_b on a wall, T_c the tangential dU/dm_c on the wall;
J = [[-1, 0], [-1, 1]] is the a<->b relabel, P the b<->c swap of tau entries.

Wall 1, {m_b = 0} (control-b tie), the BC1 generalization. Smoothness of
V = C + U forces a unit kink (outside normal derivative = D- - 1); the a<->b
relabel U(m_b, m_c, T) = U(-m_b, m_c - m_b, J^T T J) turns this into a
wedge-side-only condition linking each wall point with its mirror T' = J^T T J:

    D-(m_c, T) + D-(m_c, T') + T_c(m_c, T') = 1

- Kills the never-explore solution: U = 0 gives 0 = 1. This is the
  degeneracy-breaking condition (the doc section on the pde loss depends on it).
- Two-arm collapse (arm c decoupled): T' = T, T_c = 0, so D- = 1/2 — exactly
  docs/two_arm.md BC1 seen from the control-best side.
- Localizes to 2 D- + T_c = 1 on the slice where T' = T.

Wall 2, {m_b = m_c} (b-c mirror), pure symmetry. Both sides of this wall have
C = 0 (the b-c envelope kink lives at m_b = m_c > 0, off the wedge), so the
condition is homogeneous Neumann via the swap, with d_n = d/dm_b - d/dm_c:

    d_n U(m, T) + d_n U(m, P T P) = 0

- The evenness analogue: shapes the solution, does not break degeneracy.
- Locally d_n U = 0 on the tau_bb = tau_cc slice.

NEW vs two_arm: both conditions are nonlocal — the relabels fix the walls as
sets but move the precision matrix, so each condition pairs a wall point with
its mirror wall point. Loss terms are pair-sampled: draw wall states, map to
mirrors (affine, already tested in the loss S3 check), evaluate both, penalize
the combination.

## 13. The premium envelope (derived and verified 2026-08-05)

Why an envelope: the premium net's tanh response is bounded (roughly by 1), so
the multiplier in u = envelope * N(features) must bound the TRUE premium from
above wherever the net is supposed to reach it. Too tight is fatal (the net
cannot represent the solution); too loose only wastes resolution. Two_arm's
envelope was guessed and patched; this one is derived.

Step 1 — the free-information bound (proved, pathwise). At every instant, any
split earns reward alpha_b theta_b + alpha_c theta_c: a weighted average of
(0, theta_b, theta_c), so it never beats max(0, theta_b, theta_c) -- on every
path, at every time, whatever the policy learns later. Discounting and taking
posterior expectations (rho = 1 units):

    V(m, T) <= E[ max(0, theta_b, theta_c) ]     (expectation under N(m, T^-1))

Subtract the commit value; on the wedge it is 0, so

    U(m, T) <= E[ max(0, theta_b, theta_c) ]

Interpretation: the premium can never exceed the value of being handed the
truth for free.

Step 2 — split the max (proved). Pointwise max(0, x, y) <= max(0, x) +
max(0, y), and the expected positive part of a Gaussian is the standard
formula

    nu(m, sd) = m Phi(m/sd) + sd phi(m/sd)       # E[max(0, X)], X ~ N(m, sd^2)

(Phi, phi the normal cdf and density). With the MARGINAL standard deviations
sd_b = precision_b^{-1/2}, sd_c = precision_c^{-1/2} (Schur complements,
section on models -- not the raw tau diagonals):

    U(m, T) <= nu(m_b, sd_b) + nu(m_c, sd_c)

This is the envelope. In code: exp(log_scale) times the nu-sum, log_scale
initialized to 0 -- at scale exactly 1 the bound is a theorem, so the whole
true premium starts inside the net's range.

Verified numerically (200 folded Sobol states x 40k Monte Carlo draws each,
plus the two_arm champion in the a-b far field):

- The chain MC free-info <= nu-sum holds everywhere (no violations in 40k
  pointwise draws).
- CORRECTED 2026-08-05: the original reading here ("nearly tight near the
  triple point, median slack 1.01x") was an artifact of the measurement's
  sampling, dominated by states with means a standard deviation or more below
  zero (where one nu term vanishes and the sum trivially approaches the
  bound). At the ACTUAL triple point m = 0 the nu-sum slack over the
  free-info bound is 1.17x uncorrelated, growing to 1.87x at correlation
  0.99 (MC, 2e7 draws): the max-splitting inequality of step 2 is strict
  wherever both challengers are live. Consequence for the model: the
  response factor must learn ~0.85 (uncorrelated) down to ~0.53 (high
  correlation) at the startup triple point, NOT 1. The exact startup
  solution is E[max(0, theta_b, theta_c)] itself (section 14); the nu-sum
  solves the startup interior equation but misses the control wall by
  exactly Phi(z_c').
- Far field: dominates the trained two_arm premium throughout the corridor
  (min slack 1.9x at tauhat = 0.2, growing like sqrt(tauhat)) -- the same
  safe-direction mismatch two_arm's envelope had.
- The cruder m = 0 version of the bound, (sd_b + sd_c)/sqrt(2 pi), is
  REJECTED: it never decays as the treatments become hopeless (median 10x
  looser, unbounded in the deep wedge). nu decays correctly in every far
  field.

precision_bc is NOT in the envelope, now confirmed rather than argued: the
bound contains only the two control-contrast marginals, and correlation
enters E[max] only by shrinking it. Knowledge of b-vs-c alone cannot sustain
a premium when both treatments are hopeless.

## 14. Self-similar coordinates (implementation chart; derived and verified 2026-08-05)

The dimensionless chart of sections 0-13 remains the reference frame for all
thinking; this section exists to justify ONE line of code (the residual
weight in `loss.py`) and to record what the derivation taught us. Derivation
by agent, every identity verified in float64 against the raw-coordinate
implementation (128 wedge states, max relative error $2.7\times10^{-13}$).

**Motivation.** The training cloud spans many decades of information scale
$S = \sqrt{\det T}$, and the residual $v - \max H$ has a natural magnitude
that scales like $S^{-3/2}$: two states with the *same shape error* -- same
lead in units of their own uncertainty, same precision shape -- produce
residuals differing by orders of magnitude for pure bookkeeping reasons. An
unweighted mean-of-squares lets the low-information decades shout down
everything else; a per-point "relative residual" hack divides by a noisy
local estimate. What we want is to measure every state's error against its
own scale's natural units -- and we suspect (from two_arm's near
self-similarity: the same tent shape at every scale) that a clean change of
units exists.

**Method.** Change variables so that each quantity is measured in the units
its own state defines: means in posterior standard deviations (z-scores),
the premium likewise, the information scale on a log clock, and the two
leftover degrees of freedom of $T$ as scale-free shapes. Then push the
section 4 PDE through the chain rule and see what the scale factors do.
This is a units change, not new physics. Sections 14.1-14.2 give the two
maps; 14.3 the transformed equation.

**What the math answers back** (the payoff, in decreasing order of surprise):

1. *The stiffness was notation.* Every derivative term of the transformed
   PDE has bounded, $O(1)$ coefficients over the entire state space; the
   whole scale dependence collects into a single factor $e^{s}$ multiplying
   only derivative-free terms. The $1/\det T$ that made the raw equation
   stiff across nine decades never touches a derivative. The one genuine
   scale dependence that survives is the physics: how expensive information
   is relative to patience (the explore/commit crossover).
2. *The transformation is a weight, not code.* Point by point,
   $\text{residual}_{\text{sim}} = S^{3/2}\,\text{residual}_{\text{raw}}$.
   So grading the residual in the good units requires no transformed
   operators in code -- one multiplication by the known scaling law, which
   also retires the old guessed relative-residual scale (the weight is its
   exact replacement).
3. *The symmetry lives in the shapes.* Both wall surfaces, both wall
   conditions, and both mirror maps are independent of $s$; the mirrors are
   transpositions of the three pairwise variances (the treatment mirror is
   just $r \to -r$). Scale and symmetry fully decouple.
4. *The startup limit is exactly solvable -- and it corrects us.* As
   $s \to -\infty$ the $e^{s}$ source dies and the premium must be a belief
   martingale; the exact solution is $E[\max(0,\theta_b,\theta_c)]$,
   satisfying interior and both walls. The nu-sum envelope solves the
   interior but fails the control wall by exactly $\Phi(z_c')$ -- which is
   how we caught the section 13 tightness error.
5. *The limit of the trick.* One scalar clock cannot flatten the shape
   directions: pairwise variances still diverge as $k \to 1$ or
   $|r| \to \infty$. Normalizing by half their sum ($c$ below) spreads the
   footing across shapes too and leaves a single scalar clock
   $e^{s}/c = \det T/(I_{ab}+I_{ac}+I_{bc})$.

### 14.1 Map: dimensionless $\to$ self-similar

With $T = \begin{pmatrix}\tau_{bb} & \tau_{bc}\\ \tau_{bc} & \tau_{cc}\end{pmatrix}$
and the marginal (Schur) precisions $p_b, p_c$:

$$
S=\sqrt{\det T},\qquad s=\log S,\qquad
k=\frac{-\tau_{bc}}{\sqrt{\tau_{bb}\,\tau_{cc}}},\qquad
r=\tfrac12\log\frac{\tau_{bb}}{\tau_{cc}}
$$

$$
p_b=\frac{\det T}{\tau_{cc}},\qquad p_c=\frac{\det T}{\tau_{bb}},\qquad
z_b=m_b\sqrt{p_b},\qquad z_c=m_c\sqrt{p_c}
$$

$$
g(z_b,z_c,r,k,s)\;=\;S^{1/2}\,u(m_b,m_c,T)
$$

$S$ is the overall information scale ($\det T$ is the exact S3 invariant:
relabels act by unit-determinant congruence), $k$ the posterior correlation
of the two contrasts ($k\in[0,1)$ on the wedge), $r$ the precision asymmetry,
$z$ the marginal z-scores.

### 14.2 Map: self-similar $\to$ dimensionless

$$
\tau_{bb}=\frac{S\,e^{r}}{\sqrt{1-k^2}},\qquad
\tau_{cc}=\frac{S\,e^{-r}}{\sqrt{1-k^2}},\qquad
\tau_{bc}=\frac{-S\,k}{\sqrt{1-k^2}}
$$

$$
m_b=\frac{z_b\,e^{-r/2}}{S^{1/2}\,(1-k^2)^{1/4}},\qquad
m_c=\frac{z_c\,e^{r/2}}{S^{1/2}\,(1-k^2)^{1/4}},\qquad
u=S^{-1/2}\,g
$$

(Check: $\tau_{bb}\tau_{cc}-\tau_{bc}^2 = S^2$, and
$p_b = S\,e^{r}\sqrt{1-k^2}$, consistent with 14.1.)

### 14.3 Fixed-policy HJB in self-similar coordinates

On the wedge the commit value is $0$, so $v = u$. Dimensionless pairwise
variances (posterior variances of the three pair contrasts, in units of $1/S$):

$$
V_{ab}=\frac{e^{-r}}{\sqrt{1-k^2}},\qquad
V_{ac}=\frac{e^{r}}{\sqrt{1-k^2}},\qquad
V_{bc}=\frac{2(\cosh r-k)}{\sqrt{1-k^2}}=V_{ab}+V_{ac}-\frac{2k}{\sqrt{1-k^2}}
$$

The section 4 fixed-policy PDE, multiplied through by $S^{3/2}$, becomes

$$
e^{s}\,g
= e^{s}\!\left(\alpha_b\sqrt{V_{ab}}\,z_b+\alpha_c\sqrt{V_{ac}}\,z_c\right)
+\alpha_a\alpha_b\,\frac{V_{ab}}{2}\,M_{ab}[g]
+\alpha_a\alpha_c\,\frac{V_{ac}}{2}\,M_{ac}[g]
+\alpha_b\alpha_c\,\frac{V_{bc}}{2}\,M_{bc}[g]
$$

with the three learning operators (the dimensionless learning numbers are
$l_p = S^{-3/2}\,\tfrac{V_p}{2}\,M_p[g]$):

$$
M_{ab}[g]=g_{z_bz_b}+2k\,g_{z_bz_c}+k^2 g_{z_cz_c}
+z_b g_{z_b}+k^2 z_c g_{z_c}
+(1-k^2)\,g_r-k(1-k^2)\,g_k+g_s-\tfrac{1}{2}g
$$

$$
M_{ac}[g]=k^2 g_{z_bz_b}+2k\,g_{z_bz_c}+g_{z_cz_c}
+k^2 z_b g_{z_b}+z_c g_{z_c}
-(1-k^2)\,g_r-k(1-k^2)\,g_k+g_s-\tfrac{1}{2}g
$$

$$
M_{bc}[g]=P_b\,g_{z_bz_b}+Q\,g_{z_bz_c}+P_c\,g_{z_cz_c}
+P_b z_b g_{z_b}+P_c z_c g_{z_c}
+R_r\,g_r+R_k\,g_k+g_s-\tfrac{1}{2}g
$$

$$
P_b=\frac{e^{-r}-2k+k^2e^{r}}{2(\cosh r-k)},\qquad
P_c=\frac{e^{r}-2k+k^2e^{-r}}{2(\cosh r-k)},\qquad
Q=-\frac{1+k^2-2k\cosh r}{\cosh r-k}
$$

$$
R_r=-\frac{(1-k^2)\sinh r}{\cosh r-k},\qquad
R_k=\frac{(1-k^2)(1-k\cosh r)}{\cosh r-k}
$$

Bookkeeping details (the narrative above is the argument; these are the
numbers behind it):

- Coefficient bounds over the whole wedge: $P_b, P_c \in [0,1]$,
  $Q \in [-1,2]$, $R_r \in [-1,1]$, $R_k \in [-0.385, 1]$; the $M_{ab}$,
  $M_{ac}$ coefficients are $1, 2k, k^2, 1-k^2, k(1-k^2)$.
- The full HJB adds $\max_{\alpha \in \Delta}$: the seven candidates of
  section 10 apply unchanged with $L_p = \tfrac{V_p}{2} M_p$ and linear
  coefficients $e^{s}\sqrt{V_p}\,z + L_p$ (the max's argmax is invariant
  under the common positive rescale, which is why the weight is legal).
- Two_arm is recovered at $k = 0$ via $\log\hat\tau = s + r$.

### 14.4 Postscript: which units to grade in (failure observed 2026-08-05)

The first residual weight implemented was the equation's own units,
$S^{3/2}/c$. A training run under it produced a well-scored net whose
Hamiltonian was dead: all learning numbers $\approx 0$, policy = never
explore. Mechanism: near the solution the residual carries the equation's
full $S^{-3/2}$ magnitude, but the never-explore failure mode produces a
residual of size $u \sim S^{-1/2}$ -- one power of $S$ smaller -- so graded
in equation units it fades like $S$ and is nearly invisible below
$\tau \sim 1$. (Two_arm escapes this because BC1 pins a premium derivative
at every scale; the three-arm walls are lower-dimensional and cannot.)

**RETRACTED 2026-08-10. The grading below is WRONG and is gone from the
code.** A weight on the residual is a reweighting of the domain, not a choice
of units: measured on two_arm_drift, the analogous chart weight was
$\tau^3$ across 15 decades and spent 70% of the gradient on a corner that was
already exact to ~1e-6 relative. The residual is graded in the equation's own
units, always; the never-explore degeneracy is the tie losses' job, and if a
dead-Hamiltonian net scores well again the answer is a stronger breaker, not a
thumb on the residual. See learnings section 3 for the standing rule. The
derivation is kept below because the failure mode it names is real.

~~The correct grading is one power of $S$ lower -- PREMIUM units,~~

$$
w \;=\; \frac{S^{3/2}}{e_1(I)} \;=\; \frac{(\det T)^{3/4}}{\tau_{bb}+\tau_{cc}+\tau_{bc}},
$$

which keeps the never-explore residual $O(g)$-loud at every scale while
retaining decade-fair footing for shape error (up to a benign tilt toward
low-information states, the side the two_arm benchmark favors). This is
exactly the analytic form of the old reactive scale $1 + |H - \text{commit}|$:
that hack measured the premium magnitude per point empirically; the
similarity analysis supplies the same magnitude as a law.

## 15. REJECTED: the complementarity term (tried and dropped 2026-08-07)

**The defect it targeted is real.** Where the simplex max sits at the
do-nothing vertex, $f(0,0)=0$ identically, so the HJB reads $u=0$ there. The
trained nets disagree with themselves on a band of states -- argmax at vertex
a while the premium is still alive -- covering 7.8% (64:64:64) and 8.8%
(k16) of the cloud. On that band the relative residual EQUALS the premium
exactly (max discrepancy 0.00e+00 over 5114 points), so it is a 100% error
by construction. It survives training because it is starved: per-point
parameter gradient 2.9e-4 against 6.9e-3 for the bulk, ~20x down, since the
similarity weight carries $(\det T)^{3/4}\sim\tau^{3/2}$ and the band sits at
median $\tau_{bb}$ 0.09 against 0.37 for the cloud.

**The fix tried.** A second graded term, `(committed * u)**2` averaged over
the batch WITHOUT the similarity weight, added at weight $W$, with
`committed` the exact test `best.x == 0 and best.y == 0` (the vertex
candidate is a literal zero in the candidate stack, so the equality is exact,
not a tolerance).

**What happened.** It works on its target and costs more than it earns:

| | baseline | $W=1$ (3700 it) | $W=0.1$ |
|---|---|---|---|
| band share | 7.8% | 5.7% | 6.6% |
| pde mean-of-squares | 2.58e-6 | 3.92e-6 (+52%) | 2.68e-6 (+4%) |
| pde p-mean | 7.67e-6 | 1.92e-5 (+150%) | 8.97e-6 |
| control tie | 2.52e-7 | up 6x | 3.99e-7 (+58%) |
| treatment tie | 1.20e-7 | up 11x | 1.69e-7 (+40%) |

**Why rejected, three reasons, in order of weight.**

1. It erodes the degeneracy breaker in EVERY configuration tried. Both walls
   climb monotonically, 6-11x at $W=1$ and 40-58% at $W=0.1$ (4x for k16 at
   the same setting). Since the term's own global minimum is $u\equiv 0$ --
   the never-explore solution of section 12 -- it pulls in the same direction
   as the documented attractor, and only the control tie opposes it.
2. It cannot improve the pde number, by construction. The band carries just
   9.8% of the pde loss, so a PERFECT fix caps the gain at ~10%; the same
   $(\det T)^{3/4}$ suppression that let the defect survive also caps the
   reward for repairing it. Judged on pde it can only break even.
3. The premise was never measured. The band was hypothesised to drive the
   arena's 10.4% three-arm wrong-commit rate; two training rounds were spent
   without running the arena to check.

**What is NOT established.** At $W=0.1$ the champion's blob signed bias fell
+4.41e-3 -> +3.39e-3 (best of that day) with pde flat. That run also had
hours of extra training and there was no weight-0 control arm of equal
length, so the improvement is unattributed. Do not cite it as evidence for
the term.

**Reopen only if** the arena establishes that three-arm regret is lost to
premature commitment AND a weight-0 control run of equal length isolates the
term's effect from ordinary training. Absent both, this is settled.

## 16. The pairwise basis: how much of u is three two-arm premia? (measured 2026-08-12/13)

The N-arm scaling route is to build the premium out of the TWO-arm premium of
each pair -- one basis net whatever N is, with only the call count growing
(3 pairs at N = 3, 6 at N = 4), against a monolith whose feature stack grows
quadratically. This section is what the ansatz is worth, measured rather than
argued.

Each pair is a two_arm problem in that pair's own marginal (Schur) precision:
on the fundamental wedge m_c <= m_b <= 0 arm a leads every pair, so with
`det = tau_bb tau_cc - tau_bc^2`,

    p_ab = u2(-m_b,      tau_bb - tau_bc^2 / tau_cc)
    p_ac = u2(-m_c,      tau_cc - tau_bc^2 / tau_bb)
    p_bc = u2(m_b - m_c, det / (tau_bb + tau_cc + 2 tau_bc))

Regressing the three_arm champion's premium on those three, 20k wedge points,
basis `data/two_arm.pt` (a 402-parameter net):

    quantity        R2 (all)   R2 (alive)   coefficients (ab, ac, bc)
    premium value     0.9790     0.9766     +0.923, +0.476, +0.102
    d/d tau_bb        0.7651     0.7606     +0.724, +0.703, +0.697
    d/d tau_cc        0.8664     0.8719     +0.944, +0.359, +0.600

Four things follow, and the last two are the load-bearing ones.

- The basis contributes SHAPE, not precision. A 406-parameter basis and a
  9,154-parameter one give identical R2, identical coefficients and identical
  Jaccard (0.93) between the leading pair's dead set and three_arm's commit
  region. The basis can be as small as two_arm's own winner.
- The 98% is not an artifact of the dead region. Restricting to points where
  the premium is strictly positive (48.6% of draws) barely moves it.
- VALUE is not the graded quantity. The HJB residual is built from
  derivatives -- the learning operator is (mean diffusion) + v_tbb -- and the
  fit there is 0.76-0.87, not 0.98. About a quarter of what gets scored is not
  linear in the pair premia, so a correction net needs real capacity; it is
  not a perturbation.
- The coefficients DIFFER by quantity. On the value the leading pair dominates
  (0.92 / 0.48 / 0.10); on d/d tau_bb all three enter about equally
  (0.72 / 0.70 / 0.70). A combiner fitted on values is wrong by ~7x on the
  third pair for the slopes, so a learned head must NOT be seeded from the
  value fit.

The combiner is therefore a learned linear term plus a correction net, summed
in RESPONSE space, not premium space. The coefficients sum to 1.501: added as
premia they overcount by half and the result is not a value function, because
each pair premium assumes the whole budget goes to that pair. Behind the
envelope-times-saturated-response gate, `0 <= u < nu2` stays architectural
whatever the coefficients do.

Cost, measured on a crowded 4090 at batch 4096: a v2 step is 1.30x a v1 step,
and step cost is FLAT in width for both (619 ms at 2,754 parameters against
670 ms at 9,739; 894 ms at 914 against 872 ms at 9,794). The step is
dispatch-bound on the analytic machinery -- five derivatives, the 2x2 Hessian,
24-node Genz quadrature, the seven-candidate max -- so the basis toll is fixed
rather than scaling, and a SMALLER v2 is relatively more expensive because
there is less net to amortise it over. v2's payoff at N = 3 cannot be training
time; it has to be accuracy-per-parameter and the N-arm scaling.

### Verdict: tried, and rejected (2026-08-13)

Built as `three_arm_v2` and deleted the same week; the code and the full
reasoning are at the git tag `three-arm-v2-final`. It never led at matched
parameters AND matched iterations -- v1 32:32:32 reached 9.6e-2 by iteration
20k, the v2 of the same size needed 50k to reach 1.07e-1.

The bound is where it broke, and the two ways to place the combination are
both blocked:

- In RESPONSE space (inside the `relu(r)**2 / (1 + relu(r)**2)` gate),
  `0 <= u < nu2` stays architectural, but no linear-in-r term survives a
  squaring and a rescaling. Trained 315k iterations there, the head found
  0.081 / 0.004 / 0.013 against the fit's 0.923 / 0.476 / 0.102, contributed
  6% of the response, and the correction net weighted the pair columns at
  0.38x the other features. Permuting the premia damaged the residual LESS
  than permuting any other feature group but the raw anchors.
- In PREMIUM space, where the fit lives, nothing bounds the sum. The head
  trained to the OPPOSITE sign -- -0.812 / -0.362 / +0.002 -- using the
  premia as a subtraction the correction then overshoots to cancel, because
  two large opposing terms fit a residual more easily than one term that has
  to be right. The result leaves the space of value functions: u < 0 on 4.37%
  of states (worst -2.02) and u > nu2 on 5.49% (worst ratio 1.4e28). Its
  residual is not comparable to a net whose premium is provable.

A next attempt needs the combination in premium space AND a bound. p_ab and
p_ac are each bounded by nu2 (max ratio 0.88 over 50k wedge states), so a
simplex over those two plus an enveloped correction is legal. p_bc is bounded
by nu2 on no state: it exceeds it on 19.8%, without limit, because the
losers' contest stays alive where both challengers are hopeless against the
control and nu2 -> 0. p_bc can be a FEATURE, never a term of the value.

## 17. The drop-one subsolution (derived 2026-08-14)

Ignoring one arm — never sampling it, never committing to it — is an
admissible policy, and what remains is EXACTLY the two-arm problem on the
surviving pair. Restricting the policy class can only lower the value, so
each drop is a global lower bound and so is their max. On the wedge
(C = 0, v = u), in section 16's pair-premium notation:

    u(m, T) >= B(m, T) := max( 0,  p_ab,  p_ac,  m_b + p_bc )

    p_ab = u2(-m_b,      tau_bb - tau_bc^2 / tau_cc)
    p_ac = u2(-m_c,      tau_cc - tau_bc^2 / tau_bb)
    p_bc = u2(m_b - m_c, det / (tau_bb + tau_cc + 2 tau_bc))

Why each term is the exact drop value:

- Drop c: on the alpha_c = 0 edge the (m_b, p_b)-dynamics are two_arm's bit
  for bit — dtau_cc = dtau_bc = 0, so d p_b = dtau_bb =
  (alpha_a alpha_b / sigma^2) dt, and the mean diffusion collapses to
  gb / p_b^2 (the section 4 two-arm collapse). The restricted problem IS
  two_arm at (-m_b, p_b); a leads the pair on the wedge, so the pair's
  commit value is the wedge's own 0 and the drop value is the bare premium
  p_ab. Drop b is the b<->c swap.
- Drop a: inside {b, c} only the contrast theta_b - theta_c matters (the
  common level is earned in expectation whichever arm is chosen); the best
  commit inside the pair is m_b, and the contrast problem is two_arm at
  (m_b - m_c, pair marginal). The m_b ballast is what section 16's verdict
  demanded: p_bc alone exceeds nu2 on 19.8% of states (the losers' contest
  outlives both challengers), but exactly there m_b -> -inf and the max
  discards the term. Ballasted, it is the value of a feasible policy,
  hence legal.

B is S3-equivariant for free (relabels permute the candidates), zero
wherever all three pair premia are dead (each u2 carries an architectural
exact-zero region, so the exact-zero commit region survives the base), and
EXACT in all three one-arm-dead far fields of section 8 — the limits
promoted to a global bound. It is also a viscosity subsolution of the full
HJB, not only a pointwise bound: each drop value satisfies the three-arm
HJB with equality using only its own edge's candidates, the full-simplex
max only raises the right side, and a max of subsolutions is a subsolution.

The same construction lifts to the drift pair: dropping an arm in
three_arm_drift leaves two_arm_drift on the pair, etahat carried through,
so two_arm_drift.pt is the base for a 3ad sibling.

### The architecture it suggests, and the issues in the way

u = B + c with c >= 0 architectural (envelope times saturated response, as
everywhere): the correction's target u - B is provably nonnegative, the
dead solution u = 0 leaves the representable class (B > 0 wherever any
pair premium is alive, so the never-explore attractor and the all-dead
absorbing state both die architecturally, and a zero-init correction head
starts at u = B — far fields exact at step 0), and the correction owns only
the genuinely three-arm part: the junction region, where the dominant error
blob already lives. Known issues, decreasing weight:

1. The bound is a theorem for the TRUE u2; the implemented base is the
   two_arm champion net. Wherever that net overclaims, B_net > u and a
   positive correction cannot come back down — base error is a hard bias,
   uncorrectable downward, unlike any loss term. Escape hatches: the
   constant shift (learnings 9) — subtract two_arm's
   sup(residual+)/rho for a proven base — or run the two_arm
   subsolution program first. Until one lands, "positive correction" is a
   strong prior, not a proof.
2. The max has gradient jumps on its argmax seams; the true u has none
   (the seams are representation artifacts, not free boundaries). The
   smoothing fix — the Boltzmann softmax
   sum(x_i exp(b x_i / scale)) / sum(exp(b x_i / scale)), a convex
   combination hence <= max at any temperature (logsumexp is the WRONG
   side) — was the plan and is REFUTED by the step-0 sweep (bare_b.py, since
   deleted;
   2026-08-14): fenced median 1.4e3 at b = 3 against 9.0e-6 for the hard
   max, worsening with b, no basin anywhere. Mechanism, isolated by
   detaching the softmax weights (residual falls 1.4e4 -> 9.9e-3): the
   blowup IS the weight-derivative terms. The hard max is locally one
   candidate everywhere off the measure-zero seam, so its residual is the
   winning two_arm solution's own; the mixture band's extra derivatives
   (dw x, d2w x) are new physics the HJB rejects over a finite-measure
   region, priced at natural-units value scale — the worst points sit at
   sqrt(det) 1e-3..7e-3 where values run ~30. Even value-only mixing
   (weights detached, not a legal architecture) costs 250x over hard.
   Separate trap from the same sweep: normalizing the exponent by the
   candidate SUM makes B degree-1 homogeneous in the candidates — a cone
   at the dead corner whose second derivatives diverge like 1/|candidates|
   along the whole commit boundary; normalize by nu2 if a softmax is ever
   wanted for something else.
   VERDICT: the hard max stands. It is the theorem-clean object (a max of
   subsolutions is a viscosity subsolution; the upward kink is vacuous
   under the from-above test-function condition), the losses cannot see
   the seams (collocation and both wall-sampling laws hit them on null
   sets; measured near-seam decile loss share 0.1%), and the one real
   cost — the correction cannot cancel the base's gradient jump while
   staying positive, so u keeps hairline kinks the true solution lacks —
   is accepted for v1, revisited only if the junction blob demands it.
3. Input-domain coverage: the base is evaluated at the pair MARGINALS
   (det/tau_cc etc.), which the three_arm sampler does not directly floor.
   If they land decades outside two_arm's trained support, the base emits
   garbage the bound does not cover — the same failure class as 3ad's
   det-only floor (kb/three_arm_drift.md section 6). Measure the marginal
   ranges under the three_arm law before training.
4. The Hamiltonian differentiates through three frozen two_arm net calls
   per point (2x2 Hessian + three tau derivatives). Section 16 measured
   the pairwise-basis toll at 1.30x a v1 step, dispatch-bound; expect the
   same class of cost. The base must be FROZEN — a co-trained base is v2's
   failure mode again, and the bound story dies with it. Freezing binds
   the ceiling to the base checkpoint: swapping in a better two_arm later
   moves B under a trained correction, a retraining event, not a drop-in.
5. Envelope for the correction: nu2 - B, giving the interpolation form
   u = (1 - r) B + r nu2 with r in [0, 1) the gated response — BOTH proven
   bounds architectural at once, conditional on B <= nu2 pointwise. That
   inequality is a theorem for the true u2 by the section 13 chain
   (nu(m_b, sd_b) <= nu2, and m_b + nu(-(m_b - m_c), sd_bc) =
   E[max(theta_b, theta_c)] <= nu2), but the two_arm champion's log_scale
   is trained to +0.100 (exp 1.105: the response cannot reach 1, so the
   scale compensates above the proven envelope), so for the NET base it is
   empirical: measured 2026-08-14 over 65k wedge states, B < nu2 wherever
   nu2 > 0, max ratio 0.978, q99 0.921. The pinch sits in far fields where
   the true correction is ~0; on alive states the median B/nu2 is 0.378,
   so the junction keeps ~60% of nu2 in room against a true triple-point
   premium of 0.53-0.85 nu2 (section 13). Two costs: the envelope inherits
   B's seam kinks (hairline, same class as the base's own — see issue 2's
   verdict), and the alive-start init must NOT be the standard head
   bias 1 — that starts r at ~0.5, halfway to nu2, discarding the step-0
   head start; a bias near 0.1 starts alive at ~1% contamination of B.
6. The base changes the pde term's magnitude and deletes the dead-solution
   floor argument (the tie losses' breaker role shrinks) — re-derive every
   auxiliary weight per the house rule.

### Step-0 measurement (2026-08-14; the bare_b.py that made it is deleted)

Bare B — hard max, two_arm.pt (16x16) as u2, correction absent — graded by
the stock three_arm pde_loss: natural units, POWER = 1, batch 4096, medians
over 7 draws, champion = three_arm.pt on the SAME draws:

    bare B 1.6e-4 (per-draw spread 2.1e-5 .. 1.4e-3)     champion 8.3e-5
    fenced to states with all three pair marginals >= two_arm's PRIOR_FLOOR
    (99.67% of the cloud):        bare B 9.1e-6          champion 7.4e-5

With zero training the base beats the trained champion by 8x on 99.7% of
the cloud. The unfenced tail is issue 3 realized, and only issue 3: every
worst point has a pair marginal at 5-8e-4 — below two_arm's sampled
support, where the response net extrapolates under an exploding nu envelope
— with single points carrying up to 98% of a draw's loss (the 3ad
mis-sampling failure class, arriving through the base's INPUT this time).
The 0.33% of wedge states below the fence exist because the sampler floors
the pair COORDINATES, not the Schur marginals, which correlation pushes
under the floor. Issue 2 is priced and small at step 0: the alive-base
decile nearest a seam carries 0.1% of the loss (the smoothing argument is
about the correction's training gradients, not this eval). The rest of the
loss sits where the theory says: exactly 0 on the argmax-dead 52%, and 96%
of it in the ab-argmax region that contains the junction.

Rulings (2026-08-14): roll with the unfenced law and the two_arm net as
they are, sub-floor spikes on the watchlist. The base is
B = max(p_ab, p_ac), hard max, two candidates: on the wedge arm a leads,
so the drop-a candidate is never a far-field limit and hard-argmaxes on
only 0.1% of states (its ballast would also drag any average negative),
and the 0 candidate is redundant among nonnegatives; m_b + p_bc is
demoted to a correction-net feature, which is where section 16 put p_bc.
Priced by the sweep: pairs-only 8.922e-6 fenced against 8.924e-6 for all
four candidates. The Boltzmann temperature sweep that same day refuted
smoothing entirely (issue 2).

### Verdict: SCRUBBED 2026-08-16, and what it taught

`pinn/problems/three_arm_v3` was built, measured and deleted. The code is
gone; this is the record. Read it before proposing a base-plus-correction
architecture again.

**THE HEADLINE IS THE BASE, NOT THE CORRECTION.** The drop-one heuristic
ALONE -- `u = max(p_ab, p_ac)`, two frozen two_arm evaluations and a max,
with NO three-arm training of any kind -- beats Thompson sampling by 16% and
is at parity with the trained three_arm champion. Pooled over 9,000 paired
reps in two disjoint seed blocks at production parameters (frontier
scenario):

    drop-one - Thompson      -3410   [-3796, -3023]
    drop-one - champion       +571   [  -28, +1171]   covers zero
    champion - Thompson      -3981   [-4454, -3508]

That is the result worth keeping from the whole exercise. Rebuilding it is
three lines: evaluate the two_arm champion at each pair's marginal state
(section 16's formulas), take the max, wrap it as a premium. It is the
honest BASELINE for any future three-arm work -- Thompson is the wrong bar.

**The correction added nothing, in any variant.** `trained v3 - drop-one`
was +412 [-44, +867] after 35k steps. Three gates were tried:

- ONE-SIDED (`u = B + (nu2 - B) G+`). Deadlocked at iteration 10,200: the
  base overclaims on 27% of wedge states (it is a trained net, so `B <= u`
  is not a theorem), the correction needed to go negative there, could not,
  and `relu(r)**2` zeroed the correction AND its gradient.
- TWO-SIDED (`relu(r)**2` on each sign, B outside the gate). Inert for a
  measured reason: both branches have zero value AND ZERO SLOPE at r = 0, so
  `u = B` is a STATIONARY POINT (du/dr = 2.0e-3 at r = -1e-3, 0 at r = 0). A
  35k run parked against it -- 81.3% of live states at u ~= B, 18.5% below,
  0.2% above -- while the truth sits ~15% ABOVE B (the champion's premium
  exceeds it on 99.7% of live states, median 1.155x). The C1 smoothness at
  u = B was the design's proudest property and is what made it inert.
- ANCHORED (B inverted into the gate argument: `anchor = sqrt(t/(1-t))`,
  `t = B/nu2`, `u = exp(ls) nu2 G(anchor + r)`). Fixed the mechanism -- the
  net finally climbed, u/B 1.005 -> 1.122 in ~20 minutes, u > B on 94.9%,
  non-concavity 23.6% -> 4.0% -- and the POLICY GOT WORSE: 29,230 against
  Thompson's 22,543 over 3,000 paired reps, +6,687 +/- 2,250. Every
  subsolution diagnostic improved while the only thing that matters
  degraded.

**Convexity is the property, `L[f] >= 0` is a proxy, and the net gamed it.**
`B = max(p_ab, p_ac)` is a max of convex functions, so a fresh v3 is 100%
convex in the belief mean (worst eigenvalue -5e-10). The 35k run took that
to 27% of live states (worst curvature -0.99) while IMPROVING the sampled
directional test 80.3% -> 93.3%. `L` mixes the mean-Hessian with the
precision-drift term, so a net can buy `L >= 0` by inflating `dv/dtau` while
the mean-Hessian goes negative. Convexity is checkable directly and cheaply
(2x2 Hessian: `v_bb >= 0`, `det >= 0`), needs no random direction, and has
none of the 4.7x draw noise that made the concavity weight uncalibratable.
If a curvature condition is graded again, grade THAT one.

**Traps banked, all measured here:**

- A dead relu-squared correction prints a HEALTHY loss. Under the two-sided
  residual: pde 8.6e-6, control tie 2e-2, and treatment tie AND concavity
  EXACTLY 0.000e+00. Under the subsolution objective: `climb` and
  `violation` both exactly 0 with the ridge still satisfied at ~1e-11 (a
  sliver alive at m_b = 0 holds BC1 while everything the cloud samples is
  dead).
- Checkpoints do not survive a gate change. Loading two-sided weights into
  the anchored gate reinterprets the response entirely and produced
  `u/B = 0.123` and an arena regret of 123,639 -- both artifacts, not
  measurements.
- `sqrt` in the anchor has an infinite derivative at 0 and `B = 0` on 55% of
  the wedge, so the hamiltonian's chain nans from step 0 without a
  `clamp_min` whose backward is zero where it binds.

**What outlived it.** The SAMPLER FENCE is a real bug fix and three_arm
still lacks it: three_arm floors the two diagonal precision entries, which
leaves the b-c pair coordinate bare, and the fold then permutes that bare
coordinate into any slot -- so every Schur marginal reaches ~0.5x the floor.
Those states were 1.0% of the cloud and carried 99.996% of the pde loss
(mean residual squared 1.2e3 against 4.7e-4 fenced). Flooring all three PAIR
coordinates puts every marginal in support by construction, with no
rejection step. Worth porting.

## 18. The subsolution objective (THE three-arm objective since 2026-08-16)

two_arm's objective of kb/two_arm.md section 10, carried to N = 3. Built as
the `three_arm_v4` clone (v2 and v3 are burned names, so the counter kept
going), measured, and PROMOTED; the clone is deleted and the two-sided
residual it replaced survives only in this record.

    loss = violation + TIE * (control + treatment) + CONCAVITY * concavity
           - CLIMB * climb
    violation = mean(relu(v - max H))     LINEAR
    climb     = mean(u)                   no units factor
    TIE 10, CONCAVITY 2.6, CLIMB 4.0e-2

Three design points differ from the two-arm pair:

- THE CLIMB NEEDS NO UNITS FACTOR. Both two-arm problems grade a similarity
  chart and divide `tauhat**1.5` back out of the residual, which forces the
  climb into the same units or the floor decade is sacrificed. This problem
  is graded in value form, so residual and premium share units by
  construction.
- CONCAVITY STAYS, and it is NOT subsumed the way two_arm's pos_learning was.
  That proof needs a violated learning number to force the max onto a VERTEX;
  at N = 3 the max can sit on an EDGE and stay feasible while a direction is
  non-concave. Measured on the champion of this date: 84% of live non-concave
  states were feasible.
- CLIMB from the dead-solution floor, not a sweep. The champion's
  violation / climb is 4.1e-3, below which the commit envelope wins the LP
  objective outright; 4.0e-2 is 10x clear. Deliberately not calibrated on a
  short sweep -- see kb/two_arm_drift.md section 10 for what that costs.

RESULT, trained off the two-sided champion (kept as
`data/three_arm.64x64x64k8.two-sided.pt`), 8192-state medians over 5 draws:

    net              violation   overclaim   sup(res+)   climb    non-concave
    two-sided         1.92e-3      26.8%      3.52e-1    0.4667      0.02%
    subsolution       2.59e-5       1.44%     2.28e-2    0.4681      0.02%

74x on the mean violation and 15x on the SUP, which neither two-arm problem
managed -- there the sup barely moved. THE PREMIUM IS UP 0.3%, and that is the
number to read against section 17: v3's anchored gate climbed the premium 12%
and destroyed the policy. This objective leaves the function where it is and
fixes WHERE IT OVERCLAIMS.

ARENA, 20,000 paired reps in two disjoint seed blocks (0-9,999 and
500,000-509,999) at production parameters, frontier scenario:

                        block 1          block 2           pooled
    v4 - drop-one     -671 +/- 518   -1031 +/- 484     -863 +/- 354
    v4 - champion     -545 +/- 520    -274 +/- 477     -398 +/- 352
    v4 - Thompson    -4177 +/- 422   -4494 +/- 375    -4354 +/- 280

THE FIRST TRAINED THREE-ARM NET TO BEAT THE DROP-ONE HEURISTIC, at 4.8 sigma
and replicated in both blocks -- section 17 named that the real baseline and
recorded the old champion FAILING it (+571, CI covering zero). Against the
champion it is 2.2 sigma, favourable in both blocks: suggestive, not
established, and promoted on the bound numbers with the arena as corroboration.

LEARNING RATES: only the first descends. 1e-4 over 122k gave 45x and fell
throughout; 3e-5 (17%), 1e-5 (19%) and 3e-6 (10%) each stepped down at the cut
and then sat flat for their whole stretch. A cut that buys a step and no
descent is shrinking the Adam wander radius around a fixed point, which is the
topology's floor and not the schedule's.

### The uniform-shift bound LOSES to the free one (2026-08-16)

Section 17's `B = max(0, p_ab, p_ac, m_b + p_bc)` is a PROVABLE subsolution --
each drop is an admissible policy -- and costs no training. The trained net is
one only after subtracting its worst overclaim (learnings section 9). Measured
on 32,768 wedge states:

    mean premium      trained 0.4706   drop-one B 0.4307
    mean PROVEN       trained 0.4235   drop-one B 0.4307
    trained >= B before the shift: 99.15%;  after: 60.2%

So the trained bound is better everywhere until it is made rigorous, and then it
is worse: on states where B is alive the shifted net is a median 0.60x of B
and worse on 88.9%. The whole difference is the tail. Priced at q99.9
(shift 6.4e-3) instead of the sup, the trained bound wins by 7.8%, and the sup
estimate is not even stable -- it moved 1.6e-2 -> 1.4e-1 as the draw count
grew. The positive residual is exactly 0 at the 90th percentile, 1.9e-4 at
q99, 6.4e-3 at q99.9.

NOT a sampler artifact, which was the first suspicion given the un-ported
fence below. The worst states have their smallest Schur marginal at 5.9e-3 --
above `PRIOR_FLOOR`, merely 20x below the cloud median -- and the sub-floor
states the fence would remove are 0.98% of the cloud carrying 0.02% of the
positive residual. The fence would not move the bound. That also
re-prices the fence itself: it was measured against a SQUARED residual, where
those states carried 99.996% of the loss; the linear violation term de-weights
them by four orders on its own.

## To come

- Nothing mathematical: the problem is fully derived.
- The sampler fence of section 17 is still un-ported, and now known to be
  worth much less than its original measurement suggested (see above).
