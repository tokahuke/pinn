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

    H(b, c) = b (m_b + Lab) + c (m_c + Lac)
            - Lab b^2 - Lac c^2 + (Lbc - Lab - Lac) bc

The max of a quadratic over a triangle sits at an interior stationary point,
an edge's stationary point, or a vertex — nowhere else. Seven candidates, all
closed form:

    1.   interior: solve grad H = 0 (2x2 linear; safe-denominator its det
         4 Lab Lac - (Lbc - Lab - Lac)^2); feasible iff b, c >= 0, b + c <= 1
    2-4. edges — each one IS the two-arm clamped vertex:
         c = 0:      b* = clamp( (m_b + Lab) / (2 Lab), 0, 1 )
         b = 0:      c* = clamp( (m_c + Lac) / (2 Lac), 0, 1 )
         b + c = 1:  s* = clamp( 1/2 + (m_b - m_c) / (2 Lbc), 0, 1 ),
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
thinking; this chart is an implementation detail for *grading residuals*: in
it every derivative term of the PDE has bounded coefficients, so numerical
differentiation error is never amplified by $1/\det T$. Derivation by agent,
every identity verified in float64 against the raw-coordinate implementation
(128 wedge states, max relative error $2.7\times10^{-13}$).

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

Structure worth registering:

- $e^{s}$ multiplies only derivative-free terms; every derivative coefficient
  is bounded on the whole wedge ($P_b, P_c \in [0,1]$, $Q \in [-1,2]$,
  $R_r \in [-1,1]$, $R_k \in [-0.385, 1]$). The full HJB adds
  $\max_{\alpha \in \Delta}$; the seven candidates of section 10 apply
  unchanged with $L_p = \tfrac{V_p}{2} M_p$ and linear coefficients
  $e^{s}\sqrt{V_p}\,z + L_p$.
- The $V_p$ prefactors are NOT bounded ($V_{ab}V_{ac} = 1/(1-k^2)$); the
  recommended, argmax-invariant normalization divides the equation by
  $c = (V_{ab}+V_{ac}+V_{bc})/2$, giving diffusion weights in $[0,1]$
  summing to 1 and the single scalar clock
  $e^{s}/c = \det T/(I_{ab}+I_{ac}+I_{bc})$.
- Both wall conditions and both mirror maps are $s$-free; the mirrors act as
  transpositions of $(V_{ab}, V_{ac}, V_{bc})$ (treatment mirror:
  $r \to -r$).
- Startup ($s \to -\infty$): the source dies and the exact stationary
  solution is $g_0 = S^{1/2}\,E[\max(0,\theta_b,\theta_c)]$ — it satisfies
  the interior AND both walls. The nu-sum envelope solves the startup
  interior (it is a belief martingale) but fails the control wall by exactly
  $\Phi(z_c')$ — see the section 13 correction.
- Two_arm is recovered at $k = 0$ via $\log\hat\tau = s + r$.

## To come

- Nothing mathematical: the problem is fully derived. Remaining work is code
  (the similarity-graded residual per section 14, mirroring two_arm's) and
  then training.
