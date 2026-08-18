# Release naming

GitHub releases bundle the champion models, one asset per problem. The
tag is a CODENAME, not a date: a SURNAME, nothing else -- `anscombe`, title
`Anscombe: best models, <date>`. The date lives in the title and in
the release's own metadata, which is where a date belongs; the tag is what
people say out loud.

Rules:

- One PERSON per release. Never a pair -- `black` and `scholes` are two
  releases, `hamilton` and `jacobi` are two more, and HJB is not whole until a
  `bellman` has shipped as well. The alphabet does the separating and putting
  the collaborators back together across releases is the point.
- Names run in ALPHABETICAL order. A letter with several candidates takes
  CONSECUTIVE releases (Bachelier, Bellman, Black, Blackwell, then Chernoff);
  the alphabet advances when a letter runs dry, not once per release.
- Q, U, X, Y, Z have no candidate worth the name. The alphabet does not have
  to close.
- Past Wiener the surnames come round again and the LAP is an INTEGER on the
  end: `anscombe2`, title `Anscombe 2`. First lap is bare. The series
  therefore cannot run out and no second theme is ever needed.
- A codename change is a good place to say IN THE NOTES that the grading
  changed and the metrics do not compare backwards. It is not a substitute for
  saying it: the reader cannot infer a functional change from a letter.

## What the notes say

The reader is downloading a model to run it, not reviewing the training. Write
for that reader and nobody else.

Include, in this order: the `pinn.release.load` snippet, a table of asset /
problem / parameter count, the arena result against the baseline that matters (Thompson
for the two-arm problems, drop-one for three_arm), and one line on any problem
whose model is being withheld and why.

Leave out: residual numbers, loss values, violation fractions, sup estimates:
they decide PROMOTION and they belong in the problem doc, and nobody choosing
a download cares. The training objective, unless a number in the notes cannot
be read without it. Any word an outsider would have to look up, `checkpoint`
included; the files hold models.

Never call a trained net a SUBSOLUTION. Every net shipped so far overclaims on
0.5-2% of states, and it takes subtracting the worst point to make the bound
rigorous, which loses to the free drop-one bound. Say what the objective drove,
not what the net certifies.

Prose runs through the stop-slop skill: no em dashes, active voice, no
throat-clearing openers, no punchy one-liner sign-offs. When a note is wrong,
rewrite it from a blank page. Patching a bad note sentence by sentence keeps
its shape and its shape is what was wrong.

## Roster

Everyone here has a claim on this specific problem -- the HJB, its free
boundary, or the bandit. The one-liner is the claim; it is what stops the
series drifting into a general mathematician's hall of fame.

| Name | Claim |
|---|---|
| Anscombe | *Sequential medical trials* (1963): stop early to stop giving people the losing arm. This problem with the ethics attached. |
| Arrow | Arrow-Blackwell-Girshick (1949), Bayes solutions of sequential decision problems. |
| Bachelier | 1900 thesis, Brownian motion for prices. The first diffusion. |
| Bellman | The B in HJB; his 1956 Sankhya paper is the two-armed bandit. |
| Black | Half of the PDE that made free-boundary valuation computational. |
| Blackwell | Comparison of experiments. The mean-preserving-spread argument behind `L_ab >= 0` is his theorem in other clothes. |
| Chernoff | *Sequential design of experiments* (1959): the continuous-time bandit and its free boundary. The nearest ancestor of this repo. |
| Doob | Martingales and optional stopping -- why a supermartingale bound means anything. |
| Dynkin | Optimal stopping; the Dynkin formula. |
| El Karoui | Reflected BSDEs: free boundaries in optimal stopping, done properly. |
| Feynman | Feynman-Kac. PDE <-> expectation, the bridge the net stands on. |
| Gittins | The index theorem: the exact bandit solution for INDEPENDENT arms, and why the correlated case here is not it. |
| Hamilton | The H in HJB. |
| Ito | The calculus. Every derivative chain in the loss is his lemma. |
| Jacobi | The J in HJB. |
| Kolmogorov | Forward/backward equations: the belief's own diffusion. |
| Lai | Lai-Robbins (1985), the regret lower bound the arena measures against. |
| Lindley | Bayesian experimental design; expected information gain. The value of a look. |
| McKean | The American-option free boundary. Smooth pasting, which this net learns instead of imposing. |
| Merton | Continuous-time control; the HJB as a working tool. |
| Pontryagin | The maximum principle: the control half. |
| Robbins | *Some aspects of the sequential design of experiments* (1952). The bandit paper. |
| Scholes | The other half of Black-Scholes, deliberately eighteen releases later. |
| Snell | The Snell envelope: the value function of optimal stopping, by name. |
| Stefan | The moving-boundary problem (melting ice, 1890s). Every free boundary is a Stefan problem. |
| Thompson | 1933. The baseline the nets beat. |
| Ville | Martingale inequalities; anytime-valid stopping. |
| Wald | *Sequential Analysis* and the SPRT: the original stopping boundary. |
| Wiener | The process itself. |

Years and attributions above are from memory and have not been checked against
the papers. Check one before it goes in a release note.

## Shipped

| tag | date | what it was |
|---|---|---|
| `anscombe` | 2026-08-17 | two_arm, two_arm_drift, three_arm, all trained by the one-sided objective. No three_arm_drift: it still loses to Thompson. |
