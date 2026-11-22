# Period-copy evidence (P3, 16:25 PST)

The copy is parameter-free. With T = 2 pi / k, the forecast copies the history forward by whole periods, using
linear time interpolation. It is gated by how well the history repeats with period T (a normalised residual).
Because it has no trained parameters, it can be scored on any training case.

- With a 10-frame history (the real input), only k >= 6.98 qualifies. On the 21 training windows with
  k >= 6.98, the copy is used on 16 (NMSE 0.025), and the hybrid scores 0.119 vs 0.158 for the CNN (which
  trained on these cases).
- With the history lengthened to one period: 2100 windows from 523 cases with k 3.5-7. The gate (residual < 0.3)
  passes 1944 windows with mean copy NMSE 0.059 and 6 above 0.5 (max 0.68). Rejected windows would have
  averaged 0.59.
- Accuracy does not depend on k. Periodic windows pass 98-100% of the time, with copy NMSE 0.044-0.058 across
  every k bin from 3.5 to 7.
- Half-period copy with reflection (applies for k >= 3.5): on the frequency proxy (model trained on k <= 5.5,
  tested on k 6-7), gating it in at residual < 0.3 takes NMSE from 0.984 to 0.518. On the theta0 proxy it gives
  no gain (0.314 -> 0.316). On validation (transition band) it never passes the gate (residual about 2).
