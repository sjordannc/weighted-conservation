# NGC 6503 local-operator compact-term test

This run keeps the geometric backbone fixed and replaces the previous enclosed-mass-like
compact operator with local operators built directly from the central-excess stellar profile.

Models compared:
- geometric only
- earlier best cumulative compact model: baseline_gauss_Rd3 + cumulative operator
- local central-excess operator: excess4_raw + local_sigma
- local central-excess operator: excess4_raw + local_sqrtRsigma

Key results:
- Geometric only reduced chi^2: 3.077
- Best cumulative compact reduced chi^2: 0.738
- Best local operator reduced chi^2: 0.965
- Local_sigma reduced chi^2: 0.976

Outer compact leakage at the last radius:
- cumulative: 16.61 km/s
- local_sigma: 0.00 km/s
- local_sqrtRsigma: 0.00 km/s

Interpretation:
- The local operators are worse on raw chi^2 than the best cumulative compact model,
  but they strongly suppress the unwanted outer compact tail.
- The best local compromise is local_sqrtRsigma on total chi^2.
- The cleanest inner residual is local_sigma, which keeps the inner mean residual closest to zero.
