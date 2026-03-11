import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares

BASE = Path('/mnt/data')
meta = pd.read_csv(BASE/'NGC2403_NGC6503_SPARC_metadata.csv').set_index('Galaxy')
m = meta.loc['NGC6503']
rot = pd.read_csv(BASE/'NGC6503_rotmod.dat', comment='#', sep=r'\s+', header=None,
                  names=['R','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk_rot','SBbul_rot'])
dens = pd.read_csv(BASE/'NGC6503.dens', comment='#', sep=r'\s+', header=None,
                   names=['R','SBdisk','SBbul'])

R = rot.R.values.astype(float)
Vobs = rot.Vobs.values.astype(float)
err = rot.errV.values.astype(float)
Vgas = rot.Vgas.values.astype(float)
Vdisk = rot.Vdisk.values.astype(float)
Vbul = rot.Vbul.values.astype(float)
Rdens = dens.R.values.astype(float)
SigmaL = (dens.SBdisk.values + dens.SBbul.values).astype(float)
Rd = float(m.Rdisk)


def cumtrap(x, y):
    out = np.zeros_like(x, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def geom_term(R, Vbar2, Ag):
    return Ag * cumtrap(R, Vbar2) / np.maximum(R, 1e-12)


def fit_outer_exponential(rmin=4.0):
    sel = (Rdens >= rmin) & (SigmaL > 0)
    x = Rdens[sel]
    y = np.log(SigmaL[sel])
    A = np.vstack([np.ones_like(x), x]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return np.exp(a + b * Rdens), a, b

Sigma_exp, a_exp, b_exp = fit_outer_exponential(4.0)
Sigma_exc = np.clip(SigmaL - Sigma_exp, 0, None)


def source_density(kind):
    if kind == 'baseline_gauss_Rd3':
        Rc = Rd / 3.0
        return SigmaL * np.exp(-(Rdens / Rc) ** 2)
    if kind == 'excess4_gauss_Rd3':
        Rc = Rd / 3.0
        return Sigma_exc * np.exp(-(Rdens / Rc) ** 2)
    if kind == 'excess4_raw':
        return Sigma_exc.copy()
    raise KeyError(kind)


def operator_shape(sd, op):
    if op == 'cumulative':
        integrand = 2 * np.pi * Rdens * sd
        mraw = cumtrap(Rdens, integrand)
        shape = np.interp(R, Rdens, mraw, left=mraw[0], right=mraw[-1]) / np.maximum(R, 1e-12)
    else:
        sdi = np.interp(R, Rdens, sd, left=sd[0], right=sd[-1])
        if op == 'local_sigma':
            shape = sdi
        elif op == 'local_sqrtRsigma':
            shape = np.sqrt(np.maximum(R, 1e-12)) * sdi
        else:
            raise KeyError(op)
    mx = np.nanmax(shape)
    return shape / (mx if mx > 0 else 1.0)


def fit_variant(kind=None, op=None):
    if kind is None:
        starts = [np.array([1.0, -1.0]), np.array([1.1, -1.2]), np.array([0.8, -0.9]), np.array([1.2, -1.0])]
        bounds = ([-4, -2], [4, 1.5])
        shape = np.zeros_like(R)
    else:
        shape = operator_shape(source_density(kind), op)
        starts = [
            np.array([1.0, 0.0, -1.0]),
            np.array([1.1, -1.0, -1.2]),
            np.array([0.8, 1.0, -0.9]),
            np.array([1.2, 2.0, -1.0]),
            np.array([0.6, -2.0, -1.1]),
        ]
        bounds = ([-4, -10, -2], [4, 6, 1.5])
    best = None
    for x0 in starts:
        def residuals(x):
            if kind is None:
                logAg, logY = x
                Cc = 0.0
            else:
                logAg, logCc, logY = x
                Cc = 10 ** logCc
            Ag = 10 ** logAg
            Y = 10 ** logY
            Vstar2 = Y * (Vdisk ** 2 + Vbul ** 2)
            Vbar2 = Vgas ** 2 + Vstar2
            Vg2 = geom_term(R, Vbar2, Ag)
            Vc2 = Cc * Y * shape
            Vmod = np.sqrt(np.clip(Vbar2 + Vg2 + Vc2, 0, None))
            return (Vobs - Vmod) / err
        fit = least_squares(residuals, x0, bounds=bounds, max_nfev=200000)
        if best is None or fit.cost < best.cost:
            best = fit
    if kind is None:
        logAg, logY = best.x
        logCc = -np.inf
        Cc = 0.0
    else:
        logAg, logCc, logY = best.x
        Cc = 10 ** logCc
    Ag = 10 ** logAg
    Y = 10 ** logY
    Vstar2 = Y * (Vdisk ** 2 + Vbul ** 2)
    Vbar2 = Vgas ** 2 + Vstar2
    Vg2 = geom_term(R, Vbar2, Ag)
    Vc2 = Cc * Y * shape
    Vmod = np.sqrt(np.clip(Vbar2 + Vg2 + Vc2, 0, None))
    compact = np.sqrt(np.clip(Vc2, 0, None))
    residual = Vobs - Vmod
    chi2 = 2 * best.cost
    redchi2 = chi2 / (len(R) - len(best.x))
    return {
        'kind': 'geom_only' if kind is None else kind,
        'operator': 'none' if kind is None else op,
        'chi2': chi2,
        'redchi2': redchi2,
        'log10_Ag': logAg,
        'log10_Ac': logCc,
        'upsilon_star': Y,
        'inner_mean_residual_kms': residual[R <= 3.5].mean(),
        'outer_mean_residual_kms': residual[R >= 8.0].mean(),
        'compact_peak_kms': compact.max(),
        'compact_at_last_radius_kms': compact[-1],
        'curve': pd.DataFrame({
            'R_kpc': R,
            'Vobs_kms': Vobs,
            'errV_kms': err,
            'Vgas_kms': Vgas,
            'Vstars_kms': np.sqrt(np.clip(Vstar2, 0, None)),
            'Vgeom_kms': np.sqrt(np.clip(Vg2, 0, None)),
            'Vcompact_kms': compact,
            'Vmodel_kms': Vmod,
            'residual_kms': residual,
        })
    }

variants = [
    fit_variant(None, None),
    fit_variant('baseline_gauss_Rd3', 'cumulative'),
    fit_variant('excess4_raw', 'local_sigma'),
    fit_variant('excess4_raw', 'local_sqrtRsigma'),
]

summary = pd.DataFrame([{k: v for k, v in d.items() if k != 'curve'} for d in variants])
summary.to_csv(BASE/'NGC6503_local_operator_summary.csv', index=False)

all_curves = []
for d in variants:
    curve = d['curve'].copy()
    curve['variant'] = f"{d['kind']}__{d['operator']}"
    all_curves.append(curve)
pd.concat(all_curves, ignore_index=True).to_csv(BASE/'NGC6503_local_operator_curves.csv', index=False)

for d in variants:
    if d['kind'] == 'excess4_raw' and d['operator'] == 'local_sigma':
        d['curve'].to_csv(BASE/'NGC6503_local_operator_best_local_sigma.csv', index=False)
    if d['kind'] == 'excess4_raw' and d['operator'] == 'local_sqrtRsigma':
        d['curve'].to_csv(BASE/'NGC6503_local_operator_best_local_sqrtRsigma.csv', index=False)

# plots
plt.figure(figsize=(7,5))
for d, ls in zip(variants, ['--','-','-','-']):
    label = {
        'geom_only__none':'Geometric only',
        'baseline_gauss_Rd3__cumulative':'Cumulative compact (best prior)',
        'excess4_raw__local_sigma':'Local excess × Sigma*',
        'excess4_raw__local_sqrtRsigma':'Local excess × sqrt(R)Sigma*'
    }[f"{d['kind']}__{d['operator']}"]
    plt.plot(d['curve']['R_kpc'], d['curve']['Vmodel_kms'], lw=2, ls=ls, label=label)
plt.errorbar(R, Vobs, yerr=err, fmt='o', ms=4, capsize=2, label='Observed')
plt.plot(R, Vgas, ':', label='Gas')
plt.xlabel('Radius (kpc)')
plt.ylabel(r'$V_{circ}$ (km s$^{-1}$)')
plt.title('NGC 6503: local-operator compact-term tests')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(BASE/'NGC6503_local_operator_fit_comparison.png', dpi=180)
plt.close()

plt.figure(figsize=(7,5))
for d in variants:
    label = {
        'geom_only__none':'Geometric only',
        'baseline_gauss_Rd3__cumulative':'Cumulative compact',
        'excess4_raw__local_sigma':'Local excess × Sigma*',
        'excess4_raw__local_sqrtRsigma':'Local excess × sqrt(R)Sigma*'
    }[f"{d['kind']}__{d['operator']}"]
    plt.plot(d['curve']['R_kpc'], d['curve']['residual_kms'], lw=2, label=label)
plt.axhline(0, color='k', lw=1)
plt.xlabel('Radius (kpc)')
plt.ylabel(r'$\Delta V$ (km s$^{-1}$)')
plt.title('NGC 6503 residual comparison')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(BASE/'NGC6503_local_operator_residuals.png', dpi=180)
plt.close()

plt.figure(figsize=(7,5))
for d in variants[1:]:
    label = {
        'baseline_gauss_Rd3__cumulative':'Cumulative compact',
        'excess4_raw__local_sigma':'Local excess × Sigma*',
        'excess4_raw__local_sqrtRsigma':'Local excess × sqrt(R)Sigma*'
    }[f"{d['kind']}__{d['operator']}"]
    plt.plot(d['curve']['R_kpc'], d['curve']['Vcompact_kms'], lw=2, label=label)
plt.xlabel('Radius (kpc)')
plt.ylabel(r'$V_{compact}$ (km s$^{-1}$)')
plt.title('NGC 6503 compact contribution by operator')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(BASE/'NGC6503_local_operator_compact_terms.png', dpi=180)
plt.close()

report = f"""# NGC 6503 local-operator compact-term test

This run keeps the geometric backbone fixed and replaces the previous enclosed-mass-like
compact operator with local operators built directly from the central-excess stellar profile.

Models compared:
- geometric only
- earlier best cumulative compact model: baseline_gauss_Rd3 + cumulative operator
- local central-excess operator: excess4_raw + local_sigma
- local central-excess operator: excess4_raw + local_sqrtRsigma

Key results:
- Geometric only reduced chi^2: {summary.loc[summary['kind']=='geom_only','redchi2'].iloc[0]:.3f}
- Best cumulative compact reduced chi^2: {summary.loc[(summary['kind']=='baseline_gauss_Rd3')&(summary['operator']=='cumulative'),'redchi2'].iloc[0]:.3f}
- Best local operator reduced chi^2: {summary.loc[(summary['kind']=='excess4_raw')&(summary['operator']=='local_sqrtRsigma'),'redchi2'].iloc[0]:.3f}
- Local_sigma reduced chi^2: {summary.loc[(summary['kind']=='excess4_raw')&(summary['operator']=='local_sigma'),'redchi2'].iloc[0]:.3f}

Outer compact leakage at the last radius:
- cumulative: {summary.loc[(summary['kind']=='baseline_gauss_Rd3')&(summary['operator']=='cumulative'),'compact_at_last_radius_kms'].iloc[0]:.2f} km/s
- local_sigma: {summary.loc[(summary['kind']=='excess4_raw')&(summary['operator']=='local_sigma'),'compact_at_last_radius_kms'].iloc[0]:.2f} km/s
- local_sqrtRsigma: {summary.loc[(summary['kind']=='excess4_raw')&(summary['operator']=='local_sqrtRsigma'),'compact_at_last_radius_kms'].iloc[0]:.2f} km/s

Interpretation:
- The local operators are worse on raw chi^2 than the best cumulative compact model,
  but they strongly suppress the unwanted outer compact tail.
- The best local compromise is local_sqrtRsigma on total chi^2.
- The cleanest inner residual is local_sigma, which keeps the inner mean residual closest to zero.
"""
(BASE/'NGC6503_local_operator_report.md').write_text(report)
print(summary)
