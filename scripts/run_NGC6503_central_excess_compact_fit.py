
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares

BASE=Path("/mnt/data")
meta = pd.read_csv(BASE/'NGC2403_NGC6503_SPARC_metadata.csv').set_index('Galaxy')
m = meta.loc['NGC6503']
rot = pd.read_csv(BASE/'NGC6503_rotmod.dat', comment='#', sep=r'\s+', header=None,
                  names=["R","Vobs","errV","Vgas","Vdisk","Vbul","SBdisk_rot","SBbul_rot"])
dens = pd.read_csv(BASE/'NGC6503.dens', comment='#', sep=r'\s+', header=None,
                   names=["R","SBdisk","SBbul"])

R = rot.R.values.astype(float)
Vobs=rot.Vobs.values.astype(float); err=rot.errV.values.astype(float)
Vgas=rot.Vgas.values.astype(float); Vdisk=rot.Vdisk.values.astype(float); Vbul=rot.Vbul.values.astype(float)
Rdens=dens.R.values.astype(float)
SigmaL=(dens.SBdisk.values+dens.SBbul.values).astype(float)
Rd=float(m.Rdisk)

def cumtrap(x,y):
    out=np.zeros_like(x, dtype=float)
    out[1:] = np.cumsum(0.5*(y[1:]+y[:-1])*np.diff(x))
    return out

def geom_term(R,Vbar2,Ag):
    return Ag*cumtrap(R,Vbar2)/np.maximum(R,1e-12)

def fit_outer_exponential(rmin):
    sel=(Rdens>=rmin)&(SigmaL>0)
    x=Rdens[sel]; y=np.log(SigmaL[sel])
    A=np.vstack([np.ones_like(x),x]).T
    a,b=np.linalg.lstsq(A,y,rcond=None)[0]
    sigma_exp=np.exp(a+b*Rdens)
    return sigma_exp,a,b

Sigma_exp4,a4,b4 = fit_outer_exponential(4.0)
Sigma_exp5,a5,b5 = fit_outer_exponential(5.0)

def source_density(kind):
    if kind=='baseline_gauss_Rd3':
        Rc=Rd/3
        return SigmaL*np.exp(-(Rdens/Rc)**2)
    elif kind=='excess4_gauss_Rd3':
        Rc=Rd/3
        return np.clip(SigmaL-Sigma_exp4,0,None)*np.exp(-(Rdens/Rc)**2)
    elif kind=='excess4_raw':
        return np.clip(SigmaL-Sigma_exp4,0,None)
    else:
        raise KeyError(kind)

def compact_shape(kind):
    sd = source_density(kind)
    integrand = 2*np.pi*Rdens*sd
    Mraw = cumtrap(Rdens, integrand)
    shape = np.interp(R, Rdens, Mraw, left=Mraw[0], right=Mraw[-1])/np.maximum(R,1e-12)
    return shape

def fit_variant(kind):
    shape=compact_shape(kind)
    starts=[np.array([1.0,0.0,-1.0]), np.array([1.1,-1.0,-1.2]), np.array([0.8,1.0,-0.9])]
    bounds=([-4,-10,-2],[4,6,1.5])
    best=None
    for x0 in starts:
        def residuals(x):
            logAg, logCc, logY = x
            Ag=10**logAg; Cc=10**logCc; Y=10**logY
            Vstar2=Y*(Vdisk**2+Vbul**2)
            Vbar2=Vgas**2+Vstar2
            Vg2=geom_term(R,Vbar2,Ag)
            Vc2=Cc*Y*shape
            Vmod=np.sqrt(np.clip(Vbar2+Vg2+Vc2,0,None))
            return (Vobs-Vmod)/err
        fit=least_squares(residuals,x0,bounds=bounds,max_nfev=200000)
        if best is None or fit.cost<best.cost:
            best=fit
    return best

for variant in ["baseline_gauss_Rd3","excess4_gauss_Rd3","excess4_raw"]:
    fit = fit_variant(variant)
    print(variant, fit.cost, fit.x)
