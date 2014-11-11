#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import numpy as np
from scipy.misc import factorial
import matplotlib.pyplot as p

from gencmd.model import SSP
from gencmd.util.phot_sys import col_dict, ecoeff

import emcee
from triangle import corner

def sig(x, a, b, c):
    return a + np.exp((x-b)/c)

def interp_model(X,Y,Z):
    from scipy.interpolate import interp2d
    return interp2d(X, Y, Z, kind='linear', fill_value=0, bounds_error=False)

def lhood(P, F, B, obs):
    g_tot = P[0]
    g_bg = P[1]
    if g_tot < 0 or g_bg < 0 or g_bg > g_tot:
        return -np.Inf
    else:
        col = obs[0]
        mag = obs[1]
        mod = g_bg*B(mag, col) + (g_tot-g_bg)*F(mag, col)
        mod = np.ma.array(mod)
        # spectral-element-wise likelihood
        #lnL = N*np.log(mod) - mod - np.log(factorial(N))
        lnL = np.ma.log(mod) - mod
        return np.sum(lnL)

idir = "/scratch/isocdir/"
nobs = 1000
nbg = 10
nmod = 2000000
dmod = 16.0
age = 10.08
Z = 0.015/100

## Mock with some reasonable magnitude limits
mmin, mmax = 12, 24
cmin, cmax = -0.5, 2.0

bg = mmin + (mmax-mmin)*np.random.rand(nbg)
br = mmin + (mmax-mmin)*np.random.rand(nbg)

mycl = SSP(age, Z, mf='kroupa', isocdir=idir, a=[2.35], mlim=[0.4,2.0])
_, ostars = mycl.populate(n=nobs)
_, mstars = mycl.populate(n=nmod)

## Add some photometric errors
g = ostars[:,3] + dmod
g +=  np.random.randn(len(g))*sig(g, *ecoeff['des']['g'])
r = ostars[:,4] + dmod
r +=  np.random.randn(len(r))*sig(r, *ecoeff['des']['r'])

I = (g-r<cmax)&(g-r>cmin)&(g<mmax)&(g>mmin)
I = (bg-br<cmax)&(bg-br>cmin)&(bg<mmax)&(bg>mmin)
true_bg = len(bg[I])

## background stars
g = np.r_[g,bg]
#print(g)
r = np.r_[r,br]

mg = mstars[:,3] + dmod
mg +=  np.random.randn(len(mg))*sig(mg, *ecoeff['des']['g'])
mr = mstars[:,4] + dmod
mr +=  np.random.randn(len(mr))*sig(mr, *ecoeff['des']['r'])

fcl,xl,yl = np.histogram2d(mg-mr, mg, bins=(60,120),
                           range=[[cmin,cmax],[mmin,mmax]], normed=True)

fbg = np.ones_like(fcl)
fbg /= np.sum(fbg)

ext = [xl[0], xl[-1], yl[-1], yl[0]]

tx = (xl[1:] + xl[:-1])/2
ty = (yl[1:] + yl[:-1])/2
F = np.vectorize(interp_model(ty,tx,fcl))
B = np.vectorize(interp_model(ty,tx,fbg))

c = g-r
m = g

I = (c<cmax)&(c>cmin)&(m<mmax)&(m>mmin)
C = c[np.where(I)]
M = m[np.where(I)]
true_ntot = len(C)
#p.scatter(C,M,c=F(M,C),s=100)
#p.show()
#print(lhood([10.,100.], F, B, np.array([c,m])))
ndim, nwalkers = 2, 8
p0 = [10*np.random.rand(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lhood, args=[F, B, [C,M]],
                                threads=8)

pos, prob, state = sampler.run_mcmc(p0, 50)
sampler.reset()
sampler.run_mcmc(pos, 100)

chain = sampler.flatchain

corner(chain, labels=[r'$N_{ntot}$', r'$N_{bg}$'], truths=[true_ntot,true_bg])
p.show()

#nx = np.arange(cmin, cmax, 0.01)
#ny = np.arange(mmin, mmax, 0.01)
#tmpb = np.array(B(ny,nx))
#p.imshow(tmpf.T, extent=ext)
#p.colorbar()
#p.show()






