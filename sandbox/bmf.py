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

def interp_model_spl(X, Y, Z):
    from scipy.interpolate import RectBivariateSpline
    return RectBivariateSpline(X, Y, Z)

def gen_mock():

    idir = "/scratch/isocdir/"
    nobs = 30
    nbg = 1000
    nmod = 2000000
    dmod = 16.0
    age = 10.08
    Z = 0.015/100

    ## Mock with some reasonable magnitude limits
    mmin, mmax = 12, 24
    cmin, cmax = -0.5, 2.0

    mycl = SSP(age, Z, mf='kroupa', isocdir=idir, a=[2.35], mlim=[0.4,1.2])
    _, ostars = mycl.populate(n=nobs)
    _, mstars = mycl.populate(n=nmod)

    ## Add some photometric errors
    g = ostars[:,3] + dmod
    g +=  np.random.randn(len(g))*sig(g, *ecoeff['des']['g'])
    r = ostars[:,4] + dmod
    r +=  np.random.randn(len(r))*sig(r, *ecoeff['des']['r'])

    bg = mmin + (mmax - mmin)*np.random.rand(nbg)
    bgr = cmin + (cmax - cmin)*np.random.rand(nbg)
    br = -(bgr - bg)
    #p.plot(bg-br, br, 'k.', ms=1)
    #p.show()
    #exit()

    I = (bg-br<cmax)&(bg-br>cmin)&(bg<mmax)&(bg>mmin)
    bg = bg[np.where(I)]
    br = br[np.where(I)]
    np.savetxt('sim_bg.dat', np.c_[bg,br], fmt='%.3f %.3f'.split())

    I = (g-r<cmax)&(g-r>cmin)&(g<mmax)&(g>mmin)
    g = g[np.where(I)]
    r = r[np.where(I)]
    np.savetxt('sim_cl.dat', np.c_[g,r], fmt='%.3f %.3f'.split())

    mg = mstars[:,3] + dmod
    mg +=  np.random.randn(len(mg))*sig(mg, *ecoeff['des']['g'])
    mr = mstars[:,4] + dmod
    mr +=  np.random.randn(len(mr))*sig(mr, *ecoeff['des']['r'])

    fcl,xl,yl = np.histogram2d(mg-mr, mg, bins=(80,160),
                               range=[[cmin,cmax],[mmin,mmax]], normed=True)

    np.savez('cl_model.npz', fcl=fcl, xl=xl, yl=yl)

def lhood(P, F, B, obs, N):

    l = P[0]
    g = P[1]
    col = obs[0]
    mag = obs[1]

    #f = F(col, mag)
    #b = B(col, mag)
    f = F
    b = B
    if l < 0 or g < 0 or l+g > len(col)+8*np.sqrt(len(col)):
        return -np.Inf
    else:

        prob = l*f/(l*f + g*b)
        prob = np.ma.array(prob)

        lnL = -(l + g) - np.sum(np.ma.log((1-prob)/(g*b)))
        return lnL

gen_mock()

g,r = np.loadtxt('sim_cl.dat', unpack=True)
bg,br = np.loadtxt('sim_bg.dat', unpack=True)
x = np.load('cl_model.npz')
fcl, xl, yl = x['fcl'], x['xl'], x['yl']

true_cl = len(g)
true_bg = len(bg)
true_ntot = true_cl + true_bg

print(true_cl, true_bg, true_ntot)

#p.plot(bg-br, bg, 'ro')
#p.plot(g-r, g, 'ko')
#p.show()

#g = np.r_[g,bg]
#r = np.r_[r,br]

g = bg
r = br

from scipy.ndimage import gaussian_filter as gs
fcl = gs(fcl, 2)
fcl /= np.sum(fcl)

fbg = np.ones_like(fcl)
fbg /= np.sum(fbg)

ext = [xl[0], xl[-1], yl[-1], yl[0]]
tx = (xl[1:] + xl[:-1])/2
ty = (yl[1:] + yl[:-1])/2

F = np.vectorize(interp_model_spl(tx,ty,fcl))
B = np.vectorize(interp_model_spl(tx,ty,fbg))

#nx = cmin + (cmax-cmin)*np.random.rand(10000)
#ny = mmin + (mmax-mmin)*np.random.rand(10000)
#tmpb = B(nx,ny)
#p.scatter(nx, ny, c=tmpb, s=30, lw=0)
#p.colorbar()
#p.show()
#tmpb = np.array(F(ny,nx))
#p.imshow(tmpb, extent=ext, vmin=0)
#p.colorbar()
#p.show()

c = g - r
m = g

C = c
M = m
f = F(C, M)
b = B(C, M)

Num = true_ntot + 8*np.sqrt(true_ntot)
ndim, nwalkers = 2, 8

p0 = [[true_cl + 5*np.random.randn(),true_bg + 5*np.random.randn()] for i in
      range(nwalkers)]

#p0 = [*np.random.rand(ndim) for i in range(nwalkers)]
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lhood, args=[F, B, [C,M], Num],
#                                threads=8)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lhood, args=[f, b, [C,M], Num],
                                threads=8)

pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
sampler.run_mcmc(pos, 1500)

lbl = [r'$N_{cl}$', r'$N_{bg}$']

chain = sampler.flatchain
corner(chain, labels=lbl, truths=[true_cl, true_bg])

for i, n in enumerate(lbl):
    p.figure()
    p.plot(chain[:,i], 'k-', alpha=0.3)
    p.xlabel('Iter')
    p.ylabel(n)

    print(n, np.median(chain[:,i]), u'±', np.std(chain[:,i]))

p.figure()
p.plot(c, m, 'k.', ms=1)
p.ylim(p.ylim()[::-1])

p.show()
