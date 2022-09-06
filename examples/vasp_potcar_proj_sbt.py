#!/usr/bin/env python

import numpy as np
from paw import pawpotcar

from pysbt import sbt

mo_pot = pawpotcar(potfile='mo.pot')
nproj_l = mo_pot.proj_l.size

N    = 512
rmin = 2.0 / 1024 / 32
rmax = mo_pot.proj_rgrid[-1]
rr   = np.logspace(np.log10(rmin), np.log10(rmax), N, endpoint=True)
f1   = [mo_pot.spl_rproj[ii](rr) for ii in range(nproj_l)]

ss = sbt(rr)
g1 = [
    4*np.pi*ss.run(f1[ii], l=mo_pot.proj_l[ii]) for ii in range(nproj_l)
]

import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
plt.style.use('ggplot')

prop_cycle = plt.rcParams['axes.prop_cycle']
cc         = prop_cycle.by_key()['color']


fig, axes = plt.subplots(
    nrows=nproj_l, ncols=2,
    dpi=300,
    figsize=(7.2, 6.4)
)

for ii in range(nproj_l):
    axes[ii,0].plot(
        mo_pot.proj_rgrid, mo_pot.rprojs[ii],
        ls='none', marker='o', mew=0.5, mfc='w', ms=4, color=cc[ii], 
        label='vasp')
    axes[ii,0].plot(rr, f1[ii], ls='-', lw=1.0, color=cc[ii],
        label=f'interp')
        # label=f'$\ell = {mo_pot.proj_l[ii]}$')

    # axes[ii,0].set_ylabel(f'$f_{mo_pot.proj_l[ii]}(r)$')
    axes[ii,0].legend(loc='upper right', fontsize='small')

    axes[ii,1].plot(
        mo_pot.proj_qgrid, mo_pot.qprojs[ii],
        ls='none', marker='o', mew=0.5, mfc='w', ms=4, color=cc[ii],
        label='vasp')
    axes[ii,1].plot(ss.kk, g1[ii], ls='-', lw=1.0, color=cc[ii],
        label=f'pysbt')
        # label=f'$\ell = {mo_pot.proj_l[ii]}$')

    axes[ii,1].legend(loc='upper right', fontsize='small')

    axes[ii,1].set_xlim(-0.05 * mo_pot.proj_gmax, 1.05 * mo_pot.proj_gmax)

    if ii < nproj_l - 1:
        axes[ii,0].set_xticklabels([])
        axes[ii,1].set_xticklabels([])

    for ax in axes[ii]:
        ax.text(0.05, 0.90, f'$\ell = {mo_pot.proj_l[ii]}$',
                ha="left",
                va="top",
                fontsize='small',
                # family='monospace',
                # fontweight='bold'
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.4)
        )
        

axes[-1, 0].set_xlabel(r'$r$ [$\AA$]', labelpad=5)
axes[-1, 1].set_xlabel(r'$q$ [$\AA^{-1}$]', labelpad=5)

axes[0, 0].set_title(r'$f_\ell(r)$', fontsize='medium')
axes[0, 1].set_title(r'$g_\ell(q)$', fontsize='medium')

plt.tight_layout()
plt.savefig('vasp.svg')
# plt.show()

from subprocess import call
call('eog vasp.svg'.split())

