#!/usr/bin/env python

import numpy as np
from pysbt import sbt
from pysbt import GauntTable

N    = 256
rmin = 2.0 / 1024 / 32
rmax = 30
rr   = np.logspace(np.log10(rmin), np.log10(rmax), N, endpoint=True)
f1   = 2*np.exp(-rr)                                # R_{10}(r)
Y10  = 1 / 2 / np.sqrt(np.pi)

ss = sbt(rr)                                        # init SBT
g1 = ss.run(f1, direction=1, norm=True)             # SBT
s1 = ss.run(g1 * g1, direction=-1, norm=False)      # iSBT

sR = 4*np.pi * GauntTable(0, 0, 0, 0, 0, 0) * Y10 * s1

import matplotlib.pyplot as plt
plt.style.use('ggplot')

fig = plt.figure(
  dpi=300,
  figsize=(4, 2.5),
)

ax = plt.subplot()

ax.plot(rr, sR, ls='none', color='b',
        marker='o', ms=3, mew=0.5, mfc='w',
        label='Numeric')
ax.plot(rr, np.exp(-rr) * (1 + rr + rr**2 / 3),
        color='r', lw=0.8, ls='--', label='Analytic')

ax.legend(loc='upper right')

ax.set_xlabel('$R$ [Bohr]', labelpad=5)
ax.set_ylabel('$S(R)$', labelpad=5)

plt.tight_layout()
plt.savefig('s10.svg')
plt.show()
