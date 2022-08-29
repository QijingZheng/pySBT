#!/usr/bin/env python

import numpy as np
from pysbt import sbt

N    = 256
rmin = 2.0 / 1024 / 32
rmax = 30
rr   = np.logspace(np.log10(rmin), np.log10(rmax), N, endpoint=True)
f1   = 2*np.exp(-rr)                                # R_{10}(r)

ss = sbt(rr)                                        # init SBT
g1 = ss.run(f1, direction=1, norm=False)            # SBT
s1 = ss.run(g1 * g1, direction=-1, norm=False)      # iSBT

import matplotlib.pyplot as plt
# plt.style.use('ggplot')

fig = plt.figure(
  dpi=300,
  figsize=(4, 2.4),
)

ax = plt.subplot()

ax.plot(rr, s1 * 2 / np.pi, ls='-', color='r', label='Numeric')
ax.plot(rr, np.exp(-rr) * (1 + rr + rr**2 / 3), ls='--', label='Analytic')

ax.set_xlabel('$R$', labelpad=5)
ax.set_ylabel('$S(R)$', labelpad=5)

plt.tight_layout()
plt.savefig('pysbt_s10.svg')
plt.show()

