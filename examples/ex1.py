#!/usr/bin/env python

import numpy as np
from pysbt import sbt

N    = 256
rmin = 2.0 / 1024 / 32
rmax = 30
rr   = np.logspace(np.log10(rmin), np.log10(rmax), N, endpoint=True)
beta = 2
f1   = 0.5*beta**3 *np.exp(-beta*rr)

ss = sbt(rr)                                        # init SBT
g1 = ss.run(f1, direction=1, norm=False)            # SBT

import matplotlib.pyplot as plt
# plt.style.use('ggplot')

fig = plt.figure(
  dpi=300,
  figsize=(4, 3.5),
)
ax = plt.subplot()

ax.loglog(
    ss.kk, g1,
   ls='-', color='r',
   label=r'$g(k)=\mathrm{SBT}\{\frac{\beta^3}{2} e^{-\beta r}\}$'
)
ax.loglog(
    ss.kk, beta**4/(ss.kk**2 + beta**2)**2,
    ls='--', color='b',
    label=r'$\dfrac{\beta^4}{(k^2+\beta^2)^2};\quad\beta=2$'
)

ax.legend(loc='lower left', fontsize='small')

ax.set_xlabel('$k$', labelpad=5)
ax.set_ylabel('$g(k)$', labelpad=5)


plt.tight_layout()
plt.savefig('ex1.svg')
plt.show()
