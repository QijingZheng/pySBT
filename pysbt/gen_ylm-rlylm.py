#!/usr/bin/env python

import numpy as np
from gpaw.gaunt import nabla

lmax = 3
nij  = np.asarray(nabla(lmax))

print(
'''
# The array of derivative intergrals.
# 
# :::
# 
#   /  ^    ^   1-l' d   l'    ^
#   | dr Y (r) r     --[r  Y  (r)]
#   /     L           _     L'
#                    dr
#
'''
)
print('_ylm_nabla_rlylm = {')
for l1 in range(lmax+1):
    for l2 in range(lmax+1):
        for m1 in range(-l1, l1+1):
            for m2 in range(-l2, l2+1):
                ii = l1**2 + l1 + m1
                jj = l2**2 + l2 + m2
                if not np.allclose(nij[ii,jj], 0):
                    print(f'  ({l1}, {m1:2d}, {l2}, {m2:2d}): [{nij[ii,jj,0]}, {nij[ii,jj,1]}, {nij[ii,jj,2]}],')
print('}')
