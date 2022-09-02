# *py*SBT



A python implementation of spherical Bessel transform (SBT) in O(Nlog(N)) time based on the algorithm proposed by J. Talman.


> "NumSBT: A subroutine for calculating spherical Bessel transforms numerically",
> [Talman, *J. Computer Physics Communications*, 2009, 180, 332-338.](https://www.sciencedirect.com/science/article/pii/S0010465508003329)

For a function f(r) defined on the *logarithmic* radial grid, the spherical Bessel transform f(r) -> g(k) and the inverse-SBT (iSBT) g(k) -> f(r) are give by

![spherical Bessel transform](img/sbt.svg) 

## Installation

To install pySBT using pip run:

```bash
pip install https://github.com/QijingZheng/pySBT
```

## Examples

- The example shows the the SBT of an exponential decaying function f(r).

  ![Eq (eq2)](img/exp_sbt.svg)
  
  ```python
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
  ```
  
  ![example 1](examples/ex1.svg)

  The full code can be found in [examples/ex1.py](examples/ex1.py).

- Overlap integral of molecular orbitals.  Suppose \psi_{lm}(r) is a molecular
  orbital composed of radial part R_{l,m}(r) and angular part Y_{lm}(\theta,
  \phi), then the overlap integral of two molecular orbitals can be written as

  ![overlap integral](img/overlap_integral.svg)

  where G is the Gaunt coefficients and g_{l1} and g_{l2} are the SBT of the two
  molecular orbitals, respectively. The above formula shows that the overlap
  integral can be obtained by inverse-SBT of g_{l1} * g_{l2}. For hydrogen 1s
  orbital

  ![hydrogen 1s orbital 1](img/hydrogen_1s.svg)

  The overlap integral can be obtained analytically

  ![overlap integral of hydrogen 1s orbitals](img/s10.svg)

  ```python
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
  ```
  ![S10 numeric](examples/pysbt_s10.svg)
