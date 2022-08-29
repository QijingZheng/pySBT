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
