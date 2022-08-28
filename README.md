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
