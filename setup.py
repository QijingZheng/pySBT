#!/usr/bin/env python

from distutils.core import setup

setup(
    name         = 'pySBT',
    version      = '1.0',
    description  = 'A python implementation of spherical Bessel transform (SBT) based on algorithm proposed by J. Talman.'
    author       = 'Qijing Zheng',
    author_email = 'zqj@ustc.edu.cn',
    url          = 'https://github.com/QijingZheng/pySBT',
    py_modules   = [
        'pysbt',
    ],
)
