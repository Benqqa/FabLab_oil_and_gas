# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:32:39 2021

@author: maxbo
"""

import sympy as sym
import math
V=sym.symbols('V')
_sum=0
for i in range(5):
    _sum+=(i-1)/(V*(i-1)+1)
res2=list(sym.solveset(sym.Eq((_sum),0)))
print(res2[0]+100)

