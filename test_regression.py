#The finite-difference check estimates a gradient as:
#(L(w + h) − L(w)) / h for tiny h

import numpy as np
from regression import make_data, gradients, loss

x, y = make_data()

w = 0.0
b = 0.0
h = 1e-5

dw_numeric = (loss(w + h, b, x, y) - loss(w, b, x, y)) / h
db_numeric = (loss(w, b + h, x, y) - loss(w, b, x, y)) / h
dw_analytic, db_analytic = gradients(w, b, x, y)

print("Numeric gradients:", dw_numeric, db_numeric)
print("Analytic gradients:", dw_analytic, db_analytic)


def test_gradients():
    assert np.isclose(dw_numeric, dw_analytic, rtol=1e-4, atol=1e-6)
    assert np.isclose(db_numeric, db_analytic, rtol=1e-4, atol=1e-6)