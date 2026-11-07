import numpy as np
from regression import make_data, gradients, loss, train

def test_gradients():
    x, y = make_data()

    w = 1.3
    b = -0.7
    h = 1e-5

    dw_numeric = (loss(w + h, b, x, y) - loss(w, b, x, y)) / h
    db_numeric = (loss(w, b + h, x, y) - loss(w, b, x, y)) / h
    dw_analytic, db_analytic = gradients(w, b, x, y)

    print("Numeric gradients:", dw_numeric, db_numeric)
    print("Analytic gradients:", dw_analytic, db_analytic)



    assert np.isclose(dw_numeric, dw_analytic, rtol=1e-6)
    assert np.isclose(db_numeric, db_analytic, rtol=1e-6)

def test_training_matches_polyfit():
    x, y = make_data()
    w, b = np.polyfit(x, y, 1)
    w_train, b_train, _ = train(x, y, steps=1000, lr=0.05)

    assert np.isclose(w_train, w, rtol=1e-6)
    assert np.isclose(b_train, b, rtol=1e-6)
