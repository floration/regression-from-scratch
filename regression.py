#Step 1: Generate the data

#Create regression.py with a function:
#It should return 200 points where y = 3x + 2 + noise. Use NumPy’s random generator (np.random.default_rng(seed), which you used in test_lesson6.py). 
#Pick a sensible spread for x and a small noise level.

#Step 2: Derive the gradients on paper (the core of the project)
#The model is ŷ = w·x + b, and the loss is the mean squared error:
#$$L = \frac{1}{n}\sum_i (w x_i + b - y_i)^2$$
#Work out ∂L/∂w and ∂L/∂b using the chain rule from Lesson 2. Hint: start with one point, (w·x + b − y)², and treat the inside as one “thing.” Then average over all the points.

import numpy as np
import matplotlib.pyplot as plt

def make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, size=n)
    y = x*3 + 2 + rng.normal(0, 1,size=n)

    return x, y

def gradients(w, b, x, y):
    e = (w * x + b - y)
    dw = np.mean(2 * e * x)
    db = np.mean(2 * e)
    return dw, db

def loss(w, b, x, y):
    return np.mean((w * x + b - y) ** 2)

def train(x, y, steps = 100, lr = 0.05):
    w = 0.0
    b = 0.0
    losses = []
    for step in range(steps):
        losses.append(loss(w, b, x, y))
        if step % 10 == 0:
            print(step, w, b, loss(w, b, x, y))
        dw, db = gradients(w, b, x, y)
        w -= lr * dw
        b -= lr * db
        
    return w, b, losses

if __name__ == "__main__":
    x, y = make_data()
    w, b, losses = train(x, y)
    print(type(losses))
    print("\n",w, b, losses[-1])
    plt.figure()                 # start a new, blank picture
    plt.plot(losses)     # one list → a line, x = 0, 1, 2, … automatically
    plt.yscale("log")
    plt.xlabel("step")           # axis labels
    plt.ylabel("loss")
    plt.savefig("example.png")   # write it to a file
    plt.close()                  # finish this picture before starting the next one
    plt.figure()
    plt.scatter(x, y, s=12, alpha=0.5, label="data")
    plt.plot(x, w * x + b, color="red", linewidth=2, label=f"fit: y = {w:.2f}x + {b:.2f}")
    plt.savefig("example2.png")
    plt.close()