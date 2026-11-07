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

def train(x, y, steps = 101, lr = 0.05):
    print(f"{'step':>5} {'w':>8} {'b':>8} {'loss':>10}")
    print("-" * 34)
    w = 0.0
    b = 0.0
    losses = []
    for step in range(steps):
        losses.append(loss(w, b, x, y))
        if step % 20 == 0:
            print(f"{step:5d} {w:8.4f} {b:8.4f} {loss(w, b, x, y):10.6f}")
        dw, db = gradients(w, b, x, y)
        w -= lr * dw
        b -= lr * db
        
    return w, b, losses



if __name__ == "__main__":
    x, y = make_data()
    w, b, losses = train(x, y)
    xs = np.array([-5, 5])

    plt.figure()
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.legend(["loss"])
    plt.savefig("loss.png")
    plt.close()

    plt.figure()
    plt.scatter(x, y, s=12, alpha=0.5, label="data")
    plt.plot(xs, w * xs + b, color="red", linewidth=2, label=f"fit: y = {w:.2f}x + {b:.2f}")
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.title("Linear regression fit")
    plt.legend()
    plt.savefig("fit.png")
    plt.close()