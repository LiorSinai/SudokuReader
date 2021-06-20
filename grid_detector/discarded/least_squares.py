import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    grad = 3
    c = 4
    xlims = [2, 20]

    n = 100
    x = np.random.random(n) * (xlims[1] - xlims[0]) + xlims[0]

    y = np.zeros(n)
    for i in range(n):
        y[i] = (grad * (1 + 2 * np.random.random() - 1)) * x[i] + (c * (1 + 2 * np.random.random() - 1))

    fig, ax = plt.subplots()
    ax.plot(x, y, 'x')
    ax.plot(xlims, [xlims[0] * grad + c, xlims[1] * grad + c], label='original')

    # least-squares inverse
    X = np.hstack([x.reshape(-1, 1), np.ones((n, 1))])
    Y = y.reshape(-1, 1)
    theta = np.matmul(np.linalg.pinv(X), Y)
    ax.plot(xlims, [xlims[0] * theta[0] + theta[1], xlims[1] * theta[0] + theta[1]], label='least-squares')
    ax.legend()

    plt.show()