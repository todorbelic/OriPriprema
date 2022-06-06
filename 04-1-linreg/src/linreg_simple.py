import random
import matplotlib.pyplot as plt


def linear_regression(x, y):
    slope = 0.0  # nagib linije
    intercept = 0.0  # tacka preseka na y-osi
    # TODO 1: implementirati linearnu regresiju
    assert(len(x) == len(y))
    n = len(x)
    x_mean = float(sum(x)) / n
    y_mean = float(sum(y)) / n

    X = [xi - x_mean for xi in x]
    Y = [yi - y_mean for yi in y]

    numerator = sum([i * j for i, j in zip(X, Y)])
    denominator = sum([i ** 2 for i in X])

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    return slope, intercept


def predict(x, slope, intercept):
    # TODO 2: implementirati racunanje y na osnovu x i parametara linije

    return intercept + slope * x


def create_line(x, slope, intercept):
    y = [predict(xx, slope, intercept) for xx in x]
    return y


if __name__ == '__main__':
    x = range(50)  # celobrojni interval [0,50]
    random.seed(1337)  # da rezultati mogu da se reprodukuju
    y = [(i + random.randint(-5, 5)) for i in x]  # y = x (+- nasumicni sum)

    slope, intercept = linear_regression(x, y)

    line_y = create_line(x, slope, intercept)

    plt.plot(x, y, '.')
    plt.plot(x, line_y, 'b')
    plt.title('Slope: {0}, intercept: {1}'.format(slope, intercept))
    plt.show()
