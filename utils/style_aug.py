import numpy as np
import random

_NTIMES = 1000
_t = np.linspace(0.0, 1.0, _NTIMES)
_BERNSTEIN_BASIS = np.array([
    (1 - _t) ** 3,
    3 * _t * (1 - _t) ** 2,
    3 * _t ** 2 * (1 - _t),
    _t ** 3,
])


def bezier_curve(points):
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    xvals = np.dot(xPoints, _BERNSTEIN_BASIS)
    yvals = np.dot(yPoints, _BERNSTEIN_BASIS)
    return xvals, yvals


def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()],
              [random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points)
    if random.random() < 0.5:
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


def nonlinear_transformation_multi_channel(image, prob=0.5):
    if random.random() >= prob:
        return image
    out = np.zeros_like(image)
    for c in range(image.shape[-1]):
        out[..., c] = nonlinear_transformation(image[..., c], prob=1.0)
    return out
