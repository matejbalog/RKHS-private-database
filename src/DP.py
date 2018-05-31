import numpy as np


def Laplace_mechanism(length, L1_sensitivity, epsilon):
    noise = np.random.laplace(scale=L1_sensitivity / epsilon, size=length)
    return noise


def Gaussian_mechanism(length, L2_sensitivity, epsilon, delta):
    noise_variance = 2 * np.log(1.25 / delta) * (L2_sensitivity ** 2) / (epsilon ** 2)
    noise = np.sqrt(noise_variance) * np.random.randn(length)
    return noise
