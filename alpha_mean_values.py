import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# alpha_mean and resulting sparsity values
alpha_means = np.array([0.1500, 0.1334, 0.1169, 0.1003, 0.0838, 0.0672, 0.0507, 0.0341, 0.0176, 0.0010])
sparsities = np.array([0.3645, 0.3765, 0.3843, 0.3966, 0.4090, 0.4384, 0.4686, 0.5273, 0.6342, 0.8972])

# Interpolate alpha_mean as a function of sparsity
interp_func = interp1d(sparsities, alpha_means, kind='linear', fill_value="extrapolate")

# Generate linearly spaced sparsity values
target_sparsities = np.linspace(0.36, 0.9, 10)
estimated_alpha_means = interp_func(target_sparsities)
