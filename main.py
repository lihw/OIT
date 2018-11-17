import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from oit import * 

num_means = 50
num_sigmas = 50

means = np.linspace(0.01, 1.0, num_means)
sigmas = np.linspace(0.01, 1.0, num_sigmas)

num_colors = 100

errors = {\
        'meshkin': np.zeros((num_means, num_sigmas)),\
        'bavoil': np.zeros((num_means, num_sigmas)),\
        'mcguire': np.zeros((num_means, num_sigmas)),\
        'mcguire_depth': np.zeros((num_means, num_sigmas)),\
}

for i, mean in enumerate(means):
    for j, sigma in enumerate(sigmas):
        colors = np.zeros((num_colors, 4))

        colors[:, 0] = np.random.normal(mean, sigma, 100)
        colors[:, 1] = np.random.normal(mean, sigma, 100)
        colors[:, 2] = np.random.normal(mean, sigma, 100)
        colors[:, 3] = np.random.normal(mean, sigma, 100)

        # no negative colors or alphas
        colors[colors < 0] = 0.01
        colors[colors > 1] = 1
        
        # background color is always opaque.
        colors[0, 3] = 1

        if not np.all(colors <= 1):
            print(colors)

        assert np.all(colors[:, 3] > 0), "all color alphas positive"
        assert np.all(colors >= 0), "all colors positive"
        assert np.all(colors <= 1), "all colors smaller than 1"

        ground_truth = GroundTruth(colors)
        meshkin = Meshkin(colors)
        bavoil = Bavoil(colors)
        mcguire = Mcguire(colors)
        mcguire_depth = McguireDepth(colors)
		

        errors['meshkin'][i, j] = abs(np.sum(meshkin - ground_truth)) / np.sum(ground_truth)
        errors['bavoil'][i, j] = abs(np.sum(bavoil - ground_truth)) / np.sum(ground_truth)
        errors['mcguire'][i, j] = abs(np.sum(mcguire - ground_truth)) / np.sum(ground_truth)
        errors['mcguire_depth'][i, j] = abs(np.sum(mcguire_depth - ground_truth)) / np.sum(ground_truth)

        #print(i, j)

print('mean:', np.mean(errors['meshkin']), 'max:', np.max(errors['meshkin']), 'min:', np.min(errors['meshkin']))

errors['meshkin'][errors['meshkin'] > 100] = 100

# Draw the errors
# Reference https://www.scipy-lectures.org/intro/matplotlib/index.html

fig = plt.figure()

X = means
Y = sigmas
X, Y = np.meshgrid(X, Y)

ax = fig.add_subplot(2, 2, 1, projection='3d', title='meshkin')
ax.set_zlim([0, 100])
ax.plot_surface(X, Y, errors['meshkin'], rstride=1, cstride=1, cmap='hot')

ax = fig.add_subplot(2, 2, 2, projection='3d', title='bavoil')
ax.set_zlim([0, 100])
ax.plot_surface(X, Y, errors['bavoil'], rstride=1, cstride=1, cmap='hot')

ax = fig.add_subplot(2, 2, 3, projection='3d', title='mcguire')
ax.set_zlim([0, 100])
ax.plot_surface(X, Y, errors['mcguire'], rstride=1, cstride=1, cmap='hot')

ax = fig.add_subplot(2, 2, 4, projection='3d', title='mcguire_depth')
ax.set_zlim([0, 100])
ax.plot_surface(X, Y, errors['mcguire_depth'], rstride=1, cstride=1, cmap='hot')

plt.savefig('error.pdf')
