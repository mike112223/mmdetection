import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator

p = np.arange(0.001, 1., 0.001)
y = np.arange(0.001, 1., 0.001)

P, Y = np.meshgrid(p, y)

QFL = - (Y - P) ** 2 * (Y * np.log(P) + (1 - Y) * np.log(1 - P))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(P, Y, QFL, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# fig.colorbar(surf, shrink=0.5, aspect=10)
plt.title('QFL')
ax.set_xlabel('p')
ax.set_ylabel('y')
plt.show()


CE = - (Y * np.log(P) + (1 - Y) * np.log(1 - P))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(P, Y, CE, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# fig.colorbar(surf, shrink=0.5, aspect=10)
plt.title('CE')
ax.set_xlabel('p')
ax.set_ylabel('y')
plt.show()


F = - (0.25 * Y * np.log(P) + 0.75 * (1 - Y) * np.log(1 - P))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(P, Y, F, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# fig.colorbar(surf, shrink=0.5, aspect=10)
plt.title('FL')
ax.set_xlabel('p')
ax.set_ylabel('y')
plt.show()


MSE = (Y - P) ** 2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(P, Y, MSE, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# fig.colorbar(surf, shrink=0.5, aspect=10)
plt.title('MSE')
ax.set_xlabel('p')
ax.set_ylabel('y')
plt.show()
