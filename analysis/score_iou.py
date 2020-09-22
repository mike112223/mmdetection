from mpl_toolkits.mplot3d import Axes3D                                                                                                                                                                     
import json

import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

results = json.load(open('result.json', 'r'))

neg_ious = np.asarray(results['neg_ious'])
pos_ious = np.asarray(results['pos_ious'])

neg_scores = np.asarray(results['neg_scores'])
pos_scores = np.asarray(results['pos_scores'])

in_gt = np.asarray(results['in_gt']).astype(np.bool)

plt.figure()
plt.scatter(pos_scores, pos_ious, alpha=0.01)
plt.title('pos score-iou distribution')
plt.xlabel('scores')
plt.ylabel('ious')
plt.show()


plt.figure()
plt.scatter(neg_scores, neg_ious, alpha=0.01)
plt.title('neg score-iou distribution')
plt.xlabel('scores')
plt.ylabel('ious')
plt.show()

mask = neg_ious > 0.5
plt.figure()
plt.hist(neg_scores[mask], bins=bins, alpha=0.7)
plt.title('neg score distribution (iou > 0.5)')
plt.xlabel('scores')
plt.show()


mask = neg_ious > 0.8
plt.figure()
plt.hist(neg_scores[mask], bins=bins, alpha=0.7)
plt.title('neg score distribution (iou > 0.8)')
plt.xlabel('scores')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = pos_scores, pos_ious
bins = 50
gap = 1 / bins

hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])

X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
surf = ax.plot_surface(X, Y, hist, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.title('pos iou-score distribution')
plt.xlabel('ious')
plt.ylabel('scores')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
bins = 50
gap = 1 / bins

x, y = neg_scores, neg_ious
hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])

hist[hist > 1e4] = 1e4
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
surf1 = ax.plot_surface(X, Y, hist, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf1, shrink=0.5, aspect=10)
plt.title('neg iou-score distribution')
ax.set_xlabel('ious')
ax.set_ylabel('scores')
plt.show()


fig = plt.figure()
ax = fig.gca(projection='3d')
bins = 50
gap = 1 / bins

x, y = neg_scores, neg_ious
x = x[in_gt]
y = y[in_gt]

hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])

# hist[hist > 1e4] = 1e4
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
surf1 = ax.plot_surface(X, Y, hist, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf1, shrink=0.5, aspect=10)
plt.title('neg iou-score distribution')
ax.set_xlabel('ious')
ax.set_ylabel('scores')
plt.show()




fig = plt.figure()
ax = fig.gca(projection='3d')
bins = 50
gap = 1 / bins

x, y = neg_scores, neg_ious
x = x[~in_gt]
y = y[~in_gt]

hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])

# hist[hist > 1e4] = 1e4
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
surf1 = ax.plot_surface(X, Y, hist, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf1, shrink=0.5, aspect=10)
plt.title('neg iou-score distribution')
ax.set_xlabel('ious')
ax.set_ylabel('scores')
plt.show()
