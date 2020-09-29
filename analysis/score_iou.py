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

neg_in_gt = np.asarray(results['neg_in_gt']).astype(np.bool)
pos_in_gt = np.asarray(results['pos_in_gt']).astype(np.bool)

pos_anchor_gt_assign = np.asarray(results['pos_anchor_gt_assign'])
neg_anchor_gt_assign = np.asarray(results['neg_anchor_gt_assign'])

gt_areas = np.asarray(results['gt_areas'])
gt_ws = np.asarray(results['gt_ws'])
gt_hs = np.asarray(results['gt_hs'])

bins = 50

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
ax = fig.add_subplot(111, projection='3d')
x, y = pos_scores, pos_ious
x = x[~pos_in_gt]
y = y[~pos_in_gt]
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
plt.title('pos iou-score distribution (outside)')
plt.xlabel('ious')
plt.ylabel('scores')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = pos_scores, pos_ious
x = x[pos_in_gt]
y = y[pos_in_gt]
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
plt.title('pos iou-score distribution (inside)')
plt.xlabel('ious')
plt.ylabel('scores')
plt.show()

##### ===================

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
x = x[neg_in_gt]
y = y[neg_in_gt]

hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])

hist[hist > 1e4] = 1e4
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
surf1 = ax.plot_surface(X, Y, hist, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf1, shrink=0.5, aspect=10)
plt.title('neg iou-score distribution (inside)')
ax.set_xlabel('ious')
ax.set_ylabel('scores')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
bins = 50
gap = 1 / bins

x, y = neg_scores, neg_ious
x = x[~neg_in_gt]
y = y[~neg_in_gt]

hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])

hist[hist > 1e4] = 1e4
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
surf1 = ax.plot_surface(X, Y, hist, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf1, shrink=0.5, aspect=10)
plt.title('neg iou-score distribution (outside)')
ax.set_xlabel('ious')
ax.set_ylabel('scores')
plt.show()


####
pos_gt_assign = np.bincount(pos_anchor_gt_assign)
plt.figure()
plt.hist(pos_gt_assign, bins=bins, alpha=0.7)
plt.title('neg score distribution (iou > 0.8)')
plt.xlabel('')
plt.show()

plt.figure()
plt.bar(np.arange(len(pos_gt_assign)), pos_gt_assign)
plt.title('neg score distribution (iou > 0.8)')
plt.xlabel('gt_idx')
plt.show()

mask = pos_gt_assign == 0
not_anchor_recall_gt_areas = gt_areas[mask]


recall_gt_assign = np.bincount(neg_anchor_gt_assign[neg_in_gt & (neg_ious > 0.7)])
plt.figure()
plt.hist(recall_gt_assign, bins=bins, alpha=0.7)
plt.title('neg score distribution (iou > 0.8)')
plt.xlabel('scores')
plt.show()

mask = recall_gt_assign == 0
not_prop_recall_gt_areas = gt_areas[mask]

pos_gt_assign + recall_gt_assign
ratio = pos_gt_assign[recall_gt_assign > 0] / (recall_gt_assign[recall_gt_assign > 0] + 1e-6)


###
max_iou = [[], []]
max_score = [[], []]
for i in range(max(pos_anchor_gt_assign)):
    mask = pos_anchor_gt_assign == i
    if mask.sum() > 0:
        iou_max_idx = np.argmax(pos_ious[mask])
        score_max_idx = np.argmax(pos_scores[mask])

        max_iou[0].append(pos_ious[mask][iou_max_idx])
        max_iou[1].append(pos_scores[mask][iou_max_idx])

        max_score[0].append(pos_ious[mask][score_max_idx])
        max_score[1].append(pos_scores[mask][score_max_idx])


plt.figure()
plt.scatter(max_iou[1], max_iou[0], alpha=0.1)
plt.title('max iou')
plt.xlabel('scores')
plt.ylabel('ious')
plt.show()


plt.figure()
plt.scatter(max_score[1], max_score[0], alpha=0.1)
plt.title('max score')
plt.xlabel('scores')
plt.ylabel('ious')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = max_iou[1], max_iou[0]
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
plt.title('max iou distribution')
plt.xlabel('ious')
plt.ylabel('scores')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = max_score[1], max_score[0]
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
plt.title('max score distribution')
plt.xlabel('ious')
plt.ylabel('scores')
plt.show()
