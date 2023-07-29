import numpy as np
import matplotlib.pyplot as plt

"""
Distance transform implementaion for 2D numpy array
Reference:
Distance Transforms of Sampled Functions
https://theoryofcomputing.org/articles/v008a019/v008a019.pdf
"""

inf = 1e10
def distance_transform_1D(f):
    n = f.size
    para_loc = np.zeros(n)
    bounds_loc = np.zeros(n + 1)
    dist = np.zeros(n)
    right_most_para = 0
    bounds_loc[0] = -inf
    bounds_loc[1] = inf
    for q in range(1, n):
        s = ((f[q] + q*q) - (f[int(para_loc[right_most_para])] + para_loc[right_most_para]*para_loc[right_most_para]))/(2*q - 2*para_loc[right_most_para])
        while (s <= bounds_loc[right_most_para]):
            right_most_para = right_most_para - 1
            s = ((f[q] + q*q) - (f[int(para_loc[right_most_para])] + para_loc[right_most_para]*para_loc[right_most_para]))/(2*q - 2*para_loc[right_most_para])
        right_most_para = right_most_para + 1
        para_loc[right_most_para] = q
        bounds_loc[right_most_para] = s
        bounds_loc[right_most_para + 1] = inf
    right_most_para = 0
    for q in range(0, n):
        while (bounds_loc[right_most_para + 1] < q):
            right_most_para = right_most_para + 1
        dist[q] = (q - para_loc[right_most_para])*(q - para_loc[right_most_para]) + f[int(para_loc[right_most_para])]
    return dist

def distance_transform_2D(image):
    row = image.shape[0]
    col = image.shape[1]
    for x in range(0, row):
        image[x, :] = distance_transform_1D(image[x, :])
    for y in range(0, col):
        image[:, y] = distance_transform_1D(image[:, y])
    return np.sqrt(image)

def distance_transform(image):
    image[image == 0] = inf
    image[image == 1] = 0
    return distance_transform_2D(image)

img = np.zeros((100, 100))
img[30][30] = 1
img[70][70] = 1
dist = distance_transform(img)

x = np.arange(0.0, dist.shape[0], 1.0)
y = np.arange(0.0, dist.shape[1], 1.0)
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots(figsize=(10,10))
ax.set_aspect('equal')
cf = ax.contourf(X, Y, dist, levels=50)
fig.colorbar(cf, ax=ax)
plt.show()
