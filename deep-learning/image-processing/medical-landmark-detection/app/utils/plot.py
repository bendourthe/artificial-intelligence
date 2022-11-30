# --------------------------------------------------------------------------- #
# IMPORTS
# --------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, morphology
import numpy as np


# --------------------------------------------------------------------------- #
# METHODS DEFINITION
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
def plot_surface(filename, image, threshold=None, fill=False):
    # Position the scan upright so the end of the patient
    # is at the top (facing the camera)
    verts, faces, _, x = measure.marching_cubes_lewiner(image, threshold)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.savefig(filename)
    plt.close('all')


# --------------------------------------------------------------------------- #
def plot_scatter(filename, clusters, colors=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(clusters)):
        color = 'r' if colors is None else colors[i]
        ax.scatter(*clusters[i], c=color)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig(filename)
    plt.close('all')


# --------------------------------------------------------------------------- #
def plot_2d(filename, xs, ys, xlabel='', ylabel='', title=''):
    plt.figure(figsize=(16, 4))
    plt.rcParams['axes.xmargin'] = 0
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close('all')


# --------------------------------------------------------------------------- #
def visual_multi_channel(img, col_num=5):
    c, n1, n2 = img.shape
    row_num = (c + col_num - 1) // col_num
    padding = 10
    N1 = n1 * row_num + (row_num + 1) * padding
    N2 = n2 * col_num + (col_num + 1) * padding
    out = np.zeros((N1, N2), dtype=np.float)
    beg1 = padding
    for i in range(row_num):
        beg2 = padding
        for j in range(col_num):
            k = i * col_num + j
            if k < c:
                out[beg1:beg1 + n1, beg2:beg2 + n2] = img[k]
            beg2 += padding + n2
        beg1 += padding + n1

    return out
