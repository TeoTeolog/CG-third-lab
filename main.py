import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.special import comb
import matplotlib.widgets as widgets
import math

d = [
        [1.0, -0.3, 0.], [1.0, 0.1, 0.1], [1.8, 0.5, 0.1], [1.0, 0.8, 0.],
        [2., -0.3, 0.], [2.1, 0.1, 0.1], [2.1, 0.5, 0.1], [2.1, 0.8, 0.1],
        [2.3, -0.3, 0.1], [2.3, 0.1, 0.2], [2.3, 0.5, 0.1], [2.4, 0.8, 0.1],
        [2.5, -0.3, 0.1], [2.5, 0.1, 0.25], [2.5, 0.5, 0.1], [2.5, 0.8, 0.2],
    ]


def getPool():
    return np.array([
        [d[0], d[1], d[2], d[3]],
        [d[4], d[5], d[6], d[7]],
        [d[8], d[9], d[10], d[11]],
        [d[12], d[13], d[14], d[15]],
    ])


def bezier_matrix(d):
    return np.array([[(-1) ** (i - j) * comb(j, i) * comb(d, j) for i
                      in range(d + 1)] for j in range(d + 1)], int)


def draw(cps: np.ndarray, resol):
    u, v = np.linspace(0, 1, resol[0]), np.linspace(0, 1, resol[1])
    count_u, count_v, _ = cps.shape
    deg_u, deg_v = count_u - 1, count_v - 1
    u_vec = np.array([u ** i for i in range(count_u)])
    v_vec = np.array([v ** i for i in range(count_v)])
    BM_u, BM_v = DT[deg_u], DT[deg_v]
    cps_x = cps[:, :, 0]
    cps_y = cps[:, :, 1]
    cps_z = cps[:, :, 2]
    m1 = u_vec.T.dot(BM_u)
    m2 = BM_v.T.dot(v_vec)
    x = m1.dot(cps_x).dot(m2)
    y = m1.dot(cps_y).dot(m2)
    z = m1.dot(cps_z).dot(m2)
    return x, y, z


DT = [bezier_matrix(i) for i in range(16)]
fig: Figure = plt.figure(figsize=(8, 8))
ax: Axes3D = fig.add_subplot(111, projection='3d')
resol = 8, 8


def onChange(event):
    ax.clear()
    x1 = float(x1_box.text)
    y1 = float(y1_box.text)
    z1 = float(z1_box.text)
    number = int(dot_box.text) - 1
    dot = [x1, y1, z1]
    d[number] = dot
    x, y, z = draw(getPool(), resol=resol)
    ax.plot_surface(x, y, z, color='g', linewidth=1)
    d1 = np.array(d)
    ax.scatter(d1[:, 0], d1[:, 1], d1[:, 2], color='r')


def xRotateMatrix(a):
    return np.array([
        [1, 0, 0],
        [0, math.cos(math.radians(a)), -1 * math.sin(math.radians(a))],
        [0, math.sin(math.radians(a)), math.cos(math.radians(a))]
    ])


def yRotateMatrix(a):
    return np.array([
        [math.cos(math.radians(a)), 0, math.sin(math.radians(a))],
        [0, 1, 0],
        [-1 * math.sin(math.radians(a)), 0, math.cos(math.radians(a))]
    ])


def zRotateMatrix(a):
    return np.array([
        [math.cos(math.radians(a)), -1 * math.sin(math.radians(a)), 0],
        [math.sin(math.radians(a)), math.cos(math.radians(a)), 0],
        [0, 0, 1]
    ])


def calculateRotatedCords(cords, x_rotate, y_rotate, z_rotate):
    temp = cords.dot(xRotateMatrix(x_rotate))
    temp = temp.dot(yRotateMatrix(y_rotate))
    temp = temp.dot(zRotateMatrix(z_rotate))
    return temp


def onRotate(event):
    ax.clear()
    x1 = int(x1r_box.text)
    y1 = int(y1r_box.text)
    z1 = int(z1r_box.text)
    for i in range(16):
        d[i] = calculateRotatedCords(np.array(d[i]), x1, y1, z1)
    x, y, z = draw(getPool(), resol=resol)
    ax.plot_surface(x, y, z, color='g', linewidth=1)
    d1 = np.array(d)
    ax.scatter(d1[:, 0], d1[:, 1], d1[:, 2], color='r')


x, y, z = draw(getPool(), resol=resol)
ax.plot_surface(x, y, z, color='g', linewidth=1)
d1 = np.array(d)
ax.scatter(d1[:, 0], d1[:, 1], d1[:, 2], color='r')
print('perfomance was fucked')
dot_box = widgets.TextBox(fig.add_axes([0.1, 0.94, 0.07,
                                        0.03]), "Dot num", '1')
x1_box = widgets.TextBox(fig.add_axes([0.2, 0.94, 0.07, 0.03]), "x",
                         '1.0')
y1_box = widgets.TextBox(fig.add_axes([0.3, 0.94, 0.07, 0.03]), "y",
                         '-0.3')
z1_box = widgets.TextBox(fig.add_axes([0.4, 0.94, 0.07, 0.03]), "z",
                         '0.0')
start_btn = widgets.Button(fig.add_axes([0.5, 0.94, 0.1, 0.03]),
                           'Set')
x1r_box = widgets.TextBox(fig.add_axes([0.8, 0.94, 0.1,
                                        0.03]), "x rotate", '10')
y1r_box = widgets.TextBox(fig.add_axes([0.8, 0.90, 0.1,
                                        0.03]), "y rotate", '10')
z1r_box = widgets.TextBox(fig.add_axes([0.8, 0.86, 0.1,
                                        0.03]), "z rotate", '10')
rotate_btn = widgets.Button(fig.add_axes([0.8, 0.82, 0.1, 0.03]),
                            'Rotate')

rotate_btn.on_clicked(onRotate)
start_btn.on_clicked(onChange)
plt.show()
