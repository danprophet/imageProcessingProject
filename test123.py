import matplotlib.pyplot as plt
import numpy as np
import string

# points a, b and, c
a, b, c, d = (0, 1, 0), (1, 0, 1), (0, -1, 2), (-1, 0, 3)

# matrix with row vectors of points
A = np.array([a, b, c, d])

B = np.array(
    [255,255,255,255,255,255],
    [255,255,255,255,255,255],
    [255,255,255,255,255,255],
    [255,255,255,255,255,255],
    [255,255,255,255,255,255]
)

T_filter_scale_2x = np.array(
    [2,0,0],
    [0,2,0],
    [0,0,1]
)

T_filter_scale_halfx = np.array(
    [0.5,0,0],
    [0,0.5,0],
    [0, 0 ,1]
)

[1,2,0] @ T_filter_scale_halfx

# 3x3 Identity transformation matrix
I = np.eye(3)

color_lut = 'rgbc'
fig = plt.figure()
ax = plt.gca()
xs = []
ys = []
# for row in A:
#     output_row = I @ row
#     x, y, i = output_row
#     xs.append(x)
#     ys.append(y)
#     i = int(i) # convert float to int for indexing
#     c = color_lut[i]
#     plt.scatter(x, y, color=c)
#     plt.text(x + 0.15, y, f"{string.ascii_letters[i]}")
# xs.append(xs[0])
# ys.append(ys[0])
# plt.plot(xs, ys, color="gray", linestyle='dotted')
# ax.set_xticks(np.arange(-2.5, 3, 0.5))
# ax.set_yticks(np.arange(-2.5, 3, 0.5))
# plt.grid()
# plt.show()


# create the scaling transformation matrix
T_s = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])

fig = plt.figure()
ax = plt.gca()
xs_s = []
ys_s = []
for row in A:
    output_row = T_s @ row
    x, y, i = row
    x_s, y_s, i_s = output_row
    xs_s.append(x_s)
    ys_s.append(y_s)
    i, i_s = int(i), int(i_s) # convert float to int for indexing
    c, c_s = color_lut[i], color_lut[i_s] # these are the same but, its good to be explicit
    plt.scatter(x, y, color=c)
    plt.scatter(x_s, y_s, color=c_s)
    plt.text(x + 0.15, y, f"{string.ascii_letters[int(i)]}")
    plt.text(x_s + 0.15, y_s, f"{string.ascii_letters[int(i_s)]}'")

xs_s.append(xs_s[0])
ys_s.append(ys_s[0])
plt.plot(xs, ys, color="gray", linestyle='dotted')
plt.plot(xs_s, ys_s, color="gray", linestyle='dotted')
ax.set_xticks(np.arange(-2.5, 3, 0.5))
ax.set_yticks(np.arange(-2.5, 3, 0.5))
plt.grid()
plt.show()