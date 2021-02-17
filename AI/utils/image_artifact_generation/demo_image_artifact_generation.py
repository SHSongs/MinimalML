##
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.transform import radon, iradon, rescale

matplotlib.use('Qt5Agg')

##
img = plt.imread("lenna.png")

# gray image
# img = np.mean(img, axis=2, keepdims=True)

sz = img.shape

cmap = "gray" if sz[2] == 1 else None

plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")
plt.show()


##
def print_imgs():
    plt.subplot(131)
    plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
    plt.title("Ground Truth")

    plt.subplot(132)
    plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
    plt.title("Uniform sampling mask")

    plt.subplot(133)
    plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
    plt.title("Sampling image")

    plt.show()

def print_imgs_args (i1, i2, i3):
    plt.subplot(131)
    plt.imshow(np.squeeze(i1), cmap=cmap, vmin=0, vmax=1)
    plt.title("Ground Truth")

    plt.subplot(132)
    plt.imshow(np.squeeze(i2), cmap=cmap, vmin=0, vmax=1)
    plt.title("Uniform sampling mask")

    plt.subplot(133)
    plt.imshow(np.squeeze(i3), cmap=cmap, vmin=0, vmax=1)
    plt.title("Sampling image")

    plt.show()
## Uniform sampling
ds_y = 2
ds_x = 4

msk = np.zeros(sz)
msk[::ds_y, ::ds_x, :] = 1

dst = img*msk

print_imgs()

## random sampling

rnd = np.random.rand(sz[0], sz[1], sz[2])

prob = 0.5

msk = (rnd > prob).astype(np.float)

dst = img*msk

print_imgs()

## Gaussian sampling
ly = np.linspace(-1, 1, sz[0])
lx = np.linspace(-1, 1, sz[1])

x, y = np.meshgrid(lx, ly)

x0 = 0
y0 = 0
sgmx = 1
sgmy = 1

a = 1
#
# gaus = a * np.exp(-((x - x0)**2 / (2 * sgmx**2) + (y - y0)**2/(2*sgmy**2)))
# gaus = np.tile(gaus[:, :, np.newaxis], (1,1,sz[2]))
# plt.imshow(gaus)
# plt.show()
# rnd = np.random.rand(sz[0], sz[1], sz[2])
# msk = (rnd < gaus).astype(np.float)


gaus = a * np.exp(-((x - x0)**2 / (2 * sgmx**2) + (y - y0)**2/(2*sgmy**2)))
gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, 1))
plt.imshow(gaus)
plt.show()
rnd = np.random.rand(sz[0], sz[1], 1)
msk = (rnd < gaus).astype(np.float)
msk = np.tile(msk, (1, 1, sz[2]))

dst = img*msk

print_imgs()



## random noise

sgm = 80.0

noise = sgm/255.0 * np.random.randn(sz[0], sz[1], sz[2])

dst = img + noise

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(noise), cmap=cmap, vmin=0, vmax=1)
plt.title("Random Noise")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Noisy image with %.2f sigma" % sgm)

plt.show()


## super-resolution

"""
----------------------
order options
----------------------
0: Nearest-neighbor
1: Bi-linear (default)
2: Bi-quadratic
3: Bi-cubic
4: Bi-quartic
5: Bi-quintic
"""

dw = 1/5.0
order = 1

dst_dw = rescale(img, scale=(dw, dw, 1), order=order)
dst_up = rescale(dst_dw, scale=(1/dw, 1/dw, 1), order=order)



plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(dst_dw), cmap=cmap, vmin=0, vmax=1)
plt.title("Downscaled image")

plt.subplot(133)
plt.imshow(np.squeeze(dst_up), cmap=cmap, vmin=0, vmax=1)
plt.title("Upscaled image")

plt.show()












