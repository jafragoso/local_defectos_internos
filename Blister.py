import skvideo.io
import skvideo.utils
import skvideo.datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from scipy.signal import savgol_filter
from scipy.fftpack import fft, fftfreq
from timeit import default_timer as timer
from skimage.io import imsave, imread
from mpl_toolkits.mplot3d import Axes3D
from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu, threshold_local
from mpl_toolkits.mplot3d import axes3d


filename = skvideo.io.vread('/Users/Jus/Documents/V_capsulas.avi', outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
tiem, col, com = filename.shape
video_img_fria = np.zeros((tiem, col, com))
video_binary_local = np.zeros((tiem, col, com))
video_segm = np.zeros((tiem, col, com))
video_RDI = np.zeros((tiem, col, com))
curva = np.zeros((tiem))
curva2 = np.zeros((tiem))
x = np.linspace(0, tiem / 29, len(curva))
img_umbral = np.zeros((col, com))
img_fria = filename[1,:,:]

# extracción de imagen fria

for i in range(0, tiem):
    video_img_fria[i, :, :] = filename[i, :, :] - img_fria
    video_binary_local[i, :, :] = filename[i, :, :] < 50
    #video_binary_local[i, :, :] = video_binary_local[i, :, :] * 256

skvideo.io.vwrite("/Users/Jus/Documents/binario.mp4", video_binary_local)

for i in range(0, tiem):
    video_segm[i, :, :] = video_img_fria[i, :, :] * video_binary_local[100, :, :]

skvideo.io.vwrite("/Users/Jus/Documents/segmentado.mp4", video_segm)

imagen_prueba_seg = video_segm[600, :, :]
fila_prueba_seg = imagen_prueba_seg[37].tolist()
imagen_prueba = filename[600, :, :]
fila_prueba = imagen_prueba[37].tolist()
pixel_numero = np.linspace(0, 80, len(fila_prueba))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(col):
    columna = np.ones((com))*i
    fila = np.linspace(0, com, com)
    intensidad = imagen_prueba_seg[i].tolist()
    ax.plot(fila, columna, intensidad)

plt.show()


np.savetxt('/Users/Jus/Desktop/linea_prieba.csv', fila_prueba_seg)

"""
plt.figure(0)
plt.plot(pixel_numero, fila_prueba_seg); plt.title('Horizontal intensity')
plt.plot(pixel_numero, fila_prueba); plt.title('Horizontal intensity')

plt.xlabel('Column No.'); plt.ylabel('Intensity')

xx, yy = np.mgrid[0:imagen_prueba_seg.shape[0], 0:imagen_prueba_seg.shape[1]]

fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, imagen_prueba_seg, rstride=1, cstride=1, cmap=plt.cm.inferno, linewidth=0)

fig = plt.figure(2)
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, imagen_prueba, rstride=1, cstride=1, cmap=plt.cm.inferno, linewidth=0)


fig, axes = plt.subplots(2, 2, figsize=(10, 7))
ax = axes.ravel()

ax[0].set_title('1')
ax[0].imshow(imagen_prueba_seg, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_axis_off()

ax[1].set_title('2')
ax[1].imshow(imagen_prueba, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_axis_off()

ax[2].set_title('1-1')
ax[2].imshow(video_segm[600,:,:], cmap=plt.cm.gray, interpolation='nearest')
ax[2].set_axis_off()

ax[3].set_title('2-2')
ax[3].imshow(video_segm[700,:,:], cmap=plt.cm.gray, interpolation='nearest')
ax[3].set_axis_off()
plt.tight_layout()

plt.show()
"""
