import matplotlib as mpl
import cv2
import os
import scipy
import yaml
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.segmentation import chan_vese
import json
import xml.dom.minidom
from skimage import morphology
from tools import read_dicom
from tools import dt, tol, max_iter, init_level_set, extended_output
from tools import man, pic, pos, lambda1, lambda2, mu

'''
# work for batch.py
man, pic, pos, lambda1, lambda2, mu = '', '', '', 0, 0, 0
def chan_vese_batch(_man, _pic, _pos, _lambda1, _lambda2, _mu):
    global man, pic, pos, lambda1, lambda2, mu
    man, pic, pos, lambda1, lambda2, mu = _man, _pic, _pos, _lambda1, _lambda2, _mu
    start()
'''

def padding(img):
    THRESHOLD = 800
    thre = THRESHOLD
    padding_location = (img <= THRESHOLD)*255
    padding_value = img.mean()*0.95
    padding_img = (img > THRESHOLD)*img + (img <= THRESHOLD)*padding_value
    return padding_img, padding_location

def start():
    all_ctimg = read_dicom(os.path.join("lumbar_data", man, pic))
    dom = xml.dom.minidom.parse("lumbar_label/" + pic.replace(".IMA", ".xml"))
    root = dom.documentElement
    if(pos == 0):
        x1, x2, y1, y2 = root.getElementsByTagName("xmin")[0].firstChild.data, root.getElementsByTagName("xmax")[
            0].firstChild.data, root.getElementsByTagName("ymin")[0].firstChild.data, \
                         root.getElementsByTagName("ymax")[0].firstChild.data
    else:
        x1, x2, y1, y2 = root.getElementsByTagName("xmin")[1].firstChild.data, root.getElementsByTagName("xmax")[
            1].firstChild.data, root.getElementsByTagName("ymin")[1].firstChild.data, \
                         root.getElementsByTagName("ymax")[1].firstChild.data
    img = all_ctimg[int(y1):int(y2), int(x1):int(x2)]
    raw_ctimg = img
    padding_img, padding_location = padding(img)
    imgs = [img, padding_location, padding_img, padding_img]
    phi = chan_vese(padding_img, mu=mu, lambda1=lambda1, lambda2=lambda2, tol=tol, max_iter=max_iter, dt=dt,
                    init_level_set=init_level_set, extended_output=extended_output)
    imgs.append(phi)
    contours = measure.find_contours(phi, 0)

    plt.ion()
    fig2 = plt.figure(1, (15, 7))
    for i in range(5):
        location = (5 // 4 + 1) * 100 + 30 + i + 1
        ax1 = fig2.add_subplot(location)
        ax1.imshow(imgs[i], interpolation='nearest', cmap=plt.cm.gray)
        if i == 3:
            for n, contour in enumerate(contours):
                ax1.plot(contour[:, 1], contour[:, 0], linewidth=1)
    new_phi = morphology.remove_small_objects(phi)
    extend = 20
    b = np.zeros((phi.shape[0] + extend, phi.shape[1] + extend), bool)
    b[int(extend / 2):int(extend / 2) + phi.shape[0], int(extend / 2):int(extend / 2) + phi.shape[1]] = new_phi
    reverse_phi = (1 - b).astype(np.bool)
    new_phi = morphology.remove_small_objects(reverse_phi, min_size=150)
    new_phi = (1 - new_phi).astype(np.bool)

    new_phi = new_phi[int(extend / 2):int(extend / 2) + phi.shape[0], int(extend / 2):int(extend / 2) + phi.shape[1]]
    ax1 = fig2.add_subplot(236)
    ax1.imshow(new_phi, interpolation='nearest', cmap=plt.cm.gray)

    dire = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    point, visited = [], np.zeros(phi.shape)
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if new_phi[i][j] and visited[i][j] == 0:
                temp, q = [], [(i, j)]
                while len(q):
                    t = q.pop(0)
                    temp.append(raw_ctimg[t[0]][t[1]])
                    for k in range(4):
                        a, b = t[0] + dire[k][0], t[1] + dire[k][1]
                        if 0 <= a < phi.shape[0] and 0 <= b < phi.shape[1] and new_phi[a][b] == True and visited[a][
                            b] == 0:
                            q.append((a, b))
                            visited[a][b] = 1
                point.append(temp)

    if(pos == 0):
        brightness = json.dumps({'up': point[0]})
        fpath = os.path.join('brightness', man, 'UP_' + pic.replace(".IMA", ".json"))
    else:
        brightness = json.dumps({'down': point[0]})
        fpath = os.path.join('brightness', man, 'DOWN_' + pic.replace(".IMA", ".json"))
    with open(fpath, 'w') as f:
        f.write(brightness)
        f.close()

    sp = {'mu': mu, 'lambda1': lambda1, 'lambda2': lambda2, 'tol': tol, 'max_iter': max_iter, 'dt': dt,
          'init_level_set': init_level_set, 'extended_output': extended_output}
    if(pos == 0):
        save_path = os.path.join('params_split', man, 'UP_'+ pic.replace(".IMA", ".yaml"))
    else:
        save_path = os.path.join('params_split', man, 'DOWN_' + pic.replace(".IMA", ".yaml"))
    with open(save_path, 'w') as f:
        yaml.dump(sp,f,default_flow_style=False)

    num = len(point)
    print(num)

    if(pos == 0):
        save_path = os.path.join('figures_split', man, 'UP_'+ pic.replace(".IMA", ".png"))
    else:
        save_path = os.path.join('figures_split', man, 'DOWN_' + pic.replace(".IMA", ".png"))
    plt.savefig(save_path)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    start()