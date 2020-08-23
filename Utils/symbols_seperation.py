import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

def cv_blobber():
    global im_with_keypoints
    # use this opencv algorithm (https://www.learnopencv.com/blob-detection-using-opencv-python-c/):
    # SET BLOCBBER PARAMETERS:
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 160;
    # Filter by Area.
    params.filterByArea = False
    params.minArea = 15
    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    Blobber = cv2.SimpleBlobDetector_create(params)
    keypoints = Blobber.detect(img)
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)
    # plt.imshow(im_with_keypoints)


# cv_blobber()

def find_cc(img, threshold=140):

    pMask = img < threshold
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(pMask.astype('int8'))
    cc = pd.DataFrame(
        data={'cx': centroids[1:, 0], 'cy': centroids[1:, 1], 'bbx': stats[1:, 0], 'bby': stats[1:, 1],
              'bbw': stats[1:, 2], 'bbh': stats[1:, 3], 'area': stats[1:, 4]})
    filter = cc.area > 10
    filter *= cc.bbw > 5
    filter *= cc.bbh > 5
    cc = cc[filter]
    cc = cc.sort_values('cx').reset_index(drop=True)
    return cc

def unite_symbols_with_gap(cc):
    # find cc that are part of the same symbol:
    # 1. check which two adjacent bb are very close:
    # xdists = cc.cx.values.reshape(-1, 1) - cc.cx.values.reshape(1, -1)
    # cand = np.abs(xdists)<cc.bbw.quantile(.25)
    candb = (cc.cx.diff()) < min(10, cc.bbw.quantile(.1))
    canda = candb.shift(-1).fillna(False)
    u = cc[canda + candb]
    mapper = {'cx': np.mean, 'cy': np.mean, 'bbx': np.min, 'bby': np.min, 'bbw': np.mean, 'bbh': np.mean,
              'area': np.sum}
    united = u.apply(mapper)
    united.bbw = (u.bbx + u.bbw).max() - united.bbx
    united.bbh = (u.bby + u.bbh).max() - united.bby
    cc = cc[~ (canda + candb)]
    cc = cc.append(united, ignore_index=True)
    return cc

def plot_detections(img, cc):
    plt.imshow(img)
    plt.plot([cc.bbx, cc.bbx + cc.bbw, cc.bbx + cc.bbw, cc.bbx, cc.bbx],
             [cc.bby, cc.bby, cc.bby + cc.bbh, cc.bby + cc.bbh, cc.bby])
    # for i in range(len(cc)):
    #     plt.text(cc.cx[i], cc.cy[i], str(cc.index[i]))
    plt.show()


if __name__=='__main__':
    data_path = '/home/yuval/Projects/EquSolve/DataSets/hand_written_eqs/'
    fn = data_path + 'example3.jpg'
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

    cc = find_cc(img)
    cc = unite_symbols_with_gap(cc)
    plot_detections(img, cc)

    pass

