import os, sys, cv2
from PIL import Image

from kitti_object import *

def visualization(image_path, label_path, calib_path, lidar_path):
    dataset = kitti_object(image_path, label_path, calib_path, lidar_path)

    objects = dataset.get_label_objects()
    print("There are %d objects.", len(objects))

    image = dataset.get_image()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape

    pc_velo = dataset.get_lidar()[:, 0:3]  # (x, y, z)
    calib = dataset.get_calibration()

    show_image_with_boxes(image, objects, calib, True)
    bev_image = get_bev_image(pc_velo, objects, calib)
    bev_image = np.asarray(bev_image)
    bev_image = cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB)
    show_lidar_topview_with_boxes(bev_image, objects, calib)

if __name__ == '__main__':
    image_path = r'/home/pxr/pxrProject/3Ddetection/MonoLSS/data/KITTI3D/kitti/testing/image_2/000228.png'
    calib_path = r'/home/pxr/pxrProject/3Ddetection/MonoLSS/data/KITTI3D/kitti/testing/calib/000228.txt'
    lidar_path = r'/home/pxr/pxrProject/3Ddetection/MonoLSS/data/KITTI3D/kitti/testing/velodyne/000228.bin'
    label_path = r'/home/pxr/pxrProject/3Ddetection/MonoLSS/result/data/000228.txt'

    visualization(image_path, label_path, calib_path, lidar_path)

