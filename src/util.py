import copy

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from pyquaternion import Quaternion

from src.transformations import quaternion_from_euler

camera_intrinsic = np.array(
    [[1109.05239567, 0, 957.849065461], [0.0, 1109.05239567, 539.672710373], [0.0, 0.0, 1.0]])


def translate(points, t):
    """
    Applies a translation to the point cloud.
    :param x: <np.float: 3, 1>. Translation in x, y, z.
    """
    for i in range(3):
        points[i, :] = points[i, :] + t[i]
    return points


def rotate(points, rot_matrix):
    """
    Applies a rotation.
    :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
    """
    points[:3, :] = np.dot(rot_matrix, points[:3, :])
    return points


def view_points(points, view, normalize):
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def project_pointcloud_on_image(pointcloud, intensities, translation, rotation, image):
    """
    :param translation: (x, y, z) in m
    :param rotation: (yaw, pitch, roll) in degrees
    """
    translation = -np.array(translation)
    rotation = Quaternion(quaternion_from_euler(*np.radians(np.array(
        rotation)))).rotation_matrix.T
    # pc = copy.deepcopy(pointcloud)
    pointcloud = pointcloud.T
    pointcloud = translate(pointcloud, translation)
    pointcloud = rotate(pointcloud, rotation)
    depths = pointcloud[2, :]
    points = view_points(pointcloud, camera_intrinsic, normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < image.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < image.shape[0] - 1)
    points = points[:, mask]
    intensities = intensities[mask]
    return points, intensities


def get_data():
    image = np.load("../data/image.npy")
    pointcloud = np.load("../data/point_cloud_intensities.npy")
    image = rgb2gray(image)
    return image, pointcloud


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def debug(image, pointcloud, params):
    image = copy.deepcopy(image)
    pointcloud = copy.deepcopy(pointcloud)

    translation = [params['translation_x'], params['translation_y'], params['translation_z']]
    # (roll, pitch, yaw)
    rotation = ([params['yaw'], params['pitch'], params['roll']])

    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()
    points, intensities = project_pointcloud_on_image(pointcloud[:, :3], pointcloud[:, 3], translation, rotation, image)
    plt.figure(figsize=(9, 16))
    plt.imshow(image, plt.get_cmap('gray'))
    viridis = cm.get_cmap('viridis', 12)
    plt.scatter(points[0, :], points[1, :], c=intensities, cmap=viridis, s=2)
    plt.show()


def get_gridsearch_params(params, mutate_angle=True, mutate_translation=False):
    gridsearch_params = {}
    angle = ['yaw', 'pitch', 'roll']
    for key, value in params.items():
        if key in angle:
            if mutate_angle:
                values = np.linspace(value - 3, value + 3, 10)
            else:
                values = [value]
        else:
            if mutate_translation:
                values = np.linspace(value - 2, value + 2, 10)
            else:
                values = [value]
        gridsearch_params[key] = values
    return gridsearch_params
