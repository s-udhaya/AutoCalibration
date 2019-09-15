from __future__ import division

import copy

from sklearn.base import BaseEstimator
from sklearn.metrics import mutual_info_score

from src.util import project_pointcloud_on_image


class AutoCalibration(BaseEstimator):
    def __init__(self, yaw=0, pitch=0.0, roll=0.0, translation_x=0.0, translation_y=0.0, translation_z=0.0,
                 use_kernel_density=False):
        self._histogram_bin_size = 255
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.translation_z = translation_z
        self.use_kernel_density = use_kernel_density
        # args, _, _, values = inspect.getargvalues(inspect.currentframe())
        # print(values)
        # values.pop("self")
        # for arg, val in values.items():
        #     setattr(self, arg, val)

    def _calculate_mutual_information(self, image, projected_point_cloud, intensities):
        image_intensities = []
        lidar_intensities = []
        projected_point_cloud = projected_point_cloud.astype(int).T
        for i in range(len(projected_point_cloud)):
            image_intensity = int(image[projected_point_cloud[i][1], projected_point_cloud[i][0]])
            lidar_intensity = int(intensities[i])
            image_intensities.append(image_intensity)
            lidar_intensities.append(lidar_intensity)
        return mutual_info_score(image_intensities, lidar_intensities)

    def _project_pointcloud_to_image(self, pointcloud, image):
        translation = [self.translation_x, self.translation_y, self.translation_z]
        rotation = [self.yaw, self.pitch, self.roll]
        projected_point_cloud, intensities = project_pointcloud_on_image(pointcloud[:, :3], pointcloud[:, 3],
                                                                         translation, rotation, image)
        return projected_point_cloud, intensities

    def score(self, image_batch, pointcloud_batch):
        pointcloud = copy.deepcopy(pointcloud_batch[0])
        image = copy.deepcopy(image_batch[0])
        projected_point_cloud, intensities = self._project_pointcloud_to_image(pointcloud, image)
        mi = self._calculate_mutual_information(image, projected_point_cloud, intensities)
        return mi

    def fit(self, X, y):
        return self

    def predict(self, X):
        pass
