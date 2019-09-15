import time

import numpy as np
from sklearn.model_selection import GridSearchCV

from src.auto_calibration import AutoCalibration
from src.util import get_data, get_gridsearch_params

true_params = {"pitch": 0.24061729, "roll": -89.61270704, "translation_x": 1.5039405282244198,
               "translation_y": -0.02676183592864872, "translation_z": 1.6584901808053665, "yaw": -88.54654913}

initial_params = {"pitch": -1, "roll": -90, "translation_x": 1.5039405282244198, "translation_y": -0.02676183592864872,
                  "translation_z": 1.6584901808053665, "yaw": -87}

image, pointcloud = get_data()

gridsearch_params = get_gridsearch_params(initial_params, mutate_angle=True, mutate_translation=False)
print(gridsearch_params)

cv = [(slice(None), slice(None))]
auto_calibration_grid_search = GridSearchCV(AutoCalibration(), cv=cv, param_grid=gridsearch_params, n_jobs=8)

start_time = time.time()
image_batch = np.expand_dims(image, axis=0)
pointcloud_batch = np.expand_dims(pointcloud, axis=0)
print(image_batch.shape)
# debug(image[0], pointcloud[0], true_params)
auto_calibration_grid_search.fit(image_batch, pointcloud_batch)

print("Grid scores on development set:")
print()
means = auto_calibration_grid_search.cv_results_['mean_test_score']
stds = auto_calibration_grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, auto_calibration_grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("true paramenters")
print(true_params)
print("best parameters and score")
print(auto_calibration_grid_search.best_params_)
print(auto_calibration_grid_search.best_score_)
print()
print("process took: %.2f seconds" % (time.time() - start_time))
