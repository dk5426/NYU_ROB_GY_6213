# External libraries
import math
import numpy as np

# UDP parameters
localIP = "192.168.1.156" # Put your laptop computer's IP here 199
arduinoIP = "192.168.1.245" # Put your arduino's IP here 200
localPort = 4010
arduinoPort = 4010
bufferSize = 1024

# Camera parameters
camera_id = 0
marker_length = 0.072
# camera_matrix = np.array([[614.35278651, 0.00000000e+00 ,674.71265471],
#  [0.00000000e+00 ,613.9367336, 314.53705006],
#  [0.00000000e+00 ,0.00000000e+00 ,1]], dtype=np.float32)
# dist_coeffs = np.array([0.0629603,  -0.22345608, -0.03449465, -0.01117499,  0.07087979], dtype=np.float32)

# camera_matrix = np.array([
#     [867.76928279, 0.00000000, 652.06776042],
#     [0.00000000, 867.76928279, 361.24567204],
#     [0.00000000, 0.00000000, 1.00000000],
# ], dtype=np.float32)

# dist_coeffs = np.array([
#     [-0.33587349, -0.02162558, 0.06744279, -0.00883542, -0.10688801]
# ], dtype=np.float32)

camera_matrix = np.array([
    [854.66187513, 0.00000000, 539.20312086],
    [0.00000000, 854.66187513, 371.03936835],
    [0.00000000, 0.00000000, 1.00000000],
], dtype=np.float32)

dist_coeffs = np.array([
    [-0.38090393, 0.10494830, 0.05949717, 0.00432355, -0.21770556]
], dtype=np.float32)


# Robot parameters
num_robot_sensors = 2 # encoder, steering
num_robot_control_signals = 2 # speed, steering

# Lab 2 calibrated motion model constants (FancySlipBias / Ackermann hybrid)
K_SE  = 2.882760254750430982e-04   # m / count
K_SS  = 6.338605358524558586e-08   # m^2 / count  (variance of s)
A1    = -2.389748299319578351e+01
A2    =  8.136904761904759642e-01
A3    = -4.658503401360606679e-03
C_R   =  3.526364371043103914e-05  # (rad/s) / (count/s)
SIGMA_W2_CONST = 1.601538811297278713e-03  # (rad/s)^2

# Logging parameters
max_num_lines_before_write = 1
filename_start = './data/robot_data'
data_name_list = ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal', 'state_mean', 'state_covariance']

# Experiment trial parameters
trial_time = 10000 # milliseconds
extra_trial_log_time = 2000 # milliseconds

# KF parameters
I3 = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
covariance_plot_scale = 100
DT = 0.05 # Timing interval for both online loops and simulation (seconds)

# Camera-to-world linear mapping coefficients
# Derived from 7-pose ground-truth calibration (least-squares fit):
#   x_act [cm] = CAM_X_SCALE * cam_x [cm] + CAM_X_OFFSET [cm]
#   y_act [cm] = CAM_Y_SCALE * cam_y [cm] + CAM_Y_OFFSET [cm]
# In code: applied in meters (tvec units), offsets in meters.
# Calibration data:
#   cam_x  cam_y  cam_z   cam_th  x_act  y_act
#   -0.3   0.1   138.68  -0.2     0      0
#    69.6  -3.8  105.2   -7.5    80      0
#    -0.4 -85.3  162.73  -0.3     0    -90
#   -71.7  -0.1  112.32   8.2   -80      0
#     0.6 -41.3  155.34  -2.1     0    -45
#   -60.5 -51.4  140.55   4.9   -60    -64
#    60.8 -51.3  134.15  -1.9    63    -58
CAM_X_SCALE  = 1.082   # [dimensionless]
CAM_X_OFFSET = 0.00722 # [metres]  (= 0.722 cm)
CAM_Y_SCALE  = 1.113   # [dimensionless]
CAM_Y_OFFSET = 0.00330 # [metres]  (= 0.330 cm)

# Camera measurement noise variances (from mapping residuals, 7 poses)
# Units: meters^2 for x/y, rad^2 for theta
sigma_cam_x2     = 8.92e-4  # ~3.0 cm std  (was 0.01535)
sigma_cam_y2     = 1.28e-3  # ~3.6 cm std  (was 0.02195)
sigma_cam_theta2 = 7.03e-3  # ~5.2 deg std (unchanged)

# Online Filter / Tracking Settings
TARGET_MARKER_ID = 2
CONTRAST_VAL = 22
CLAHE_VAL = 1.6
BETA_VAL = -16
EXPOSURE_VAL = -7

