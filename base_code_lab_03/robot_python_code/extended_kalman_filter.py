import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import parameters
import data_handling

def _wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

class ExtendedKalmanFilter:
    def __init__(self, x_0, Sigma_0, encoder_counts_0):
        self.state_mean = np.array(x_0, dtype=float)
        self.state_covariance = np.array(Sigma_0, dtype=float)
        self.predicted_state_mean = np.zeros(3)
        self.predicted_state_covariance = parameters.I3 * 1.0
        self.last_encoder_counts = float(encoder_counts_0)

    # Call the prediction and correction steps
    def update(self, u_t, z_t, delta_t):
        self.prediction_step(u_t, delta_t)
        self.correction_step(z_t)

    # Set the EKF's predicted state mean and covariance matrix
    def prediction_step(self, u_t, delta_t):
        encoder_counts = float(u_t[0])
        steering_angle_command = float(u_t[1])

        # Prediction via kinematic model
        x_bar, s = self.g_function(self.state_mean, u_t, delta_t)
        self.predicted_state_mean = x_bar

        # Jacobians and noise
        G_x = self.get_G_x(self.state_mean, s)
        G_u = self.get_G_u(self.state_mean, delta_t)
        R_t = self.get_R(s, u_t, delta_t)

        self.predicted_state_covariance = G_x @ self.state_covariance @ G_x.T + R_t
        
        # Update estimate
        self.state_mean = self.predicted_state_mean.copy()
        self.state_covariance = self.predicted_state_covariance.copy()
        
        # Store encoder counts for next step
        self.last_encoder_counts = encoder_counts

    # Set the EKF's corrected state mean and covariance matrix
    def correction_step(self, z_t):
        if z_t is None or np.any(np.isnan(z_t)) or (z_t[0] == 0.0 and z_t[1] == 0.0):
            return

        z = np.array(z_t, dtype=float)
        H = self.get_H()
        Q = self.get_Q()

        # Innovation
        innovation = z - self.get_h_function(self.predicted_state_mean)
        innovation[2] = _wrap_to_pi(innovation[2])

        # Kalman gain
        S = H @ self.predicted_state_covariance @ H.T + Q
        K = self.predicted_state_covariance @ H.T @ np.linalg.inv(S)

        # Update
        self.state_mean = self.predicted_state_mean + K @ innovation
        self.state_mean[2] = _wrap_to_pi(self.state_mean[2])
        self.state_covariance = (parameters.I3 - K @ H) @ self.predicted_state_covariance

    # Function to calculate distance from encoder counts
    def distance_travelled_s(self, encoder_counts):
        delta_e = float(encoder_counts) - self.last_encoder_counts
        e_fwd = -delta_e
        return parameters.K_SE * e_fwd

    # Function to calculate rotational velocity from steering
    def rotational_velocity_w(self, steering_angle_command):
        v = float(np.clip(parameters.DEFAULT_SPEED_CMD, 40.0, 100.0))
        r = max(0.0, parameters.A1 * v + parameters.A2 * v**2 + parameters.A3 * v**3)
        return parameters.C_R * r * float(steering_angle_command)

    # The nonlinear transition equation
    def g_function(self, x_tm1, u_t, delta_t):
        s = self.distance_travelled_s(u_t[0])
        w = self.rotational_velocity_w(u_t[1])
        dth = w * delta_t

        th_mid = x_tm1[2] + 0.5 * dth
        x_new = x_tm1[0] + s * math.cos(th_mid)
        y_new = x_tm1[1] + s * math.sin(th_mid)
        th_new = _wrap_to_pi(x_tm1[2] + dth)
        return np.array([x_new, y_new, th_new]), s

    # The nonlinear measurement function
    def get_h_function(self, x_t):
        return x_t

    # Jacobian dg/dx
    def get_G_x(self, x_tm1, s):
        th = x_tm1[2]
        return np.array([
            [1.0, 0.0, -s * math.sin(th)],
            [0.0, 1.0,  s * math.cos(th)],
            [0.0, 0.0,  1.0]
        ])

    # Jacobian dg/du
    def get_G_u(self, x_tm1, delta_t):
        th = x_tm1[2]
        ds_dec = -parameters.K_SE
        
        v = float(np.clip(parameters.DEFAULT_SPEED_CMD, 40.0, 100.0))
        r = max(0.0, parameters.A1 * v + parameters.A2 * v**2 + parameters.A3 * v**3)
        dw_dsc = parameters.C_R * r

        dg_dec = np.array([math.cos(th) * ds_dec, math.sin(th) * ds_dec, 0.0])
        dg_dsc = np.array([0.0, 0.0, dw_dsc * delta_t])
        return np.column_stack([dg_dec, dg_dsc])

    # Jacobian dh/dx
    def get_H(self):
        return parameters.I3.astype(float)

    # Transition process noise R_t
    def get_R(self, s, u_t, delta_t):
        delta_e = float(u_t[0]) - self.last_encoder_counts
        sigma_s2 = max(0.0, parameters.K_SS * abs(delta_e))
        sigma_theta2 = parameters.SIGMA_W2_CONST * delta_t**2

        Sigma_u = np.diag([sigma_s2, sigma_theta2])
        G_u = self.get_G_u(self.state_mean, delta_t)
        
        R_t = G_u @ Sigma_u @ G_u.T
        R_t = 0.5 * (R_t + R_t.T) + np.diag([1e-5, 1e-5, 1e-5])
        return R_t

    # Measurement noise Q_t
    def get_Q(self):
        return np.diag([
            parameters.sigma_cam_x2,
            parameters.sigma_cam_y2,
            parameters.sigma_cam_theta2
        ])

class KalmanFilterPlot:
    def __init__(self):
        self.dir_length = 0.1
        self.fig, self.ax = plt.subplots()

    def update(self, state_mean, state_covariance):
        plt.clf()
        ax = self.fig.gca()
        scale = parameters.covariance_plot_scale
        lambda_, v = np.linalg.eig(state_covariance * scale)
        lambda_ = np.sqrt(np.abs(lambda_))
        
        angle = np.rad2deg(np.arctan2(*v[:, 0][::-1]))
        ell = Ellipse((state_mean[0], state_mean[1]), alpha=0.5, facecolor='red',
                      width=lambda_[0], height=lambda_[1], angle=angle)
        ax.add_artist(ell)
        
        plt.plot(state_mean[0], state_mean[1], 'ro')
        plt.plot(
            [state_mean[0], state_mean[0] + self.dir_length * math.cos(state_mean[2])],
            [state_mean[1], state_mean[1] + self.dir_length * math.sin(state_mean[2])],
            'r'
        )
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis([-0.25, 2, -1, 1])
        plt.grid()
        plt.draw()
        plt.pause(0.1)

# Code to run your EKF offline with a data file.
def offline_efk():
    filename = './data/robot_data_68_0_06_02_26_17_12_19.pkl'
    ekf_data = data_handling.get_file_data_for_kf(filename)

    x_0 = [ekf_data[0][3][0] + 0.5, ekf_data[0][3][1], ekf_data[0][3][5]]
    Sigma_0 = parameters.I3 * 10.0
    encoder_counts_0 = ekf_data[0][2].encoder_counts
    ekf = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)
    kf_plot = KalmanFilterPlot()

    for t in range(1, len(ekf_data)):
        row = ekf_data[t]
        delta_t = ekf_data[t][0] - ekf_data[t - 1][0]
        u_t = np.array([row[2].encoder_counts, row[2].steering])
        z_t = np.array([row[3][0], row[3][1], row[3][5]])

        ekf.update(u_t, z_t, delta_t)
        kf_plot.update(ekf.state_mean, ekf.state_covariance[0:2, 0:2])

####### MAIN #######
if __name__ == "__main__":
    if False:
        offline_efk()
