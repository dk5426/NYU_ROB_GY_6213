# Extended Kalman Filter for Lab 03
# Motion model: AckermannBicycle (mid-heading) using Lab 2 calibrated constants
# Measurement model: direct camera pose z_t = [x, y, theta] (H = I3)
#
# Lab 2 calibrated constants (FancySlipBias / Ackermann hybrid):
#   K_SE  = 2.882760e-04  m/count  (distance per encoder count)
#   K_SS  = 6.338605e-08  m^2/count (distance variance per count)
#   C_R   = 3.526364e-05  (rad/s) per (count/s)
#   SIGMA_W2_CONST = 1.601539e-03  (rad/s)^2
#   A1, A2, A3: speed cmd -> counts/sec polynomial coefficients
#
# Camera noise (from Step 1.5 characterization, 10 poses):
#   sigma_x^2  ~ 0.01535 m^2
#   sigma_y^2  ~ 0.02195 m^2
#   sigma_th^2 ~ 7.03e-4 rad^2

# External libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Local libraries
import parameters
import data_handling

# --------------------------------------------------------------------------
# Lab 2 motion model constants (copy from simulation.ipynb FancySlipBias)
# --------------------------------------------------------------------------
K_SE  = 2.882760254750430982e-04   # m / count
K_SS  = 6.338605358524558586e-08   # m^2 / count  (variance of s)
A1    = -2.389748299319578351e+01
A2    =  8.136904761904759642e-01
A3    = -4.658503401360606679e-03
C_R   =  3.526364371043103914e-05  # (rad/s) / (count/s)
SIGMA_W2_CONST = 1.601538811297278713e-03  # (rad/s)^2

DTH_EPS = 1e-8


def _wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------
# Main class
# --------------------------------------------------------------------------
class ExtendedKalmanFilter:

    def __init__(self, x_0, Sigma_0, encoder_counts_0):
        self.state_mean = np.array(x_0, dtype=float)
        self.state_covariance = np.array(Sigma_0, dtype=float)
        self.predicted_state_mean = np.zeros(3)
        self.predicted_state_covariance = parameters.I3 * 1.0
        self.last_encoder_counts = float(encoder_counts_0)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def update(self, u_t, z_t, delta_t):
        """Call prediction then conditional correction."""
        self.prediction_step(u_t, delta_t)
        self.correction_step(z_t)

    # ------------------------------------------------------------------
    # PREDICTION STEP
    # ------------------------------------------------------------------

    def prediction_step(self, u_t, delta_t):
        """
        u_t = [encoder_counts, steering_cmd]
        delta_t = time step (seconds)
        """
        encoder_counts = float(u_t[0])
        steering_cmd   = float(u_t[1])

        # Predicted mean via kinematic model
        x_bar, s = self.g_function(self.state_mean, u_t, delta_t)
        self.predicted_state_mean = x_bar

        # Jacobians
        G_x = self.get_G_x(self.state_mean, s)
        G_u = self.get_G_u(self.state_mean, u_t, delta_t, s)

        # Process noise
        R_t = self.get_R(s, u_t, delta_t)

        # Predicted covariance
        self.predicted_state_covariance = (
            G_x @ self.state_covariance @ G_x.T + R_t
        )

        # Commit prediction as current estimate (correction may refine)
        self.state_mean = self.predicted_state_mean.copy()
        self.state_covariance = self.predicted_state_covariance.copy()

    # ------------------------------------------------------------------
    # CORRECTION STEP
    # ------------------------------------------------------------------

    def correction_step(self, z_t):
        """
        z_t = [x_cam, y_cam, theta_cam]
        Skip if z_t is None, all-zero, or contains NaN.
        """
        if z_t is None:
            return
        z = np.array(z_t, dtype=float)
        # Skip if measurement is invalid (all zeros or NaN)
        if np.any(np.isnan(z)) or (z[0] == 0.0 and z[1] == 0.0):
            return

        H = self.get_H()
        Q = self.get_Q()

        # Innovation
        innovation = z - H @ self.predicted_state_mean
        innovation[2] = _wrap_to_pi(innovation[2])

        # Kalman gain  K = Sigma_bar * H^T * (H * Sigma_bar * H^T + Q)^-1
        # Since H = I3, simplifies to:  K = Sigma_bar * (Sigma_bar + Q)^-1
        S = self.predicted_state_covariance + Q
        K = self.predicted_state_covariance @ np.linalg.inv(S)

        # Updated state
        self.state_mean = self.predicted_state_mean + K @ innovation
        self.state_mean[2] = _wrap_to_pi(self.state_mean[2])

        # Updated covariance
        self.state_covariance = (parameters.I3 - K) @ self.predicted_state_covariance

    # ------------------------------------------------------------------
    # HELPERS: distance & angular velocity from controls
    # ------------------------------------------------------------------

    def distance_travelled_s(self, encoder_counts):
        """Convert cumulative encoder count to distance (m)."""
        delta_e = float(encoder_counts) - self.last_encoder_counts
        self.last_encoder_counts = float(encoder_counts)
        e_fwd = -delta_e   # forward motion: encoder counts go negative
        return K_SE * e_fwd, abs(e_fwd)   # (s in meters, |e_fwd| for variance)

    def rotational_velocity_w(self, v_cmd, steering_cmd):
        """
        Linear yaw model: w = C_R * r(v) * alpha_cmd
        v_cmd in speed command units [40..100]
        steering_cmd in servo units [-20..20]
        Returns w in rad/s.
        """
        v = float(np.clip(v_cmd, 40.0, 100.0))
        r = max(0.0, A1 * v + A2 * v**2 + A3 * v**3)  # counts/sec
        return C_R * r * float(steering_cmd)

    # ------------------------------------------------------------------
    # TRANSITION FUNCTION
    # ------------------------------------------------------------------

    def g_function(self, x_tm1, u_t, delta_t):
        """
        Ackermann mid-heading kinematic update (deterministic mean).
        Returns (x_t, s) where s is distance travelled in meters.
        """
        encoder_counts = float(u_t[0])
        v_cmd          = parameters.default_speed_cmd   # need speed; use stored
        steering_cmd   = float(u_t[1])

        delta_e = encoder_counts - self.last_encoder_counts
        e_fwd   = -delta_e
        s       = K_SE * e_fwd
        w       = self.rotational_velocity_w(v_cmd, steering_cmd)
        dth     = w * delta_t

        x, y, th = float(x_tm1[0]), float(x_tm1[1]), float(x_tm1[2])
        th_mid = th + 0.5 * dth
        x_new  = x  + s * math.cos(th_mid)
        y_new  = y  + s * math.sin(th_mid)
        th_new = _wrap_to_pi(th + dth)

        return np.array([x_new, y_new, th_new]), s

    # ------------------------------------------------------------------
    # JACOBIANS
    # ------------------------------------------------------------------

    def get_G_x(self, x_tm1, s):
        """
        Jacobian of g wrt state x = [x, y, theta].
        dg/dx = [[1, 0, -s*sin(th + dth/2)],
                 [0, 1,  s*cos(th + dth/2)],
                 [0, 0,  1              ]]
        We approximate dth/2 ~ 0 for the Jacobian (small angle step).
        """
        th = float(x_tm1[2])
        return np.array([
            [1.0, 0.0, -s * math.sin(th)],
            [0.0, 1.0,  s * math.cos(th)],
            [0.0, 0.0,  1.0             ]
        ])

    def get_G_u(self, x_tm1, u_t, delta_t, s):
        """
        Jacobian of g wrt control u = [encoder_counts, steering_cmd].
        Column 0: dg/d(encoder_counts)  -> via ds/d(ec) = K_SE
        Column 1: dg/d(steering_cmd)    -> via dw/d(sc) = C_R * r(v)
        """
        th = float(x_tm1[2])
        v_cmd       = parameters.default_speed_cmd
        steering_cmd = float(u_t[1])

        # ds/d(encoder_counts) = -K_SE  (negative because e_fwd = -delta_e)
        ds_dec = -K_SE

        # dw/d(steering_cmd) = C_R * r(v)
        v = float(np.clip(v_cmd, 40.0, 100.0))
        r = max(0.0, A1 * v + A2 * v**2 + A3 * v**3)
        dw_dsc = C_R * r

        # dg/d(encoder_counts): same direction as dg/ds
        dg_dec = np.array([
            math.cos(th) * ds_dec,
            math.sin(th) * ds_dec,
            0.0
        ])

        # dg/d(steering_cmd): rotation changes
        dg_dsc = np.array([
            0.0,
            0.0,
            dw_dsc * delta_t
        ])

        return np.column_stack([dg_dec, dg_dsc])  # shape (3, 2)

    # ------------------------------------------------------------------
    # NOISE COVARIANCE MATRICES
    # ------------------------------------------------------------------

    def get_R(self, s, u_t=None, delta_t=1.0):
        """
        Process noise covariance R_t = G_u * Sigma_u * G_u^T
        Sigma_u = diag(sigma_s^2, sigma_w^2)
        sigma_s^2 = K_SS * |e_fwd|   (proportional to distance)
        sigma_w^2 = SIGMA_W2_CONST * delta_t^2  (variance of w integrated over dt)
        """
        if u_t is None:
            return parameters.I3 * 0.01

        x_tm1 = self.state_mean
        v_cmd = parameters.default_speed_cmd

        # Variance of s
        encoder_counts = float(u_t[0])
        delta_e = encoder_counts - self.last_encoder_counts
        e_fwd = abs(delta_e)
        sigma_s2 = max(0.0, K_SS * e_fwd)

        # Variance of w integrated: sigma_w^2 * dt^2
        sigma_theta2 = SIGMA_W2_CONST * delta_t**2

        Sigma_u = np.diag([sigma_s2, sigma_theta2])
        G_u = self.get_G_u(x_tm1, u_t, delta_t, s)

        R_t = G_u @ Sigma_u @ G_u.T
        # Ensure positive semi-definite (numerical safety)
        R_t = 0.5 * (R_t + R_t.T)
        return R_t

    def get_H(self):
        """H = I3 since camera measures [x, y, theta] directly."""
        return parameters.I3.astype(float)

    def get_Q(self):
        """
        Measurement noise Q_t from camera characterization (10 poses).
        Units: meters and radians.
        """
        return np.diag([
            parameters.sigma_cam_x2,
            parameters.sigma_cam_y2,
            parameters.sigma_cam_theta2
        ])


# --------------------------------------------------------------------------
# Plotting helper
# --------------------------------------------------------------------------
class KalmanFilterPlot:

    def __init__(self):
        self.dir_length = 0.1
        fig, ax = plt.subplots()
        self.ax = ax
        self.fig = fig

    def update(self, state_mean, state_covariance):
        plt.clf()
        ax = self.fig.gca()

        # Plot covariance ellipse (scaled for visibility)
        scale = parameters.covariance_plot_scale
        cov2 = state_covariance * scale
        lambda_, v = np.linalg.eig(cov2)
        lambda_ = np.sqrt(np.abs(lambda_))
        xy = (state_mean[0], state_mean[1])
        angle = np.rad2deg(np.arctan2(*v[:, 0][::-1]))
        ell = Ellipse(xy, alpha=0.5, facecolor='red',
                      width=lambda_[0], height=lambda_[1], angle=angle)
        ax.add_artist(ell)

        # Plot state estimate dot + heading arrow
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


# --------------------------------------------------------------------------
# Offline EKF runner
# --------------------------------------------------------------------------
def offline_efk():
    """Load a data file and run the EKF offline over it."""

    filename = './data/robot_data_68_0_06_02_26_17_12_19.pkl'
    ekf_data = data_handling.get_file_data_for_kf(filename)

    # Initial state (slightly offset from first camera reading to test robustness)
    x_0 = [ekf_data[0][3][0] + 0.5, ekf_data[0][3][1], ekf_data[0][3][5]]
    Sigma_0 = parameters.I3 * 10.0   # large initial uncertainty
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
if False:
    offline_efk()
