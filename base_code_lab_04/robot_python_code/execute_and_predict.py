import socket
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
from pathlib import Path

# --- Robot Communication Logic ---

class RobotConnection:
    def __init__(self, arduino_ip="192.168.0.150", local_ip="192.168.0.148", port=4010, buffer_size=1024):
        self.arduino_ip = arduino_ip
        self.arduino_port = port
        self.local_ip = local_ip
        self.local_port = port
        self.buffer_size = buffer_size
        self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.local_ip, self.local_port))
        self.socket.settimeout(1.0)
        print(f"Connected to Robot at {self.arduino_ip}:{self.arduino_port}")

    def send_command(self, speed, steering_offset):
        msg = f"{speed}, {steering_offset}\n"
        self.socket.sendto(str.encode(msg), (self.arduino_ip, self.arduino_port))

    def receive_data(self):
        try:
            message, _ = self.socket.recvfrom(self.buffer_size)
            data = message.decode().split(',')
            if len(data) >= 2:
                return int(data[0]), int(data[1])
        except (socket.timeout, ValueError, IndexError):
            return None, None
        return None, None

    def flush_buffer(self):
        self.socket.setblocking(False)
        try:
            while True:
                self.socket.recv(self.buffer_size)
        except BlockingIOError:
            pass
        finally:
            self.socket.setblocking(True)
            self.socket.settimeout(1.0)

    def get_fresh_data(self, wait_sec=1.0):
        self.flush_buffer()
        start = time.time()
        while time.time() - start < wait_sec:
            enc, steer = self.receive_data()
            if enc is not None:
                return enc, steer
        return None, None

    def stop(self):
        self.send_command(0, 0)

# --- Path Prediction Logic (Fancy Slip-Bias) ---

def get_ellipse_pts(mean, cov, n_std=3.0, n_pts=50):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.linspace(0, 2*np.pi, n_pts)
    scale = n_std * np.sqrt(np.maximum(vals, 1e-12))
    ellipse_x = scale[0] * np.cos(theta)
    ellipse_y = scale[1] * np.sin(theta)
    pts = np.column_stack([ellipse_x, ellipse_y]) @ vecs.T
    return pts + mean

def predict_path(history, k_se, k_ss, c_r, sigma_w2, sigma_beta, sigma_bias):
    """
    Propagates 4D state [x, y, theta, bias_w] using Fancy Slip-Bias logic.
    """
    x, y, theta, bias_w = 0.0, 0.0, 0.0, 0.0
    P = np.zeros((4, 4))
    
    path_x, path_y = [x], [y]
    path_P = [P.copy()]
    
    for i in range(1, len(history)):
        t_prev, enc_prev, steer_prev = history[i-1]
        t_curr, enc_curr, steer_curr = history[i]
        
        dt = max(t_curr - t_prev, 1e-9)
        delta_e = enc_curr - enc_prev
        
        ds = k_se * delta_e
        # Mean yaw rate from command
        w_cmd = (c_r * delta_e * steer_curr) / dt
        dth = (w_cmd + bias_w) * dt
        
        # Jacobian F = dX/dX
        F = np.eye(4)
        F[0, 2] = -ds * np.sin(theta)
        F[1, 2] =  ds * np.cos(theta)
        F[2, 3] = dt
        
        # Process Noise Q in control/hyperparam space
        # u = [ds, w_cmd, beta, noise_bias_w]
        var_s = k_ss * abs(delta_e)
        var_w = sigma_w2 * dt
        var_beta = sigma_beta**2
        var_bias = (sigma_bias**2) * dt
        
        Q = np.diag([var_s, var_w, var_beta, var_bias])
        
        # Jacobian G = dX/du
        G = np.zeros((4, 4))
        G[0, 0] = np.cos(theta)
        G[1, 0] = np.sin(theta)
        G[2, 1] = dt
        G[0, 2] = -ds * np.sin(theta)
        G[1, 2] =  ds * np.cos(theta)
        G[3, 3] = 1.0
        
        # Covariance update
        P = F @ P @ F.T + G @ Q @ G.T
        
        # Mean update
        x += ds * np.cos(theta)
        y += ds * np.sin(theta)
        theta += dth
        # bias_w constant in expectation
        
        path_x.append(x)
        path_y.append(y)
        path_P.append(P.copy())
        
    return path_x, path_y, path_P

# --- Execution Logic ---

def execute_trajectory(robot, sequence, k_se, k_ss, c_r, sigma_w2, sigma_beta, sigma_bias):
    print("\n--- Executing Trajectory Sequence ---")
    input("Place robot at (0,0) with Heading 0 and press Enter to start...")
    
    history = []
    enc_start, _ = robot.get_fresh_data()
    if enc_start is None:
        print("Error: No robot comms.")
        return
    
    start_time = time.perf_counter()
    history.append((0.0, enc_start, 0))
    
    try:
        for speed, steer_offset, duration in sequence:
            print(f"Running: Spd={speed}, Str={steer_offset}, Dur={duration}s")
            seg_start = time.perf_counter()
            while time.perf_counter() - seg_start < duration:
                robot.send_command(speed, steer_offset)
                enc, _ = robot.receive_data()
                if enc is not None:
                    history.append((time.perf_counter() - start_time, enc, steer_offset))
                time.sleep(0.05)
        robot.stop()
        time.sleep(1.0)
        enc_end, _ = robot.get_fresh_data()
        if enc_end is not None:
            history.append((time.perf_counter() - start_time, enc_end, 0))
    except KeyboardInterrupt:
        robot.stop()

    history.sort(key=lambda x: x[0])
    px, py, pP = predict_path(history, k_se, k_ss, c_r, sigma_w2, sigma_beta, sigma_bias)
    
    plt.figure(figsize=(10, 8))
    plt.plot(px, py, 'b-', label='Fancy Slip-Bias Prediction')
    
    interval = max(1, len(px) // 6)
    for i in range(0, len(px), interval):
        pts = get_ellipse_pts(np.array([px[i], py[i]]), pP[i][:2, :2], n_std=2.0)
        plt.plot(pts[:, 0], pts[:, 1], 'r--', alpha=0.3)
    
    pts_final = get_ellipse_pts(np.array([px[-1], py[-1]]), pP[-1][:2, :2], n_std=3.0)
    plt.plot(pts_final[:, 0], pts_final[:, 1], 'r-', label='Final 3-sigma region')

    plt.plot(px[0], py[0], 'go', label='Start')
    plt.plot(px[-1], py[-1], 'ro', label='End')
    plt.axis('equal'); plt.grid(True); plt.legend()
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.title('Fancy Slip-Bias Trajectory Verification')
    
    plot_file = "fancy_trajectory_plot.png"
    plt.savefig(plot_file)
    print(f"\nPlot saved to {plot_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_se", type=float)
    parser.add_argument("--k_ss", type=float)
    parser.add_argument("--c_r", type=float)
    parser.add_argument("--sigma_w2", type=float)
    parser.add_argument("--sigma_beta", type=float)
    parser.add_argument("--sigma_bias", type=float)
    args = parser.parse_args()
    
    import parameters as params
    def get_p(key, default): return getattr(params, key) if hasattr(params, key) else default
    
    k_se = args.k_se or get_p('K_SE', -5.57e-4)
    k_ss = args.k_ss or get_p('K_SS', 1.08e-6)
    c_r = args.c_r or get_p('C_R', -1.92e-5)
    sigma_w2 = args.sigma_w2 or get_p('SIGMA_W2_CONST', 0.058)
    sigma_beta = args.sigma_beta or get_p('SIGMA_BETA', 0.017)
    sigma_bias = args.sigma_bias or get_p('SIGMA_BIAS', 0.008)

    robot = RobotConnection(arduino_ip=params.arduinoIP, local_ip=params.localIP)
    
    SQUARE_TRAJ = [(100, 0, 1.5), (0, 15, 1.0), (100, 0, 1.5), (0, 15, 1.0),
                   (100, 0, 1.5), (0, 15, 1.0), (100, 0, 1.5), (0, 15, 1.0)]
    
    execute_trajectory(robot, SQUARE_TRAJ, k_se, k_ss, c_r, sigma_w2, sigma_beta, sigma_bias)
