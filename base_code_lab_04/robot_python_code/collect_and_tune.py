import socket
import time
import numpy as np
import csv
import argparse
import math
import pickle
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
        # Re-use address to avoid bind errors on restart
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.local_ip, self.local_port))
        self.socket.settimeout(1.0)
        print(f"Connected to Robot at {self.arduino_ip}:{self.arduino_port}")

    def send_command(self, speed, steering):
        """Sends (speed, steering) command to the robot."""
        msg = f"{speed}, {steering}\n"
        self.socket.sendto(str.encode(msg), (self.arduino_ip, self.arduino_port))

    def receive_data(self):
        """Receives sensor data from the robot."""
        try:
            message, _ = self.socket.recvfrom(self.buffer_size)
            data = message.decode().split(',')
            # Unpack: [encoder, steering, num_lidar, ...]
            if len(data) >= 2:
                return int(data[0]), int(data[1])
        except (socket.timeout, ValueError, IndexError):
            return None, None
        return None, None

    def flush_buffer(self):
        """Clears stale packets from the UDP buffer."""
        self.socket.setblocking(False)
        try:
            while True:
                self.socket.recv(self.buffer_size)
        except BlockingIOError:
            pass
        finally:
            self.socket.setblocking(True)
            self.socket.settimeout(1.0)

    def get_fresh_data(self, wait_sec=2.0):
        """Flushes buffer and waits for a fresh packet."""
        self.flush_buffer()
        start = time.time()
        while time.time() - start < wait_sec:
            enc, steer = self.receive_data()
            if enc is not None:
                return enc, steer
        return None, None

    def stop(self):
        """Sends stop command."""
        self.send_command(0, 0)

# --- Data Collection Logic ---

def collect_straight(robot, samples=1):
    print("\n--- Straight Line Data Collection ---")
    print("The robot will drive forward. Place it at a marked start line.")
    input("Press Enter to start...")

    results = []
    for i in range(samples):
        print(f"\nTrial {i+1}/{samples}")
        
        # Get initial encoder (ensuring it's fresh)
        enc_start, _ = robot.get_fresh_data()
        if enc_start is None:
            print("Warning: Could not get initial encoder data!")
            # Fallback to older method if fresh fails
            enc_start, _ = robot.receive_data()
            
        print("Driving...")
        # Drive for 2 seconds at speed 100
        start_time = time.time()
        while time.time() - start_time < 3.0:
            robot.send_command(100, 0)
            time.sleep(0.05)
        
        robot.stop()
        print("Waiting 20 seconds for encoder to stabilize...")
        time.sleep(30.0)
        
        enc_end, _ = robot.get_fresh_data()
        if enc_end is None:
            enc_end, _ = robot.receive_data()

        delta_e = enc_end - enc_start
        e_fwd = delta_e # Forward is negative in some firmwares
        
        print(f"Stopped. Encoder delta (fwd): {e_fwd}")
        dist = float(input("Measure distance traveled (meters) and enter: "))
        results.append([e_fwd, dist])
    
    # Save to CSV
    file_path = Path("straight_data.csv")
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)
    print(f"Saved {len(results)} trials to {file_path}")

def collect_circle(robot, speed, steering):
    print(f"\n--- Circle Data Collection (Speed={speed}, Steer={steering}) ---")
    print("The robot will drive in a circle.")
    print("Instructions: Press ENTER exactly when the robot completes ONE FULL 360-turn.")
    input("Press Enter to start driving...")

    enc_start, _ = robot.get_fresh_data()
    if enc_start is None:
        enc_start, _ = robot.receive_data()
    
    start_time = time.time()
    
    # Start driving
    print("Robot is moving. WATCH IT CLOSELY!")
    try:
        # We can't block on input() if we need to keep sending UDP commands
        # So we run the robot in a loop until some trigger
        # However, a simple way is to rely on robot keeping the last command or 
        # using a thread. Let's do a simple loop.
        import sys
        import select

        def is_enter_pressed():
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                return True
            return False

        print("Press ENTER at 360 degrees...")
        while not is_enter_pressed():
            robot.send_command(speed, steering)
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        pass

    end_time = time.time()
    robot.stop()
    print("Waiting 20 seconds for encoder to stabilize...")
    time.sleep(20.0)
    
    enc_end, _ = robot.get_fresh_data()
    if enc_end is None:
        enc_end, _ = robot.receive_data()

    duration = end_time - start_time
    e_fwd = -(enc_end - enc_start)
    
    print(f"Recorded: Time={duration:.2f}s, Encoder Delta={e_fwd}, Steer={steering}")
    
    # Save to CSV
    file_path = Path("circle_data.csv")
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([duration, e_fwd, steering, speed])
    print(f"Saved trial to {file_path}")

# --- Tuning Logic ---

def tune_parameters():
    print("\n--- Tuning Motion Model Parameters (Fancy Slip-Bias) ---")
    
    # 1. Distance Mapping (s = K_SE * e_fwd)
    straight_file = Path("straight_data.csv")
    if straight_file.exists():
        data = np.genfromtxt(straight_file, delimiter=',')
        if data.ndim == 1: data = data.reshape(1, -1)
        
        e_fwd = data[:, 0]
        dist = data[:, 1]
        
        # Fit through origin: dist = K_SE * e_fwd
        # Sum(x*y) / Sum(x*x)
        k_se = np.sum(e_fwd * dist) / np.sum(e_fwd * e_fwd)
        
        # Variance: sigma_s^2 = K_SS * e_fwd
        residuals_sq = (dist - k_se * e_fwd)**2
        k_ss = np.sum(residuals_sq) / np.sum(np.abs(e_fwd))
        
        print(f"Distance Model:")
        print(f"  K_SE = {k_se:.8e} (meters/count)")
        print(f"  K_SS = {k_ss:.8e} (m^2/count)")
    else:
        print("No straight_data.csv found. Skip distances.")

    # 2. Yaw Rate Mapping (w = C_R * r_v * alpha)
    circle_file = Path("circle_data.csv")
    if circle_file.exists():
        data = np.genfromtxt(circle_file, delimiter=',')
        if data.ndim == 1: data = data.reshape(1, -1)
        
        # Columns: [duration, e_fwd, steering, speed]
        duration = data[:, 0]
        e_fwd = data[:, 1]
        steering = data[:, 2]
        
        # We know delta_theta = 2*pi (360 degrees) for each trial.
        # delta_theta = C_R * (e_fwd * steering)
        # 2*pi = C_R * X
        x = e_fwd * steering
        y = np.ones_like(x) * (2 * math.pi)
        
        c_r = np.sum(x * y) / np.sum(x * x)
        
        # Variance: sigma_w^2 (Constant)
        # Estimated w = (c_r * e_fwd * steering) / duration
        # Actual w = 2*pi / duration
        w_est = (c_r * e_fwd * steering) / duration
        w_act = (2 * math.pi) / duration
        sigma_w2 = np.var(w_act - w_est)
        
        print(f"Yaw Rate Model:")
        print(f"  C_R = {c_r:.8e} (rad/s per count-steering/s)")
        print(f"  SIGMA_W2_CONST = {sigma_w2:.8e} (rad/s)^2")
    else:
        print("No circle_data.csv found. Skip yaw rates.")

    print("\nUpdate your parameters.py with these values for the Fancy Slip-Bias model!")

# --- Main CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and tune robot motion model parameters.")
    parser.add_argument("--collect", choices=["straight", "circle"], help="Collect specific data type")
    parser.add_argument("--tune", action="store_true", help="Calculate parameters from existing data")
    parser.add_argument("--speed", type=int, default=80, help="Speed for circle collection")
    parser.add_argument("--steering", type=int, default=15, help="Steering for circle collection")
    parser.add_argument("--clear", action="store_true", help="Clear all collected CSV data and exit")
    
    args = parser.parse_args()
    
    if args.clear:
        for f in ["straight_data.csv", "circle_data.csv"]:
            p = Path(f)
            if p.exists():
                p.unlink()
                print(f"Cleared {f}")
        exit()

    if args.tune:
        tune_parameters()
    elif args.collect:
        # Load IPs from parameters.py if possible
        import parameters as params
        robot = RobotConnection(arduino_ip=params.arduinoIP, local_ip=params.localIP)
        
        try:
            if args.collect == "straight":
                collect_straight(robot, samples=15)
            elif args.collect == "circle":
                collect_circle(robot, args.speed, args.steering)
        finally:
            robot.stop()
    else:
        parser.print_help()
