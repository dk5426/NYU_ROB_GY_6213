import socket
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

# --- Robot Communication Logic ---
class LidarCollector:
    def __init__(self, arduino_ip="192.168.0.150", local_ip="192.168.0.148", port=4010, buffer_size=1024):
        self.arduino_ip = arduino_ip
        self.local_ip = local_ip
        self.port = port
        self.buffer_size = buffer_size
        self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.local_ip, self.port))
        self.socket.settimeout(1.0)
        print(f"Listening for Lidar data at {self.local_ip}:{self.port}")

    def receive_lidar_data(self):
        try:
            message, _ = self.socket.recvfrom(self.buffer_size)
            data = message.decode().split(',')
            if len(data) < 3:
                return None
            
            num_rays = int(data[2])
            rays = []
            # Lidar data starts at index 3: [angle1, dist1, angle2, dist2, ...]
            for i in range(num_rays):
                angle = float(data[3 + 2*i])
                dist = float(data[4 + 2*i]) / 1000.0 # Convert mm to meters
                rays.append((angle, dist))
            return rays
        except (socket.timeout, ValueError, IndexError):
            return None

def characterize_lidar(duration=10, target_angle=0, tolerance=5):
    import parameters as params
    collector = LidarCollector(arduino_ip=params.arduinoIP, local_ip=params.localIP)
    
    print(f"\n--- Characterizing Lidar Noise ---")
    print(f"Target Angle: {target_angle}° (±{tolerance}° window)")
    print(f"Collection Duration: {duration}s")
    input("Place robot 1-2m from a flat wall and press Enter...")
    
    start_time = time.time()
    collected_distances = []
    
    try:
        while time.time() - start_time < duration:
            rays = collector.receive_lidar_data()
            if rays:
                for angle, dist in rays:
                    # Handle angle wrapping/windowing
                    diff = abs(angle - target_angle)
                    if diff > 180: diff = 360 - diff
                    
                    if diff <= tolerance and dist > 0.1: # Ignore too-close readings
                        collected_distances.append(dist)
            
            if len(collected_distances) % 50 == 0 and len(collected_distances) > 0:
                print(f"Collected {len(collected_distances)} samples...")
                
    except KeyboardInterrupt:
        print("\nInterrupted.")

    if not collected_distances:
        print("Error: No data points collected. Check robot/lidar connection.")
        return

    # Statistics
    dist_arr = np.array(collected_distances)
    mean_dist = np.mean(dist_arr)
    variance = np.var(dist_arr)
    std_dev = np.std(dist_arr)
    
    print("\n--- RESULTS ---")
    print(f"Samples:    {len(collected_distances)}")
    print(f"Mean Dist:  {mean_dist:.4f} m")
    print(f"Variance:   {variance:.8f} m^2")
    print(f"Std Dev:    {std_dev:.4f} m")
    print("-" * 20)
    print(f"ADD TO parameters.py: distance_variance = {variance:.8f}")
    
    # Plotting Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(dist_arr, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(mean_dist, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_dist:.3f}m')
    plt.title(f"Lidar Range Distribution (Variance = {variance:.6f})")
    plt.xlabel("Distance (meters)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_file = "lidar_characterization.png"
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=10, help="Seconds to collect data")
    parser.add_argument("--angle", type=float, default=0.0, help="Target angle in degrees")
    parser.add_argument("--tol", type=float, default=5.0, help="Angle tolerance window")
    args = parser.parse_args()
    
    characterize_lidar(duration=args.duration, target_angle=args.angle, tolerance=args.tol)
