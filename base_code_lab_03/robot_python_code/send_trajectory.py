import time
import numpy as np
import parameters
import robot_python_code
from robot import Robot

def send_trajectory():
    # Initialize Robot and UDP Communication
    robot = Robot()
    
    print(f"Connecting to robot at {parameters.arduinoIP}:{parameters.arduinoPort}...")
    udp, udp_success = robot_python_code.create_udp_communication(
        parameters.arduinoIP, 
        parameters.localIP, 
        parameters.arduinoPort, 
        parameters.localPort, 
        parameters.bufferSize
    )
    
    if not udp_success:
        print("Failed to setup UDP communication. Exiting.")
        return

    robot.setup_udp_connection(udp)
    robot.connected_to_hardware = True

    # Define Trajectory (matching ekf_simulation.ipynb)
    # Format: (speed_cmd, steering_cmd, duration_sec)
    TRAJECTORY_SEQUENCE = [
        (100,  0,  1.0),   # Straight
        (100, 10,  2.0),   # Left turn
        ( 80,-10,  2.0),   # Right turn
        (100,  0,  1.0),   # Straight
    ]

    print("Starting trajectory execution with Online EKF monitoring...")
    print("Format: [Time] X: {:.3f}, Y: {:.3f}, Theta: {:.3f} | Cam: {}")
    
    # 1. Wait for a valid camera detection to set the initial pose
    print("Waiting for initial camera detection...")
    while True:
        # Flush buffer
        for _ in range(5):
            robot.camera_sensor_signal = robot.camera_sensor.get_signal(robot.camera_sensor_signal)
            
        if len(robot.camera_sensor_signal) >= 7 and any(robot.camera_sensor_signal[:3]):
            break
        time.sleep(0.1)
        
    init_cam_x = robot.camera_sensor_signal[0]
    init_cam_y = robot.camera_sensor_signal[1]
    init_cam_theta = robot.camera_sensor_signal[6]
    
    print(f"\nINITIAL CAMERA POSE -> X: {init_cam_x:.3f}, Y: {init_cam_y:.3f}, Theta: {init_cam_theta:.3f} rad\n")
    
    # Set the EKF state to this initial camera pose
    robot.extended_kalman_filter.state_mean = np.array([init_cam_x, init_cam_y, init_cam_theta])
    robot.extended_kalman_filter.predicted_state_mean = np.array([init_cam_x, init_cam_y, init_cam_theta])
    
    print("Waking up robot to sync encoders...")
    for _ in range(5):
        robot.msg_sender.send_control_signal([0, 0])
        robot.robot_sensor_signal = robot.msg_receiver.receive_robot_sensor_signal(robot.robot_sensor_signal)
        time.sleep(0.05)
        
    robot.extended_kalman_filter.last_encoder_counts = float(robot.robot_sensor_signal.encoder_counts)
    start_time_total = time.perf_counter()
    
    try:
        for speed, steering, duration in TRAJECTORY_SEQUENCE:
            print(f"\n--- Executing: Speed={speed}, Steering={steering}, Duration={duration}s ---")
            
            step_start_time = time.perf_counter()
            while time.perf_counter() - step_start_time < duration:
                loop_start = time.perf_counter()
                
                # 1. Update and send control signals
                robot.msg_sender.send_control_signal([speed, steering])
                
                # 2. Receive sensor data from robot
                robot.robot_sensor_signal = robot.msg_receiver.receive_robot_sensor_signal(robot.robot_sensor_signal)
                
                # 3. Get camera signal (if any)
                robot.camera_sensor_signal = robot.camera_sensor.get_signal(robot.camera_sensor_signal)
                
                # 4. Update EKF state estimate
                robot.update_state_estimate()
                
                # 5. Print current ABSOLUTE pose
                mu = robot.extended_kalman_filter.state_mean
                
                cam_seen = "YES" if (len(robot.camera_sensor_signal) >= 7 and any(robot.camera_sensor_signal[:3])) else "NO"
                elapsed = time.perf_counter() - start_time_total
                print(f"[{elapsed:6.2f}s] X: {mu[0]:6.3f}, Y: {mu[1]:6.3f}, Th: {mu[2]:6.3f} | Cam: {cam_seen}", end='\r')
                
                # 6. Maintain loop rate (DT = 0.05s)
                loop_duration = time.perf_counter() - loop_start
                sleep_time = parameters.DT - loop_duration
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Optional: Print warning if loop is too slow
                    # print(f"WARNING: Loop lag! ({loop_duration:.3f}s > {parameters.DT}s)")
                    pass

        # Stop the robot
        print("\n\nTrajectory complete. Stopping robot.")
        for _ in range(5):
            robot.msg_sender.send_control_signal([0, 0])
            time.sleep(0.1)
            
        final_x = robot.extended_kalman_filter.state_mean[0]
        final_y = robot.extended_kalman_filter.state_mean[1]
        final_theta = robot.extended_kalman_filter.state_mean[2]
        
        diff_x = final_x - init_cam_x
        diff_y = final_y - init_cam_y
        diff_theta = final_theta - init_cam_theta
        diff_theta = (diff_theta + np.pi) % (2 * np.pi) - np.pi
        
        print("\n=== FINAL RESULTS ===")
        print(f"Initial Camera -> X: {init_cam_x:6.3f}, Y: {init_cam_y:6.3f}, Theta: {init_cam_theta:6.3f} rad")
        print(f"Final EKF State-> X: {final_x:6.3f}, Y: {final_y:6.3f}, Theta: {final_theta:6.3f} rad")
        print(f"Difference     -> dX: {diff_x:6.3f}, dY: {diff_y:6.3f}, dTheta: {diff_theta:6.3f} rad\n")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted! Stopping robot.")
        for _ in range(5):
            robot.msg_sender.send_control_signal([0, 0])
            time.sleep(0.1)
    finally:
        robot.camera_sensor.close()

if __name__ == "__main__":
    send_trajectory()
