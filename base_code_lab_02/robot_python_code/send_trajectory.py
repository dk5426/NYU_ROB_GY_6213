import time
import parameters
from robot_python_code import UDPCommunication
from my_trajectory import TRAJECTORY_SEQUENCE

def send_trajectory():
    # Initialize UDP Communication
    print(f"Connecting to robot at {parameters.arduinoIP}:{parameters.arduinoPort}...")
    udp_comm = UDPCommunication(
        parameters.arduinoIP, 
        parameters.localIP, 
        parameters.arduinoPort, 
        parameters.localPort, 
        parameters.bufferSize
    )

    print("Starting trajectory execution...")
    
    try:
        for speed, steering, duration in TRAJECTORY_SEQUENCE:
            print(f"Sending: Speed={speed}, Steering={steering}, Duration={duration}s")
            
            # Send command loop for the duration
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < duration:
                # Format: "speed,steering"
                msg = f"{speed},{steering}"
                udp_comm.send_msg(msg)
                time.sleep(0.1) # Send at ~10Hz
                
        # Stop the robot
        print("Trajectory complete. Stopping robot.")
        for _ in range(5): # Send stop command multiple times to ensure receipt
            udp_comm.send_msg("0,0")
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nInterrupted! Stopping robot.")
        for _ in range(5):
            udp_comm.send_msg("0,0")
            time.sleep(0.1)

if __name__ == "__main__":
    send_trajectory()
