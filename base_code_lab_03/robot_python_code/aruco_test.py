import cv2
import cv2.aruco as aruco
import numpy as np
import parameters

def main():
    print("Starting Aruco Visualizer...")
    print("Press 'q' in the video window to quit.")
    
    # Initialize Camera (using DSHOW backend for stability on Windows)
    cap = cv2.VideoCapture(parameters.camera_id, cv2.CAP_AVFOUNDATION)
    
    # Force MJPG codec! Many webcams reject 1080p requests over USB 2.0 if using uncompressed YUYV
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # Explicitly pull 1280x720 to bypass the 1080p hardware sensor crop!
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Note: If it still returns 640x480, your specific webcam doesn't support 1080p via OpenCV DSHOW.
    
    print("\n--- Current Camera Properties ---")
    print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"Contrast:   {cap.get(cv2.CAP_PROP_CONTRAST)}")
    print(f"Saturation: {cap.get(cv2.CAP_PROP_SATURATION)}")
    print(f"Sharpness:  {cap.get(cv2.CAP_PROP_SHARPNESS)}")
    print(f"Auto-Focus: {cap.get(cv2.CAP_PROP_AUTOFOCUS)}")
    print(f"Exposure:   {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print("---------------------------------\n")
    
    # Configure Aruco detector
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
    aruco_params = aruco.DetectorParameters()
    # Turn on sub-pixel corner refinement. This is CRITICAL for stable Z-axis depth estimation!
    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Define object points for the marker (centered at origin)
    half_length = parameters.marker_length / 2.0
    obj_points = np.array([
        [-half_length, half_length, 0],
        [half_length, half_length, 0],
        [half_length, -half_length, 0],
        [-half_length, -half_length, 0]
    ], dtype=np.float32)
    
    # Delta Measurement States
    pos_1 = None
    pos_2 = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # --- Fixed Software Filter (Matching Calibration Settings) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Fixed parameters
        contrast_value = 22
        clahe_value = 1.6
        beta_value = -16
        blur_kernel = 0

        # Software contrast & brightness
        alpha = max(0.1, contrast_value / 50.0)
        gray = gray.astype(np.float32)
        gray = gray * alpha + beta_value
        gray = np.clip(gray, 0, 255)
        gray = gray.astype(np.uint8)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=clahe_value, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Optional blur (currently off)
        if blur_kernel > 0:
            k = blur_kernel * 2 + 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        
        # Create colored version of filtered image for drawing
        display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        current_x = None
        current_y = None
        current_z = None
        current_theta = None
        
        if ids is not None:
            # Draw detected marker on filtered image instead of original
            aruco.drawDetectedMarkers(display_frame, corners, ids)
            
            for i in range(len(ids)):
                if ids[i][0] == 2:  # Only look for marker ID 1
                    
                    # Estimate the 3D pose using SQPNP (Significantly better depth than IPPE_SQUARE)
                    ret_pnp, rvec, tvec = cv2.solvePnP(
                        obj_points, corners[i][0], 
                        parameters.camera_matrix, parameters.dist_coeffs, 
                        flags=cv2.SOLVEPNP_SQPNP
                    )
                    
                    if ret_pnp:
                        # Draw 3D axis on the marker
                        cv2.drawFrameAxes(
                            display_frame, parameters.camera_matrix, parameters.dist_coeffs, 
                            rvec, tvec, parameters.marker_length * 1.5
                        )
                        
                        # Extract the translation values in centimeters
                        current_x = tvec[0][0] * 100.0
                        current_y = tvec[1][0] * 100.0
                        current_z = tvec[2][0] * 100.0

                        # --- Z Temporal Smoothing ---
                        if hasattr(main, "prev_z"):
                            current_z = 0.75 * main.prev_z + 0.25 * current_z
                        main.prev_z = current_z


                        
                        # Extract the rotation matrix to find the yaw angle (theta)
                        R, _ = cv2.Rodrigues(rvec)
                        # theta is the rotation of the marker's X-axis around the camera's Z-axis
                        current_theta = np.arctan2(R[1, 0], R[0, 0])
                        
                        # Overlay the 3D position text on the video feed
                        cv2.putText(display_frame, f"LIVE -> X: {current_x:.1f}cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"        Y: {current_y:.1f}cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"        Z: {current_z:.2f}cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"      Yaw: {np.degrees(current_theta):.1f}deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Print lightly to terminal
                        print(f"X: {current_x:6.1f},  Y: {current_y:6.1f},  Z: {current_z:6.2f}, Yaw: {np.degrees(current_theta):6.1f}deg ({current_theta:.2f}rad)")
        else:
            cv2.putText(display_frame, "No Marker Found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw the saved Delta points on the screen
        cv2.putText(display_frame, "[1] Set Pos 1 | [2] Set Pos 2 | [r] Reset", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if pos_1:
            cv2.putText(display_frame, f"POS 1  -> X: {pos_1[0]:.1f}, Y: {pos_1[1]:.1f}, Z: {pos_1[2]:.2f}, Yaw: {np.degrees(pos_1[3]):.1f}deg", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if pos_2:
            cv2.putText(display_frame, f"POS 2  -> X: {pos_2[0]:.1f}, Y: {pos_2[1]:.1f}, Z: {pos_2[2]:.2f}, Yaw: {np.degrees(pos_2[3]):.1f}deg", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
        if pos_1 and pos_2:
            dx = pos_2[0] - pos_1[0]
            dy = pos_2[1] - pos_1[1]
            dz = pos_2[2] - pos_1[2]
            dtheta = pos_2[3] - pos_1[3]
            # Normalize angle difference between -pi and pi
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
            euclidean = np.sqrt(dx**2 + dy**2 + dz**2)
            
            cv2.putText(display_frame, f"DELTA  -> dx: {dx:.1f}, dy: {dy:.1f}, dz: {dz:.1f}, dtheta: {np.degrees(dtheta):.1f}deg", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, f"3D DIST-> {euclidean:.1f} cm", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Show detection on filtered image only
        cv2.imshow("Aruco Tracking Test (Filtered)", display_frame)
        
        # Keyboard handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            if current_x is not None:
                pos_1 = (current_x, current_y, current_z, current_theta)
                print(f"\n--- Point 1 Saved! {pos_1} ---\n")
            else:
                print("Cannot save Point 1: No marker in view.")
        elif key == ord('2'):
            if current_x is not None:
                pos_2 = (current_x, current_y, current_z, current_theta)
                print(f"\n--- Point 2 Saved! {pos_2} ---\n")
            else:
                print("Cannot save Point 2: No marker in view.")
        elif key == ord('r'):
            pos_1 = None
            pos_2 = None
            print("\n--- Points Reset! ---\n")

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")

if __name__ == "__main__":
    main()