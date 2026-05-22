#!/usr/bin/env python3
"""
Runway Corner Projection System
Projects runway corners from GPS coordinates to image

Coordinate Transform: GPS → ECEF → NED → Body → Camera → Image
Navigation Frame (NED): North=X, East=Y, Down=Z  
Camera Frame (RDF): Right=X, Down=Y, Front=Z
"""

import cv2
import numpy as np
import argparse
from typing import Dict, Optional
import pymap3d as pm

class CoordinateTransformer:
    """Coordinate transformation utilities following the step-by-step approach"""
    
    def __init__(self):
        # WGS84 ellipsoid parameters
        self.a = 6378137.0
        self.f = 1 / 298.257223563
        self.e_sq = 2*self.f - self.f**2
    
    def lla_to_ecef(self, lat_deg: float, lon_deg: float, alt: float) -> np.ndarray:
        """Convert LLA to ECEF coordinates using WGS84 model"""
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)
        N = self.a / np.sqrt(1 - self.e_sq * np.sin(lat_rad)**2)
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - self.e_sq) + alt) * np.sin(lat_rad)
        return np.array([x, y, z])
    
    def ecef_to_enu(self, ecef_target: np.ndarray, lat_origin_deg: float, lon_origin_deg: float, alt_origin: float) -> np.ndarray:
        """Convert ECEF to ENU coordinates relative to origin"""
        # Get origin ECEF coordinates
        ecef_origin = self.lla_to_ecef(lat_origin_deg, lon_origin_deg, alt_origin)
        
        # Calculate delta ECEF
        delta_ecef = ecef_target - ecef_origin

        # Build rotation matrix from ECEF to NED
        lat_origin_rad = np.radians(lat_origin_deg)
        lon_origin_rad = np.radians(lon_origin_deg)
        R_ecef_to_enu = np.array([
            [-np.sin(lon_origin_rad),                          np.cos(lon_origin_rad),                          0                    ],
            [-np.sin(lat_origin_rad) * np.cos(lon_origin_rad), -np.sin(lat_origin_rad) * np.sin(lon_origin_rad),  np.cos(lat_origin_rad)],
            [np.cos(lat_origin_rad) * np.cos(lon_origin_rad), np.cos(lat_origin_rad) * np.sin(lon_origin_rad), np.sin(lat_origin_rad)]
        ])
        
        # Transform to ENU
        P_enu = R_ecef_to_enu @ delta_ecef
        return P_enu

    def enu_to_body_to_camera(self, corner_enu: np.ndarray, yaw_deg: float, pitch_deg: float, roll_deg: float, t_cam_in_body: np.ndarray) -> np.ndarray:
        """Convert ENU to Camera coordinates relative to origin"""
        #航向角 ψ（载体纵轴与北的夹角，顺时针为正）、俯仰角 θ（载体纵轴与水平面的夹角，抬头为正）、横滚角 φ（载体绕纵轴的旋转角，右滚为正）。
        #the pitch value during approaching should be negative
        yaw_rad, pitch_rad, roll_rad = np.radians(yaw_deg), -np.radians(pitch_deg), -np.radians(roll_deg)

        R_yaw = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])#Z
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])#X
        R_roll = np.array([
            [np.cos(roll_rad), 0, np.sin(roll_rad)],
            [0, 1, 0],
            [-np.sin(roll_rad), 0, np.cos(roll_rad)]
        ])#Y

        R_enu_to_body = R_roll @ R_pitch @ R_yaw 

        #convert from body coordinate [lateral,forward,upward] to camera coordinate[lateral,downward,forward]
        R_body_to_cam = np.array([[1, 0, 0],   # Camera X (lateral) = Body X (lateral)
                                  [0, 0, -1],   # Camera Y (downward) = -Body Z (upward)
                                  [0, 1, 0]])  # Camera Z (forward) = Body Y (forward)

        t_offset_camera = R_body_to_cam @ t_cam_in_body
        R_enu_to_camera = R_body_to_cam @ R_enu_to_body
        cam_pos = R_enu_to_camera @ corner_enu - t_offset_camera

        #R_enu_to_camera = R_body_to_cam @ R_enu_to_body
        #camera_offset_enu = R_enu_to_camera @ t_cam_in_body
        #corner_camera_rel = corner_enu - camera_offset_enu
        #cam_pos = R_enu_to_camera @ corner_camera_rel

        return cam_pos

    def camera_to_pixel(self, corner_camera: np.ndarray, camera_params: np.ndarray) -> np.ndarray:
        
        X_c, Y_c, Z_c = corner_camera[0], corner_camera[1], corner_camera[2]
            
        # Safety check: point must be in front of camera
        if Z_c <= 0:
            return np.array([-1, -1])
            
        # Get camera intrinsics
        fx = camera_params[0]
        fy = camera_params[1]
        cx = camera_params[2]
        cy = camera_params[3]

        # Perspective projection
        u = fx * (X_c / Z_c) + cx
        v = fy * (Y_c / Z_c) + cy
        
        return np.array([u,v])

class RunwayProjector:
    """Runway corner projector"""
    
    def __init__(self):
        # Camera intrinsic parameters
        self.camera_params = {
            "fx": 1506.58,   # Focal length x (pixels)
            "fy": 1506.58,   # Focal length y (pixels)
            "cx": 960.0,     # Principal point x
            "cy": 540.0,      # Principal point y (optimized to match expected corner positions)
            "width": 1920,   # Image width
            "height": 1080   # Image height
        }
        
        # Camera mounting parameters (meters)
        self.camera_mounting = {
            "offset_x": 0.0,    # Lateral offset
            "offset_y": 27.65,  # Forward offset
            "offset_z": -1.830, # Up offset
            "offset_pitch": 0.0,       # Camera pitch angle (degrees)
            "offset_roll": 0.0,        # Camera roll angle (degrees)
            "offset_yaw": 0.0          # Camera yaw angle (degrees)
        }

        # Runway corners GPS coordinates (Beijing Capital Airport)
        self.runway_corners = [
            {"name": "bottom_left",  "latitude": 40.0553918, "longitude": 116.59979033, "altitude": 30},
            {"name": "bottom_right", "latitude": 40.05545686, "longitude": 116.60047592, "altitude": 30},
            {"name": "top_left",     "latitude": 40.08943081, "longitude": 116.59447129, "altitude": 25},
            {"name": "top_right",    "latitude": 40.08949503, "longitude": 116.59515247, "altitude": 25}
        ]
        
        # Corner colors (BGR format)
        self.colors = {
            'bottom_left': (0, 0, 255),    # Red
            'top_left': (0, 255, 0),       # Green  
            'top_right': (255, 0, 0),      # Blue
            'bottom_right': (255, 0, 255)  # Magenta
        }

        self.runway_heading_deg = 353 #353.15
        self.transformer = CoordinateTransformer()
        
        # Pre-compute corner ECEF coordinates
        self.corners_ecef = []
        for corner in self.runway_corners:
            ecef = self.transformer.lla_to_ecef(corner['latitude'], corner['longitude'], corner['altitude'])
            self.corners_ecef.append({'name': corner['name'], 'pos': ecef})
        
        # Pre-compute runway coordinate system origin (center of bottom edge)
        bottom_left_idx = 0
        bottom_right_idx = 1
        bottom_left_ecef = self.corners_ecef[bottom_left_idx]['pos']
        bottom_right_ecef = self.corners_ecef[bottom_right_idx]['pos']
        
        # Origin is at center of bottom edge
        self.runway_origin_ecef = (bottom_left_ecef + bottom_right_ecef) / 2
        self.runway_origin_lat = (self.runway_corners[bottom_left_idx]['latitude'] + self.runway_corners[bottom_right_idx]['latitude']) / 2
        self.runway_origin_lon = (self.runway_corners[bottom_left_idx]['longitude'] + self.runway_corners[bottom_right_idx]['longitude']) / 2
        self.runway_origin_alt = (self.runway_corners[bottom_left_idx]['altitude'] + self.runway_corners[bottom_right_idx]['altitude']) / 2
        
        runway_heading_rad = np.radians(self.runway_heading_deg)
        
        # In ENU coordinates: East=X, North=Y, Up=Z
        # Y-axis (runway direction) in ENU coordinates
        self.runway_y_axis = np.array([
            np.sin(runway_heading_rad),  # East component (X in ENU)
            np.cos(runway_heading_rad),  # North component (Y in ENU)  
            0.0                         # No vertical component
        ])
        
        # X-axis: perpendicular to Y axis, pointing to the right
        # Right of runway direction is 90 degrees clockwise from runway heading
        right_heading_rad = np.radians(self.runway_heading_deg + 90.0)  # Right is 90° clockwise
        self.runway_x_axis = np.array([
            np.sin(right_heading_rad),   # East component (X in ENU)
            np.cos(right_heading_rad),   # North component (Y in ENU)
            0.0                          # No vertical component
        ])
        
        # Z-axis: up (completes right-handed coordinate system)
        self.runway_z_axis = np.array([0.0, 0.0, 1.0])  # Pure up in ENU coordinates
        
        # Normalize axes to ensure they are unit vectors
        self.runway_x_axis = self.runway_x_axis / np.linalg.norm(self.runway_x_axis)
        self.runway_y_axis = self.runway_y_axis / np.linalg.norm(self.runway_y_axis)
        self.runway_z_axis = self.runway_z_axis / np.linalg.norm(self.runway_z_axis)
    
    def project_corners(self, aircraft_pose: dict, debug_print: bool = False) -> Dict[str, Optional[np.ndarray]]:
        """Project runway corners to image coordinates using step-by-step transformation"""
        projected = {}

        # # Get aircraft pose parameters
        # aircraft_pose['latitude'] = 39.96594246
        # aircraft_pose['longitude'] = 116.61418566
        # aircraft_pose['altitude'] = 591.46221529
        # aircraft_pose['yaw'] = 352.87789917
        # aircraft_pose['pitch'] = 3.60612917
        # aircraft_pose['roll'] = 0.01789949

        lat_origin_deg = aircraft_pose['latitude']
        lon_origin_deg = aircraft_pose['longitude'] 
        alt_origin = aircraft_pose['altitude']
        yaw_deg = aircraft_pose['yaw']
        pitch_deg = aircraft_pose['pitch']
        roll_deg = aircraft_pose['roll']
        
        # Step 1 & 2: LLA -> ECEF -> ENU for each corner
        for corner in self.corners_ecef:
            corner_name = corner['name']
            
            # Step 1: LLA -> ECEF (already pre-computed)
            ecef_target = corner['pos']
        
            # Step 2: ECEF -> ENU
            corner_enu = self.transformer.ecef_to_enu(
                ecef_target, lat_origin_deg, lon_origin_deg, alt_origin
            )

            # Step 3: ENU -> Body -> Camera
            t_camera_in_body = np.array([
                self.camera_mounting['offset_x'],  # Lateral offset
                self.camera_mounting['offset_y'],  # Forward offset  
                self.camera_mounting['offset_z']   # Up offset
            ]) #camera in body coordinate

            corner_camera = self.transformer.enu_to_body_to_camera(
                corner_enu, yaw_deg, pitch_deg, roll_deg, t_camera_in_body
            )   
            
            # Step 4: Camera -> Pixel (projection)
            # Get camera intrinsics
            camera_parameters = np.array([self.camera_params['fx'],
                                          self.camera_params['fy'],
                                          self.camera_params['cx'],
                                          self.camera_params['cy']])
            corner_pixel = self.transformer.camera_to_pixel(corner_camera, camera_parameters)
            projected[corner_name] = corner_pixel
        
        return projected
    
    def solve_pnp_for_camera_pose(self, projected_corners: dict, runway_gps_corners: list = None, known_aircraft_pose: dict = None, debug_print: bool = False) -> dict:
        """
        Use PnP algorithm to estimate aircraft pose and position directly from projected runway corners
        
        Args:
            projected_corners: Dict mapping corner names to 2D image coordinates from project_corners function
            runway_gps_corners: List of runway corner GPS coordinates. If None, uses default runway corners.
            debug_print: Whether to print debug information
            
        Returns:
            Dictionary containing:
            - Airplane pose (yaw, pitch, roll) in degrees relative to runway
            - Airplane position in runway coordinates (x, y, z) in meters
            - Airplane position in W84 coordinates (lat, lon, alt) in degrees/meters
            - Error metrics comparing estimated with known truth
        """
        
        
        # Collect 3D points in runway coordinate system and 2D image points
        #debug_print = True
        valid_object_points = []
        valid_image_points = []
        valid_corner_names = []
        
        for corner in self.corners_ecef:
            corner_name = corner['name']
            
            # Check if this corner is visible in the projection
            if corner_name in projected_corners and projected_corners[corner_name] is not None:
                # Get 2D image coordinates from projection
                img_point_2d = projected_corners[corner_name]
                
                # Get 3D point coordinates in runway coordinate system
                # Transform corner position from ECEF to ENU relative to runway origin
                corner_enu = self.transformer.ecef_to_enu(
                    corner['pos'], 
                    self.runway_origin_lat, self.runway_origin_lon, self.runway_origin_alt
                )
                
                # Transform from ENU to runway coordinate system
                # ENU axes [East, North, Up] need to be mapped to runway axes
                corner_runway_x = np.dot(corner_enu, self.runway_x_axis)
                corner_runway_y = np.dot(corner_enu, self.runway_y_axis) 
                corner_runway_z = np.dot(corner_enu, self.runway_z_axis)
                corner_runway_coords = np.array([corner_runway_x, corner_runway_y, corner_runway_z])

                valid_object_points.append(corner_runway_coords)
                valid_image_points.append(img_point_2d)
                valid_corner_names.append(corner_name)
                
                if debug_print:
                    print(f"    {corner_name:12}: Runway=({corner_runway_x:.1f},{corner_runway_y:.1f},{corner_runway_z:.1f}) 2D=({img_point_2d[0]:.1f},{img_point_2d[1]:.1f})")
        
        if debug_print:
            print(f"  PnP: Using {len(valid_object_points)}/{len(self.corners_ecef)} visible corners")
        
        if len(valid_object_points) < 4:
            if debug_print:
                print(f"  PnP: Insufficient points ({len(valid_object_points)}/4)")
            return None
        
        # Convert to numpy arrays for PnP
        object_points_pnp = np.array(valid_object_points, dtype=np.float64)
        image_points_pnp = np.array(valid_image_points, dtype=np.float64)
        
        # Camera intrinsic matrix
        camera_matrix = np.array([
            [self.camera_params['fx'], 0, self.camera_params['cx']],
            [0, self.camera_params['fy'], self.camera_params['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Distortion coefficients (assuming no distortion)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        # Solve PnP to estimate camera pose relative to runway coordinate system
        success, rvec_est, tvec_est, inliers = cv2.solvePnPRansac(
            object_points_pnp, image_points_pnp, camera_matrix, dist_coeffs,
            iterationsCount=1000, reprojectionError=8.0, confidence=0.99
        )
        
        if not success:
            if debug_print:
                print("  PnP: Failed to solve")
            return None
        
        # Convert estimated rotation vector to rotation matrix
        # This gives us the rotation from runway coordinates to camera coordinates
        R_runway_to_camera, _ = cv2.Rodrigues(rvec_est)
        
        # Camera position in runway coordinates
        t_runway_to_camera = tvec_est.flatten()

        T_runway_to_camera = np.eye(4)
        T_runway_to_camera[:3, :3] = R_runway_to_camera
        T_runway_to_camera[:3, 3] = t_runway_to_camera
        
        R_camera_to_body = np.array([
            [1, 0, 0],   # Camera X (lateral) = Body X (lateral)
            [0, 0, 1],  # Camera Y (downward) = -Body Z (upward)  
            [0, -1, 0]    # Camera Z (forward) = Body Y (forward)
        ])

        t_camera_to_body = np.array([
            self.camera_mounting['offset_x'],  # Lateral (X)
            self.camera_mounting['offset_y'],  # Forward (Y)
            self.camera_mounting['offset_z']   # Upward (Z)
        ])

        T_camera_to_body = np.eye(4)
        T_camera_to_body[:3, :3] = R_camera_to_body
        T_camera_to_body[:3, 3] = t_camera_to_body

        T_runway_to_body = T_camera_to_body @ T_runway_to_camera
        R_runway_to_body = T_runway_to_body[:3, :3]
        t_runway_to_body = T_runway_to_body[:3, 3] 

        R_body_to_runway = R_runway_to_body.T
        aircraft_pos_runway = -R_body_to_runway @ t_runway_to_body
        
        sy = np.sqrt(R_body_to_runway[0,0] * R_body_to_runway[0,0] + R_body_to_runway[1,0] * R_runway_to_body[1,0])
        singular = sy < 1e-6
        
        if not singular:
            roll_est = np.arctan2(-R_body_to_runway[2,0], sy)
            pitch_est = np.arctan2(R_body_to_runway[2,1], R_body_to_runway[2,2])
            yaw_est = np.arctan2(R_body_to_runway[1,0], R_body_to_runway[0,0])
        else:
            roll_est = np.arctan2(-R_body_to_runway[2,0], sy)
            pitch_est = np.arctan2(-R_body_to_runway[1,2], R_body_to_runway[1,1])
            yaw_est = 0
        
        # Convert to degrees
        yaw_est = np.degrees(yaw_est)+self.runway_heading_deg
        pitch_est = np.degrees(pitch_est)
        roll_est = np.degrees(roll_est)
        
        # Calculate reprojection error
        projected_est, _ = cv2.projectPoints(object_points_pnp, rvec_est, tvec_est, camera_matrix, dist_coeffs)
        reprojection_error = np.mean(np.sqrt(np.sum((projected_est.reshape(-1, 2) - image_points_pnp)**2, axis=1)))
      
        # Convert estimated runway position back to GPS coordinates (W84)
        # Transform runway coordinates back to ENU
        est_aircraft_enu = (aircraft_pos_runway[0] * self.runway_x_axis + 
                           aircraft_pos_runway[1] * self.runway_y_axis + 
                           aircraft_pos_runway[2] * self.runway_z_axis)
        
        # Transform ENU back to ECEF and then to GPS
        origin_ecef = self.transformer.lla_to_ecef(
            self.runway_origin_lat, self.runway_origin_lon, self.runway_origin_alt
        )
        
        # Transform ENU to ECEF
        lat_rad = np.radians(self.runway_origin_lat)
        lon_rad = np.radians(self.runway_origin_lon)
        
        # ENU to ECEF rotation matrix (inverse of ECEF to ENU)
        R_enu_to_ecef = np.array([
            [-np.sin(lon_rad), -np.sin(lat_rad)*np.cos(lon_rad), np.cos(lat_rad)*np.cos(lon_rad)],
            [ np.cos(lon_rad), -np.sin(lat_rad)*np.sin(lon_rad), np.cos(lat_rad)*np.sin(lon_rad)],
            [ 0,               np.cos(lat_rad),                np.sin(lat_rad)]
        ])
        
        est_aircraft_ecef = origin_ecef + R_enu_to_ecef @ est_aircraft_enu
        
        # Convert ECEF back to GPS using pymap3d for accuracy
        lat_est, lon_est, alt_est = pm.ecef2geodetic(
            est_aircraft_ecef[0], est_aircraft_ecef[1], est_aircraft_ecef[2]
        )
        if debug_print:
            print(f"  PnP Results:")
            print(f"    Estimated Yaw: {yaw_est:.2f}°")
            print(f"    Estimated Pitch: {pitch_est:.2f}°")
            print(f"    Estimated Roll: {roll_est:.2f}°")
            print(f"    Estimated Latitude: {lat_est:.6f}°")
            print(f"    Estimated Longitude: {lon_est:.6f}°")
            print(f"    Estimated Altitude: {alt_est:.1f}m")
            print(f"    Runway Position (Est): X={aircraft_pos_runway[0]:.1f}m, Y={aircraft_pos_runway[1]:.1f}m, Z={aircraft_pos_runway[2]:.1f}m")
            print(f"    Reprojection Error: {reprojection_error:.2f} pixels")
            print(f"    Inliers: {len(inliers) if inliers is not None else 0}/{len(valid_object_points)}")
        
        return {
            # Airplane pose in runway coordinates
            'aircraft_pose_runway': {
                'yaw_deg': yaw_est,      # Heading angle relative to runway
                'pitch_deg': pitch_est,   # Pitch angle (positive = nose up)
                'roll_deg': roll_est      # Roll angle (positive = right wing down)
            },
            
            # Airplane position in runway coordinates (meters)
            'aircraft_position_runway': {
                'x_meters': aircraft_pos_runway[0],  # Lateral (positive = right of runway centerline)
                'y_meters': aircraft_pos_runway[1],  # Along runway (positive = toward approach end)
                'z_meters': aircraft_pos_runway[2]   # Vertical (positive = above runway level)
            },
            
            # Airplane position in W84 coordinates
            'aircraft_position_w84': {
                'latitude_deg': lat_est,    # Latitude in degrees
                'longitude_deg': lon_est,   # Longitude in degrees  
                'altitude_meters': alt_est # Altitude in meters above sea level
            },
            
            # Additional information
            'reprojection_error_pixels': reprojection_error,
            'inliers_count': len(inliers) if inliers is not None else 0,
            'used_corners': valid_corner_names
        }
    
    def put_text_safe(self, image: np.ndarray, text: str, position: tuple, 
                   font_scale: float = 0.5, color: tuple = (255, 255, 255), 
                   thickness: int = 1, font_face: int = cv2.FONT_HERSHEY_SIMPLEX) -> None:
        """Safely put text on image with character encoding handling"""
        try:
            # Ensure text is properly encoded
            if isinstance(text, str):
                # Replace any problematic characters
                clean_text = text.replace('°', 'deg').replace('±', '+/-')
                cv2.putText(image, clean_text, position, font_face, font_scale, color, thickness)
            else:
                cv2.putText(image, str(text), position, font_face, font_scale, color, thickness)
        except Exception as e:
            # Fallback to simple text if formatting fails
            try:
                fallback_text = str(text).encode('ascii', 'ignore').decode('ascii')
                cv2.putText(image, fallback_text, position, font_face, font_scale, color, thickness)
            except:
                pass

    def draw_corners(self, image: np.ndarray, projected: Dict[str, Optional[np.ndarray]], 
                    pnp_results: dict = None, aircraft_pose: dict = None) -> np.ndarray:
        """Draw projected corners on image with PnP comparison"""
        result = image.copy()
        
        # Connect visible corners to form runway outline
        order = ['bottom_left', 'top_left', 'top_right', 'bottom_right']
        points = [projected[name].astype(int) for name in order if projected.get(name) is not None]
        
        if len(points) >= 2:
            for i in range(len(points)):
                cv2.line(result, points[i], points[(i+1)%len(points)], (0, 255, 255), 2)
        
        # Draw corner points
        for i, name in enumerate(order):
            if projected.get(name) is not None:
                color = self.colors[name]
                cv2.circle(result, projected[name].astype(int), 5, color, -1)
                self.put_text_safe(result, name, projected[name].astype(int) + np.array([10, -10]), 
                                 font_scale=0.5, color=color)
        
        # Show visible corner count
        visible_count = sum(1 for p in projected.values() if p is not None)
        self.put_text_safe(result, f"Visible: {visible_count}/4", (10, 25), 
                          font_scale=0.6, color=(255, 255, 255))
        
        # Show PnP comparison if available
        if pnp_results and aircraft_pose:
            # Move all PnP pose estimation results to the left side of the image
            pnp_x = 10
            y_offset = 60
            self.put_text_safe(result, "PnP Pose Estimation:", (pnp_x, y_offset), 
                              font_scale=0.5, color=(0, 255, 0))
            
            # Get estimated pose from new structure
            est_pose = pnp_results['aircraft_pose_runway']
            est_pos = pnp_results['aircraft_position_runway']
            
            # Yaw comparison
            yaw_error = abs(est_pose['yaw_deg'] - aircraft_pose['yaw'])
            color = (0, 255, 0) if yaw_error < 5.0 else (0, 0, 255)
            yaw_text = f"Yaw: {est_pose['yaw_deg']:.1f}deg (err: {yaw_error:.1f}deg)"
            self.put_text_safe(result, yaw_text, (pnp_x, y_offset + 20), 
                              font_scale=0.4, color=color)
            
            # Pitch comparison
            pitch_error = abs(est_pose['pitch_deg'] - aircraft_pose['pitch'])
            color = (0, 255, 0) if pitch_error < 5.0 else (0, 0, 255)
            pitch_text = f"Pitch: {est_pose['pitch_deg']:.1f}deg (err: {pitch_error:.1f}deg)"
            self.put_text_safe(result, pitch_text, (pnp_x, y_offset + 40), 
                              font_scale=0.4, color=color)
            
            # Roll comparison
            roll_error = abs(est_pose['roll_deg'] - aircraft_pose['roll'])
            color = (0, 255, 0) if roll_error < 5.0 else (0, 0, 255)
            roll_text = f"Roll: {est_pose['roll_deg']:.1f}deg (err: {roll_error:.1f}deg)"
            self.put_text_safe(result, roll_text, (pnp_x, y_offset + 60), 
                              font_scale=0.4, color=color)
            
            # Reprojection error
            reproj_text = f"Reproj Error: {pnp_results['reprojection_error_pixels']:.1f}px"
            self.put_text_safe(result, reproj_text, (pnp_x, y_offset + 80), 
                              font_scale=0.4, color=(255, 255, 0))
            
            # Position information
            y_offset += 120
            self.put_text_safe(result, "Position Comparison:", (pnp_x, y_offset), 
                              font_scale=0.5, color=(0, 255, 255))
            
            # Ground truth position (from aircraft_pose)
            true_pos_text = f"True: (lat:{aircraft_pose.get('latitude', 0):.6f}, lon:{aircraft_pose.get('longitude', 0):.6f})"
            self.put_text_safe(result, true_pos_text, (pnp_x, y_offset + 20), 
                              font_scale=0.4, color=(0, 255, 0))
            
            # Estimated position from PnP (W84 coordinates)
            est_w84 = pnp_results['aircraft_position_w84']
            est_pos_text = f"Est: (lat:{est_w84['latitude_deg']:.6f}, lon:{est_w84['longitude_deg']:.6f})"
            self.put_text_safe(result, est_pos_text, (pnp_x, y_offset + 40), 
                              font_scale=0.4, color=(255, 255, 0))
            
            # Position difference (simple lat/lon error calculation)
            lat_error = abs(est_w84['latitude_deg'] - aircraft_pose['latitude']) * 111320  # meters
            lon_error = abs(est_w84['longitude_deg'] - aircraft_pose['longitude']) * 111320 * np.cos(np.radians(aircraft_pose['latitude']))  # meters
            pos_error = np.sqrt(lat_error**2 + lon_error**2)
            error_text = f"Pos Error: {pos_error:.1f}m"
            error_color = (0, 255, 0) if pos_error < 10.0 else (255, 255, 0) if pos_error < 50.0 else (0, 0, 255)
            self.put_text_safe(result, error_text, (pnp_x, y_offset + 60), 
                              font_scale=0.4, color=error_color)
            
            # Draw position indicators in runway coordinates on the left side of image
            img_width = result.shape[1]
            indicator_x = 200
            
            # Get true position in runway coordinates for comparison
            true_aircraft_ecef = self.transformer.lla_to_ecef(
                aircraft_pose['latitude'], 
                aircraft_pose['longitude'], 
                aircraft_pose['altitude']
            )
            true_aircraft_enu = self.transformer.ecef_to_enu(
                true_aircraft_ecef, 
                self.runway_origin_lat, self.runway_origin_lon, self.runway_origin_alt
            )
            true_runway_x = np.dot(true_aircraft_enu, self.runway_x_axis)
            true_runway_y = np.dot(true_aircraft_enu, self.runway_y_axis) 
            true_runway_z = np.dot(true_aircraft_enu, self.runway_z_axis)
            true_runway = np.array([true_runway_x, true_runway_y, true_runway_z])
            
            est_runway = np.array([est_pos['x_meters'], est_pos['y_meters'], est_pos['z_meters']])
            
            # Scale for visualization (pixels per meter)
            scale = 1.0  # 1 pixel per meter for runway coordinates
            center_x = indicator_x
            center_y = 200  # Center vertical position for runway coordinate display
            
            # Draw coordinate axes
            # X-axis (perpendicular to runway, pointing right) - horizontal
            cv2.arrowedLine(result, (center_x - 80, center_y), (center_x + 80, center_y), (255, 255, 255), 1)
            self.put_text_safe(result, "X (Right)", (center_x + 85, center_y), 
                              font_scale=0.3, color=(255, 255, 255))
            
            # Y-axis (along runway direction) - vertical (pointing up since runway points mostly north)
            cv2.arrowedLine(result, (center_x, center_y + 60), (center_x, center_y - 60), (255, 255, 255), 1)
            self.put_text_safe(result, "Y (Runway 353°)", (center_x + 5, center_y - 65), 
                              font_scale=0.3, color=(255, 255, 255))
            
            # True position indicator (green) in runway coordinates
            true_x = int(center_x + true_runway[0] * scale)  # X along runway
            true_y = int(center_y - true_runway[1] * scale)  # Y perpendicular (negative for screen coords)
            
            # Keep within display area bounds
            true_x = max(center_x - 150, min(center_x + 150, true_x))
            true_y = max(center_y - 100, min(center_y + 100, true_y))
            
            cv2.circle(result, (true_x, true_y), 6, (0, 255, 0), -1)
            cv2.circle(result, (true_x, true_y), 9, (0, 255, 0), 2)
            self.put_text_safe(result, "TRUE", (true_x - 20, true_y + 15), 
                              font_scale=0.3, color=(0, 255, 0))
            
            # Estimated position indicator (yellow) in runway coordinates
            est_x = int(center_x + est_runway[0] * scale)  # X along runway
            est_y = int(center_y - est_runway[1] * scale)  # Y perpendicular
            
            # Keep within display area bounds
            est_x = max(center_x - 150, min(center_x + 150, est_x))
            est_y = max(center_y - 100, min(center_y + 100, est_y))
            
            cv2.circle(result, (est_x, est_y), 5, (0, 255, 255), -1)
            cv2.circle(result, (est_x, est_y), 8, (0, 255, 255), 2)
            self.put_text_safe(result, "EST", (est_x - 18, est_y + 15), 
                              font_scale=0.3, color=(0, 255, 255))
            
            # Draw line connecting true and estimated positions
            cv2.line(result, (true_x, true_y), (est_x, est_y), (255, 255, 255), 1)
            
            # Draw scale reference (50m)
            scale_y = center_y + 80
            cv2.line(result, (center_x - 25, scale_y), (center_x + 25, scale_y), (255, 255, 255), 2)
            self.put_text_safe(result, "50m", (center_x - 35, scale_y + 12), 
                              font_scale=0.3, color=(255, 255, 255))
            
            # Display coordinate values
            coord_y = center_y - 120
            self.put_text_safe(result, "Runway Coordinates:", (center_x - 80, coord_y), 
                              font_scale=0.4, color=(255, 255, 255))
            self.put_text_safe(result, f"True: ({true_runway[0]:.0f}, {true_runway[1]:.0f}, {true_runway[2]:.0f})m", 
                              (center_x - 80, coord_y + 15), font_scale=0.3, color=(0, 255, 0))
            self.put_text_safe(result, f"Est: ({est_runway[0]:.0f}, {est_runway[1]:.0f}, {est_runway[2]:.0f})m", 
                              (center_x - 80, coord_y + 30), font_scale=0.3, color=(0, 255, 255))
            
            # Calculate and display error in runway coordinates
            error_xy = np.linalg.norm(est_runway[:2] - true_runway[:2])
            error_z = abs(est_runway[2] - true_runway[2])
            self.put_text_safe(result, f"Error: XY={error_xy:.1f}m Z={error_z:.1f}m", 
                              (center_x - 80, coord_y + 45), font_scale=0.3, color=(255, 255, 0))
        
        return result
    
    def compare_estimated_with_real(self, real_aircraft_pose: dict, estimated_results: dict, debug_print: bool = False) -> dict:
        """
        Compare estimated aircraft pose and position with real aircraft pose and position
        
        Args:
            real_aircraft_pose: Real aircraft pose and position from input file (contains lat, lon, alt, yaw, pitch, roll)
            estimated_results: Results from solve_pnp_for_camera_pose function
            debug_print: Whether to print debug information
            
        Returns:
            Dictionary containing comparison metrics and coordinates in both systems
        """
        if estimated_results is None:
            return None
        
        # Extract real pose and position
        real_pose = {
            'yaw_deg': real_aircraft_pose['yaw'],
            'pitch_deg': real_aircraft_pose['pitch'],
            'roll_deg': real_aircraft_pose['roll']
        }
        
        real_position_w84 = {
            'latitude_deg': real_aircraft_pose['latitude'],
            'longitude_deg': real_aircraft_pose['longitude'],
            'altitude_meters': real_aircraft_pose['altitude']
        }
        
        # Extract estimated pose and position
        est_pose = estimated_results['aircraft_pose_runway']
        est_position_runway = estimated_results['aircraft_position_runway']
        est_position_w84 = estimated_results['aircraft_position_w84']
        
        # Calculate pose errors
        yaw_error = self._normalize_angle_error(est_pose['yaw_deg'] - real_pose['yaw_deg'])
        pitch_error = est_pose['pitch_deg'] - real_pose['pitch_deg']
        roll_error = est_pose['roll_deg'] - real_pose['roll_deg']
        
        # Calculate position errors in W84 coordinates
        lat_error_meters = abs(est_position_w84['latitude_deg'] - real_position_w84['latitude_deg']) * 111320
        lon_error_meters = abs(est_position_w84['longitude_deg'] - real_position_w84['longitude_deg']) * 111320 * np.cos(np.radians(real_position_w84['latitude_deg']))
        alt_error_meters = est_position_w84['altitude_meters'] - real_position_w84['altitude_meters']
        horizontal_error_meters = np.sqrt(lat_error_meters**2 + lon_error_meters**2)
        
        # Convert real position to runway coordinates for comparison
        real_aircraft_ecef = self.transformer.lla_to_ecef(
            real_position_w84['latitude_deg'],
            real_position_w84['longitude_deg'], 
            real_position_w84['altitude_meters']
        )
        real_aircraft_enu = self.transformer.ecef_to_enu(
            real_aircraft_ecef,
            self.runway_origin_lat, self.runway_origin_lon, self.runway_origin_alt
        )
        real_position_runway = np.array([
            np.dot(real_aircraft_enu, self.runway_x_axis),
            np.dot(real_aircraft_enu, self.runway_y_axis),
            np.dot(real_aircraft_enu, self.runway_z_axis)
        ])
        
        est_pos_arr = np.array([est_position_runway['x_meters'], est_position_runway['y_meters'], est_position_runway['z_meters']])
        
        # Calculate position errors in runway coordinates
        runway_error = est_pos_arr - real_position_runway
        runway_error_horizontal = np.sqrt(runway_error[0]**2 + runway_error[1]**2)
        
        if debug_print:
            print(f"""
{'='*60}
AIRCRAFT POSE AND POSITION COMPARISON
{'='*60}

REAL AIRCRAFT DATA (from input file):
  W84 Position:
    Latitude:  {real_position_w84['latitude_deg']:.6f}°
    Longitude: {real_position_w84['longitude_deg']:.6f}°
    Altitude:  {real_position_w84['altitude_meters']:.1f}m
  Pose:
    Yaw:   {real_pose['yaw_deg']:.2f}°, Pitch: {real_pose['pitch_deg']:.2f}°, Roll:  {real_pose['roll_deg']:.2f}°
  Runway Coordinates:
    X: {real_position_runway[0]:.1f}m (lateral), Y: {real_position_runway[1]:.1f}m (along), Z: {real_position_runway[2]:.1f}m (vertical)

ESTIMATED AIRCRAFT DATA (from PnP):
  W84 Position:
    Latitude:  {est_position_w84['latitude_deg']:.6f}°
    Longitude: {est_position_w84['longitude_deg']:.6f}°
    Altitude:  {est_position_w84['altitude_meters']:.1f}m
  Pose:
    Yaw:   {est_pose['yaw_deg']:.2f}°, Pitch: {est_pose['pitch_deg']:.2f}°, Roll:  {est_pose['roll_deg']:.2f}°
  Runway Coordinates:
    X: {est_pos_arr[0]:.1f}m (lateral), Y: {est_pos_arr[1]:.1f}m (along), Z: {est_pos_arr[2]:.1f}m (vertical)

ERROR ANALYSIS:
  Pose Errors:     Yaw: {yaw_error:.2f}°, Pitch: {pitch_error:.2f}°, Roll: {roll_error:.2f}°
  Position W84:    Lat: {lat_error_meters:.2f}m, Lon: {lon_error_meters:.2f}m, Alt: {alt_error_meters:.2f}m
  Position Runway: X: {runway_error[0]:.2f}m, Y: {runway_error[1]:.2f}m, Z: {runway_error[2]:.2f}m
  Horizontal Err:  {horizontal_error_meters:.2f}m (W84), {runway_error_horizontal:.2f}m (Runway)
  Quality Metrics: Reprojection: {estimated_results['reprojection_error_pixels']:.2f}px, Inliers: {estimated_results['inliers_count']}
{'='*60}""")
        
        return {
            # Real aircraft data
            'real_aircraft': {
                'pose_w84': real_pose,
                'position_w84': real_position_w84,
                'position_runway': {
                    'x_meters': real_position_runway[0],
                    'y_meters': real_position_runway[1], 
                    'z_meters': real_position_runway[2]
                }
            },
            
            # Estimated aircraft data
            'estimated_aircraft': {
                'pose_w84': est_pose,
                'position_w84': est_position_w84,
                'position_runway': est_position_runway
            },
            
            # Error analysis
            'pose_errors': {
                'yaw_error_deg': yaw_error,
                'pitch_error_deg': pitch_error,
                'roll_error_deg': roll_error
            },
            
            'position_errors_w84': {
                'latitude_error_meters': lat_error_meters,
                'longitude_error_meters': lon_error_meters,
                'altitude_error_meters': alt_error_meters,
                'horizontal_error_meters': horizontal_error_meters
            },
            
            'position_errors_runway': {
                'x_error_meters': runway_error[0],
                'y_error_meters': runway_error[1],
                'z_error_meters': runway_error[2],
                'horizontal_error_meters': runway_error_horizontal
            },
            
            # Quality metrics
            'quality_metrics': {
                'reprojection_error_pixels': estimated_results['reprojection_error_pixels'],
                'inliers_count': estimated_results['inliers_count'],
                'used_corners': estimated_results['used_corners']
            }
        }
    
    def _normalize_angle_error(self, error_deg: float) -> float:
        """Normalize angle error to range [-180, 180] degrees"""
        while error_deg > 180:
            error_deg -= 360
        while error_deg < -180:
            error_deg += 360
        return error_deg
    

def load_poses(txt_path: str) -> list:
    """Load aircraft poses from text file"""
    poses = []
    try:
        with open(txt_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip() or line.startswith('#'): continue
                p = line.strip().split(',')
                if len(p) >= 10:
                    try:
                        poses.append({
                            'frame': int(p[0]), 'timestamp': float(p[1]),
                            'longitude': float(p[4]), 'latitude': float(p[5]),
                            'altitude': float(p[6]), 'pitch': float(p[7]),
                            'roll': float(p[8]), 'yaw': float(p[9])
                        })
                    except ValueError:
                        continue # 跳过如 "Frame, Timestamp..." 这样的表头文本行
    except Exception as e:
        print(f"Error loading poses: {e}")
    return poses


def main():
    parser = argparse.ArgumentParser(description="Runway corner projection")
    parser.add_argument('video', help='Video file')
    parser.add_argument('poses', help='Aircraft poses file')
    args = parser.parse_args()
    
    # Load poses
    poses = load_poses(args.poses)
    if not poses:
        print("Error: No poses loaded")
        return
    
    print(f"Loaded {len(poses)} poses")
    projector = RunwayProjector()
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames")
    
    frame_idx = pose_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find matching pose
        while (pose_idx < len(poses) - 1 and poses[pose_idx]['frame'] < frame_idx):
            pose_idx += 1
        
        if pose_idx < len(poses) and poses[pose_idx]['frame'] == frame_idx:
            pose = poses[pose_idx]
            projected = projector.project_corners(pose, debug_print=(frame_idx < 10))
            
            # Solve PnP to estimate camera pose and compare with true pose
            pnp_results = projector.solve_pnp_for_camera_pose(
                projected, 
                runway_gps_corners=None,  # Use default runway corners
                debug_print=(frame_idx < 10),
                known_aircraft_pose=pose  # Pass known pose for error analysis
            )
            
            # Compare estimated with real aircraft pose and position
            comparison_results = projector.compare_estimated_with_real(
                real_aircraft_pose=pose,
                estimated_results=pnp_results,
                debug_print=(frame_idx < 10)
            )
            
            frame = projector.draw_corners(frame, projected, pnp_results, pose)
        
        cv2.imshow('Runway Projection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
