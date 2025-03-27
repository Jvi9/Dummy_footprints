# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:26:31 2025

@author: jvila
"""
import numpy as np
import math
import random
import geopy.distance
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
import numpy as np
import geopy.distance
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import geopy.distance
import os
import cv2
class Dummy:
    def __init__(self, name, image_path, lat, lon, alt, yaw, pitch, heading):
        """Store photo metadata: name, image, GPS coordinates, altitude, yaw, pitch, heading"""
        self.name = name
        self.image = None
        self.image_path = image_path
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.yaw = yaw
        self.pitch = pitch
        self.heading = heading
        self.footprint = None
        
        self.load_images() #Autocall the images data
        
    def load_images(self):
        img_path = os.path.join(self.image_path, self.name)
        if os.path.exists(img_path):  # Check if the exact file exists
            self.image = cv2.imread(img_path)
            if self.image is not None:
                print(f"{self.name} loaded.")
            else:
                print(f"Warning: {self.name} could not be loaded.")
        else:
            print(f"Warning: {self.name} not found in {image_path}")        
        return self.image                

    def find_overlapping_backwards(self, overlap_threshold=50):
        """Example function to estimate overlap (modify as needed)"""
        overlap = self.yaw + self.pitch + self.heading
        return overlap

    def rotation_matrix(self):
        """Create a rotation matrix for given yaw (heading), pitch, and roll angles."""
        yaw = np.radians(self.heading)
        pitch = np.radians(self.pitch)
        roll = np.radians(self.yaw)

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw),  0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        return Rz @ Ry @ Rx

    def project_to_ground(self, points):
        """Project 3D camera points onto the ground plane."""
        projected_points = []
        for point in points:
            scale = -self.alt / point[2]  # Scale factor to bring Z to 0 (ground)
            x_ground = point[0] * scale
            y_ground = point[1] * scale
            projected_points.append([x_ground, y_ground])
        return np.array(projected_points)

    def to_geographic_coordinates(self, offsets):
        """Convert projected ground points to geographic coordinates."""
        geographic_points = []
        origin = (self.lat, self.lon)
        
        for offset in offsets:
            new_point = geopy.distance.distance(meters=offset[0]).destination(origin, bearing=90)
            new_point = geopy.distance.distance(meters=offset[1]).destination(new_point, bearing=0)
            geographic_points.append((new_point.latitude, new_point.longitude))
        
        return geographic_points

    def calculate_camera_footprint(self):
        """
        Calculate the geographic footprint of the camera based on its 
        position, altitude, and orientation (heading, pitch, roll).
        
        :return: List of (lat, lon) footprint corners
        """
        # Camera sensor properties (example values)
        SENSOR_WIDTH = 14.8  # mm
        SENSOR_HEIGHT = 22.3  # mm
        FOCAL_LENGTH = 600  # mm (camera focal length)

        # Define sensor corners in 3D space
        corners_3d = np.array([
            [-SENSOR_WIDTH / 2, -SENSOR_HEIGHT / 2, FOCAL_LENGTH],
            [SENSOR_WIDTH / 2, -SENSOR_HEIGHT / 2, FOCAL_LENGTH],
            [SENSOR_WIDTH / 2, SENSOR_HEIGHT / 2, FOCAL_LENGTH],
            [-SENSOR_WIDTH / 2, SENSOR_HEIGHT / 2, FOCAL_LENGTH]
        ])

        # Apply rotation based on camera orientation
        rotation = self.rotation_matrix()
        rotated_corners = corners_3d @ rotation.T

        # Project to ground
        projected_corners_ground = self.project_to_ground(rotated_corners)

        # Convert to geographic coordinates
        self.footprint = self.to_geographic_coordinates(projected_corners_ground)

        return self.footprint  # List of (lat, lon) tuples
    
    
image_path = r'C:\Users\jvila\Desktop\Development_area\CANON_PHOTOS'

image_names = [f for f in os.listdir(image_path) if f.endswith(".JPG")]
dummy_dict = {}
for i in image_names:  # Look for the images
    dummy_dict[i] = Dummy(name=i,image_path=image_path, lat=0, lon=0, alt=10000, yaw=0, pitch=0, heading=0)

first_key, first_value = next(iter(dummy_dict.items()))
print(f"Key: {first_key}, Value img: {first_value.image}")

""" If you want to calculate the footprint you have to initiate the method
    called calculate_camera_footprint  and then just call the .footprint
    this is to save time when we do not need the footprint of every dummy"""
    
dummy_dict[image_names[0]].name
dummy1 = dummy_dict['20241023_104900_px_8J4A8673.JPG']
dummy1.calculate_camera_footprint() # first you have to use the method to calculate the footprint of any
dummy1.footprint