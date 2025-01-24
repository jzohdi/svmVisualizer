from typing import List, Tuple
import numpy as np
from collections import Counter
from typing import Optional
from dataclasses import dataclass

@dataclass
class RangeValues:
    max_x: float
    min_x: float
    max_y: float
    min_y: float
    max_z: Optional[float] = None
    min_z: Optional[float] = None

    def to_dict(self):
        return {
            'max_x': self.max_x,
            'min_x': self.min_x,
            'max_y': self.max_y,
            'min_y': self.min_y,
            'max_z': self.max_z,
            'min_z': self.min_z
        }


def parse_request(data_set: List[List[int]], labels: List) -> Tuple['np.ndarray', 'np.ndarray', int, dict]:
    """
    Parse input data to extract coordinate ranges, labels, and data points.
    Returns processed data, labels, low_cv value, and range information.
    """
    dimensions = len(data_set[0])

    # Extract x and y data
    x_data = [float(entry[0]) for entry in data_set]
    y_data = [float(entry[1]) for entry in data_set]

    if len(x_data) != len(y_data):
        raise ValueError(f"x data length: {len(x_data)}. y data length: {len(y_data)}")

    # Initialize range values
    max_x, min_x = max(x_data), min(x_data)
    max_y, min_y = max(y_data), min(y_data)

    max_z, min_z = None, None

    if dimensions == 3:
        z_data = [float(entry[2]) for entry in data_set]

        if len(z_data) != len(y_data):
            raise ValueError(f"z data length: {len(z_data)}. x data length: {len(x_data)}")

        max_z, min_z = max(z_data), min(z_data)

    # Determine low_cv and data points based on dimensions
    data_points = None
    # clamp between 2 and 5
    low_cv = max(2, min(min(Counter(labels).values()), 5)) 

    if dimensions == 2:
        data_points = list(zip(x_data, y_data))
    elif dimensions == 3:
        data_points = list(zip(x_data, y_data, z_data))

    range_vals = RangeValues(
        max_x=max_x,
        min_x=min_x,
        max_y=max_y,
        min_y=min_y,
        max_z=max_z,
        min_z=min_z
    )
    return (
        np.array(data_points),
        np.array(labels),
        low_cv,
        range_vals
    )

def get_testing_map(testing_data: Optional[List[List[float]]], range_vals: RangeValues) -> np.ndarray:
    """
    Generates a testing map for a given range of values or returns the provided testing data.

    Parameters:
    - testing_data (Optional[List[List[float]]]): Predefined testing data. If provided, it is returned as is.
    - range_vals (RangeValues): Object containing the range of coordinates for x, y, and optionally z.

    Returns:
    - np.ndarray: An array of coordinates for the testing map.

    If testing_data is not provided, this function generates a grid of coordinates based on the range values:
    - For 2D data (x, y), a 50x50 grid is created.
    - For 3D data (x, y, z), a 25x25x25 grid is created.
    """
    if testing_data is not None:
        return np.array(testing_data)

    full_map_coordinates = []

    # Define ranges for x and y
    low_x = min(0, int(range_vals.min_x))
    high_x = int(range_vals.max_x)
    low_y = min(0, int(range_vals.min_y))
    high_y = int(range_vals.max_y)

    if range_vals.max_z is None:
        # Generate a 2D grid
        x_coords = np.linspace(low_x, high_x, 50).tolist()
        y_coords = np.linspace(low_y, high_y, 50).tolist()
        for x_coord in x_coords:
            for y_coord in y_coords:
                full_map_coordinates.append([x_coord, y_coord])
    else:
        # Generate a 3D grid
        low_z = min(0, int(range_vals.min_z))
        high_z = int(range_vals.max_z)

        x_coords = np.linspace(low_x, high_x, 25).tolist()
        y_coords = np.linspace(low_y, high_y, 25).tolist()
        z_coords = np.linspace(low_z, high_z, 25).tolist()
        for x_coord in x_coords:
            for y_coord in y_coords:
                for z_coord in z_coords:
                    full_map_coordinates.append([x_coord, y_coord, z_coord])

    return np.array(full_map_coordinates)