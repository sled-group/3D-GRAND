import numpy as np
import logging
import torch
import re


def random_per_point_translation_in_place(pcd_data) -> None:
    """
    Jittering the point cloud data by a random value between -0.02 and 0.02

    Args:
        pcd_data: point cloud data in the form x, y, z

    """
    translations = (
        np.random.rand(pcd_data.shape[0], 3) - 0.5
    ) * 0.04  # Random values between -0.02 and 0.02
    pcd_data[:, -3:] += translations


def compute_max_extent_and_centroid(pcd_data, epsilon=1e-4) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        pcd_data : point cloud data in the form x, y, z
        epsilon (float, optional): buffer for the max_extent. Defaults to 1e-4.

    Returns:
        max_extent: maximum extent of the point cloud data in terms of the largest dimension
        centroid: centroid of the point cloud data
    """
    min_vals = pcd_data.min(axis=0)
    max_vals = pcd_data.max(axis=0)
    centroid = (min_vals + max_vals) / 2
    max_extent = np.max(max_vals - min_vals) + epsilon
    return max_extent, centroid


def unit_cube_normalization_in_place(
    pcd_data,
    max_extent,
    centroid,
):
    """
    Normalized data point in a unit cube between 0 and 1 for each x, y, z in-place

    Args:
        pcd_data: point cloud data in the form x, y, z

    """
    # translate the centroid to the origin
    pcd_data -= centroid

    # scale the data to fit within [-0.5, 0.5]
    pcd_data /= max_extent

    # translate it back to within [0, 1]
    pcd_data += 0.5


def point_to_index(point, grid_size):
    """
    Maps a point in the unit cube to a unique index based on the grid size.

    Args:
        point (tuple): a tuple of (x, y, z) coordinates of the point. Each coordinate should be in [0, 1].
        grid_size (int): the number of divisions along each axis.

    Returns:
        int: a unique index for the point.
    """

    xi = int(point[0] * grid_size)
    yi = int(point[1] * grid_size)
    zi = int(point[2] * grid_size)

    # Ensure that the point is inside the unit cube
    if not (0 <= xi < grid_size) or not (0 <= yi < grid_size) or not (0 <= zi < grid_size):
        logging.warning(
            f"The point is outside the unit cube: point: {point}, grid_index: ({xi}, {yi}, {zi})"
        )

    # Clamp the point to be inside the unit cube
    xi = min(max(xi, 0), grid_size - 1)
    yi = min(max(yi, 0), grid_size - 1)
    zi = min(max(zi, 0), grid_size - 1)

    # Compute the unique voxel ID, row-major order
    voxel_id = xi + yi * grid_size + zi * grid_size * grid_size

    return voxel_id


def scale_bbox(bbox_str, max_extent, centroid):
    """
    Scale the bounding box to be within a unit cube and output numerically tokenized bounding box.

    Args:
        bbox_str (str): A string representing a bounding box, in the format "<x_min,y_min,z_min,x_max,y_max,z_max>".
        max_extent (float): The maximum extent of the bounding box.
        centroid (np.array): The centroid of the bounding box.

    Returns:
        str: A string representing the scaled bounding box, in the same format as the input.
    """
    # Remove < and > from the bounding box string
    bbox_str = bbox_str.strip("<>")

    bbox_values = bbox_str.split(",")
    # Convert each string to a float and store in a list
    bbox_floats = [float(value) for value in bbox_values]
    # Convert the list to a numpy array
    bbox_array = np.array(bbox_floats)
    bbox_array[:3] -= centroid
    bbox_array[3:] -= centroid
    bbox_array /= max_extent
    bbox_array += 0.5
    x_min, y_min, z_min, x_max, y_max, z_max = bbox_array
    x_min, y_min, z_min, x_max, y_max, z_max = (
        x_min.item(),
        y_min.item(),
        z_min.item(),
        x_max.item(),
        y_max.item(),
        z_max.item(),
    )
    x_min, y_min, z_min, x_max, y_max, z_max = (
        round(x_min, 3),
        round(y_min, 3),
        round(z_min, 3),
        round(x_max, 3),
        round(y_max, 3),
        round(z_max, 3),
    )
    new_bbox_str = f"< {x_min}, {y_min}, {z_min}, {x_max}, {y_max}, {z_max}>"  # adding space after < because tokenizer will not merge < and first digit or negative sign
    return new_bbox_str


def voxelize_points(
    xyz_to_be_voxelized: np.array,
    scene_min_xyz: np.array,
    scene_max_xyz: np.array,
    num_voxels_per_axis: int,
):
    """Convert points to voxel indexes

    Args:
        xyz_to_be_voxelized (np.array): shape (num_points, 3)
        scene_min_xyz (np.array): shape (3,)
        scene_max_xyz (np.array): shape (3,)
        num_voxels_per_axis (int): number of voxels per axis

    Returns:
        voxel_id (np.array): shape (num_points,)
    """
    voxel_index = np.floor(
        (xyz_to_be_voxelized - scene_min_xyz)
        / (scene_max_xyz - scene_min_xyz)
        * num_voxels_per_axis
    ).astype(
        int
    )  # range after this overations: [0, num_voxels_per_axis]
    voxel_index = np.clip(
        voxel_index, 0, num_voxels_per_axis - 1
    )  # clamp range to [0, num_voxels_per_axis - 1]
    # calculate index using row-major order
    voxel_id = (
        voxel_index[:, 0]
        + voxel_index[:, 1] * num_voxels_per_axis
        + voxel_index[:, 2] * num_voxels_per_axis * num_voxels_per_axis
    )  # range after this operation: [0, num_voxels_per_axis ** 3 - 1]
    return voxel_id


def process_one_bbox_minkowski_loc_token(
    bbox_str, scene_min_xyz, scene_max_xyz, num_voxels_per_axis
):
    # Remove < and > from the bounding box string
    bbox_str = bbox_str.strip("<>")
    bbox_values = bbox_str.split(",")
    # Convert each string to a float and store in a list
    bbox_floats = [float(value) for value in bbox_values]
    # Convert the list to a numpy array
    bbox_array = np.array(bbox_floats)  # shape: (6,)
    bbox_array = bbox_array.reshape(2, 3)  # shape: (2, 3)

    voxel_indices = voxelize_points(
        bbox_array, scene_min_xyz, scene_max_xyz, num_voxels_per_axis
    )  # shape: (2,)

    new_bbox_str = f"<loc_{voxel_indices[0]}><loc_{voxel_indices[1]}>"
    return new_bbox_str


def scale_bbox_special_token(bbox_str, max_extent, centroid, num_grid_cells):
    """
    Special token for the bbox. The bbox is scaled to the unit cube and then converted
    to a unique index based on the grid size.

    Args:
        bbox_str (str): bbox string in the form "<x_min, y_min, z_min, x_max, y_max, z_max>"
        max_extent (float): max extent of the point cloud data in terms of the largest dimension
        centroid (np.array): centroid of the point cloud data
        num_grid_cells (int): number of grids along each axis

    Returns:
        two unique special tokens for the bbox as string
    """
    # Remove < and > from the bounding box string
    bbox_str = bbox_str.strip("<>")

    bbox_values = bbox_str.split(",")
    # Convert each string to a float and store in a list
    bbox_floats = [float(value) for value in bbox_values]
    # Convert the list to a numpy array
    bbox_floats = np.array(bbox_floats)
    bbox_floats[:3] -= centroid
    bbox_floats[3:] -= centroid
    bbox_floats /= max_extent
    bbox_floats += 0.5
    min_point = bbox_floats[:3]
    max_point = bbox_floats[3:]
    index_min = point_to_index(min_point, num_grid_cells)
    index_max = point_to_index(max_point, num_grid_cells)

    new_bbox_str = f"<loc_{index_min}><loc_{index_max}>"
    return new_bbox_str


def rotate_point_cloud_90_degrees(pcd_data):
    """
    Rotate the point cloud data by 90 degrees in the x-y plane

    Args:
        pcd_data: point cloud data in the form x, y, z

    Returns:
        pcd_data: rotated point cloud data in the form x, y, z
    """
    # Randomly select among no change, clockwise, and counterclockwise
    rotation_choices = ["no change", "clockwise", "counterclockwise"]
    direction = np.random.choice(rotation_choices)

    if direction == "clockwise":
        rotation_matrix = torch.tensor([[0, 1], [-1, 0]])
        # Apply rotation on x-y plane
        pcd_data[:, -3:-1] = torch.matmul(pcd_data[:, -3:-1], rotation_matrix)
    elif direction == "counterclockwise":
        rotation_matrix = torch.tensor([[0, -1], [1, 0]])
        # Apply rotation on x-y plane
        pcd_data[:, -3:-1] = torch.matmul(pcd_data[:, -3:-1], rotation_matrix)

    return pcd_data, direction


def adjust_bbox_after_rotation(bbox_str, direction):
    """_summary_

    Args:
        bbox_str (_type_): _description_
        direction (_type_): _description_

    Returns:
        _type_: _description_
    """

    if direction == "no change":
        return bbox_str

    values = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", bbox_str)))
    x_min, y_min, z_min, x_max, y_max, z_max = values

    if direction == "clockwise":
        # adding space after < because tokenizer will not merge < and first digit or negative sign
        new_bbox_str = f"< {y_min}, {x_min}, {z_min}, {y_max}, {x_max}, {z_max}>"
    else:  # counterclockwise
        # adding space after < because tokenizer will not merge < and first digit or negative sign
        new_bbox_str = f"< {x_max}, {y_min}, {z_min}, {x_min}, {y_max}, {z_max}>"

    return new_bbox_str
