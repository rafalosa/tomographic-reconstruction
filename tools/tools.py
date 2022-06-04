from typing import Union, Callable, Tuple
import numpy as np
import scipy.fft
from multiprocessing import shared_memory as sm


def rotate_vector(vector: Union[np.ndarray, tuple], angle: float) -> np.ndarray:

    rot_matrix = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    return np.dot(rot_matrix, vector)


def generate_filter(data_length, filter_function: Union[Callable, str], cutoff) -> np.ndarray:
    """
    Function responsible for generating a filter in the Fourier domain. Filter can be defined as str:
    'ramp','cosine','shepp-logan' which are the most common. Filter can be also defined as a custom function
    in which case the standard ramp filter is multiplied by the function's result. The function requires 2
    input paramaters - frequency domain and cutoff frequency of the filter.
    """

    spatial_domain = np.array(range(data_length))
    ramp_spatial = [0 if n % 2 == 0 and n != 0 else .25 if n == 0.0 else -1 / (n * np.pi) ** 2
                    for n in spatial_domain - data_length // 2]
    freq_filter = 2 * np.abs(scipy.fft.fftshift(scipy.fft.fft(ramp_spatial)))
    freq_domain = np.linspace(-np.pi, np.pi, data_length)

    if filter_function == 'ramp':
        def filter_function(*args):
            return 1

    elif filter_function == 'cosine':
        def filter_function(*args):
            return np.abs(np.cos(args[0] / 2 / args[1]))

    elif filter_function == 'shepp-logan':
        def filter_function(*args):
            return np.abs(np.sin(args[0] / 2 / args[1]) / args[0] / 2 / args[1])

    elif type(filter_function) == str:
        raise FilterFunctionError("No filter defined as " + filter_function)

    elif callable(filter_function):
        try:
            filter_function(freq_domain[(freq_domain != 0)], cutoff)
        except (TypeError, IndexError):
            raise FilterFunctionError("Filter function takes only 2 parameters.")

    else:
        raise FilterFunctionError("filter_function takes only str or callable parameter.")

    freq_filter[freq_domain != 0] *= filter_function(freq_domain[(freq_domain != 0)], cutoff)

    freq_filter = np.array([0 if np.abs(arg) > np.pi * cutoff or arg == 0
                            else val for arg, val in zip(freq_domain, freq_filter)])

    return freq_filter


def transform_reference_frame_point(t, s, angle: float, x_offset: float, y_offset: float,
                                    back_translation: bool = True) -> Tuple[Union[float, np.ndarray],
                                                                            Union[float, np.ndarray]]:

    """The patient's reference frame is stationary, detector's reference frame
    rotates around the (0,0,0) point. Due to the way that matplotlib loads images (the image loads
    in the 1st quadrant of the coordinate system), translation transformation is first required before
    rotating the reference frame"""

    x = (t - x_offset) * np.cos(angle) - (s - y_offset) * np.sin(angle)
    y = (t - x_offset) * np.sin(angle) + (s - y_offset) * np.cos(angle)

    if back_translation:
        return x + x_offset, y + y_offset
    else:
        return x, y


def evaluate_for_parallel_rays(lines_t_size: int, lines_s_size: int, image_shape: tuple,
                               dtype: type, memory_block_name: str, angle: float) -> np.ndarray:
    width, height, _ = image_shape
    lines_t = np.linspace(0, width, lines_t_size)
    lines_s = np.linspace(0, height, lines_s_size)
    sinogram_row = np.zeros([len(lines_t), 1])
    image_mem = sm.SharedMemory(name=memory_block_name)
    image = np.ndarray(shape=image_shape, dtype=dtype, buffer=image_mem.buf)

    for t_index, t in enumerate(lines_t):
        integral = 0
        for s in lines_s:  # Computing the Radon transform for each generated X-ray
            x, y = transform_reference_frame_point(t, s, angle, width / 2,
                                                   height / 2)  # Transforming between reference frames

            if int(np.floor(x)) - 1 >= width or int(np.floor(y)) - 1 >= height \
                    or int(np.floor(x)) - 1 < 0 or int(np.floor(y)) - 1 < 0:  # Mapping coordinates to pixel value

                integral += 0

            else:
                integral += np.sum(image[int(np.floor(x)) - 1, int(np.floor(y)) - 1, 0:2]) / 3

        sinogram_row[t_index] = integral

    image_mem.close()

    return sinogram_row


def pad_matrix(matrix: np.ndarray, new_size: Tuple[int, int], pad_color: float) -> np.ndarray:
    new_matrix = np.ones(new_size) * pad_color
    if len(matrix) == 2:
        width, height = matrix.shape
    else:
        width, height = (matrix.shape[0], 0)
    left_offset = (new_size[0] - width) // 2
    top_offset = (new_size[1] - height) // 2
    for ind, row in enumerate(new_matrix[top_offset:top_offset + height]):
        row[left_offset:left_offset + width] += matrix[ind]

    return new_matrix


def crop_matrix_center(matrix: np.ndarray, new_matrix_size: Tuple[int, int]) -> np.ndarray:
    width, height = matrix.shape
    left_offset = width // 2 - new_matrix_size[0] // 2
    top_offset = height // 2 - new_matrix_size[1] // 2
    return matrix[top_offset:top_offset + new_matrix_size[1], left_offset:left_offset + new_matrix_size[0]]


def evaluate_for_fan_rays(initial_source_position: tuple, resolution: int, path_resolution: int,
                          cone_angle: float, radius: float, image_shape: tuple,
                          dtype: type, memory_block_name: str, frame_angle: float) -> np.ndarray:
    image_mem = sm.SharedMemory(name=memory_block_name)
    image = np.ndarray(shape=image_shape, dtype=dtype, buffer=image_mem.buf)
    width, height, _ = image_shape
    sinogram_row = np.zeros([resolution, 1])

    angles_rays = np.linspace((np.pi - cone_angle) / 2,
                              np.pi - (np.pi - cone_angle) / 2, resolution)  # Angles for each xray in initial position

    xray_source_position = rotate_vector(initial_source_position, frame_angle)

    for row_index, ray_ang in enumerate(angles_rays):  # Generating initial coordinates for each xray
        domain = generate_domain(ray_ang, radius, initial_source_position, path_resolution)
        ray = np.tan(ray_ang) * domain + initial_source_position[1]

        T, S = transform_reference_frame_point(domain, ray, frame_angle, 0, initial_source_position[1] + height / 2,
                                               back_translation=False)
        T += xray_source_position[0] + width / 2
        S += xray_source_position[1] + height / 2
        integral = 0
        for t, s in zip(T, S):
            if int(np.floor(t)) - 1 >= width or int(np.floor(s)) - 1 >= height \
                    or int(np.floor(t)) - 1 < 0 or int(np.floor(s)) - 1 < 0:

                integral += 0

            else:
                integral += np.sum(image[int(np.floor(t)) - 1, int(np.floor(s)) - 1, 0:2]) / 3

        sinogram_row[row_index] = integral

    image_mem.close()

    return sinogram_row


def generate_domain(angle: float, radius: float, source_coords: tuple, points_num: int) -> np.ndarray:
    if angle <= np.pi / 2:
        start_point = -radius * np.cos(angle)
        domain = np.linspace(start_point, source_coords[0], points_num)
        return domain
    else:
        end_point = radius * np.cos(np.pi - angle)
        domain = np.linspace(source_coords[0], end_point, points_num)
        return domain


class FilterFunctionError(Exception):
    def __init__(self, message='Filter function error.'):
        self.message = message
        super().__init__(message)