import numpy as np
import matplotlib.image as mpimg
import multiprocessing as mp
from multiprocessing import shared_memory as sm
from multiprocessing import Pool
from functools import partial
import scipy
import scipy.interpolate
from scipy import fft
from scipy import ndimage
from typing import Union, Tuple, Callable
import matplotlib.pyplot as plt

"""
Useful sources: http://bioeng2003.fc.ul.pt/Conference%20Files/papers/De%20Francesco,%20Fourier.pdf - English
http://ncbj.edu.pl/zasoby/wyklady/ld_podst_fiz_med_nukl-01/med_nukl_10_v3.pdf - Polish

"""

# todo: Implement other reconstruction techniques.
# todo: Check for correct sinogram orientation in loadSinogram.
# todo: Crop/pad loaded image so it is square.
# todo: Add multiprocessing to reconstruction methods.


class FilterFunctionError(Exception):
    def __init__(self,message='Filter function error.'):
        self.message = message
        super().__init__(message)


def rotate(vector: Union[np.ndarray, tuple], angle: float) -> np.ndarray:
    '''Function rotates a given vector counterclockwise by a given angle in radians (vector,angle)'''

    rot_matrix = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    return np.dot(rot_matrix, vector)


def generateFilter(data_length,filter_function:Union[Callable,str],cutoff) -> np.ndarray:
    '''Function responsible for generating a filter in the Fourier domain. Filter can be defined as str:
     'ramp','cosine','shepp-logan' which are the most common. Filter can be also defined as a custom function
     in which case the standard ramp filter is multiplied by the function's result. The function requires 2
     input paramaters - frequency domain and cutoff frequency of the filter.'''

    spatial_domain = np.array(range(data_length))
    ramp_spatial = [0 if n % 2 == 0 and n != 0 else .25 if n == 0.0 else -1 / (n * np.pi) ** 2
                    for n in spatial_domain - data_length // 2]
    freq_filter = 2*np.abs(scipy.fft.fftshift(scipy.fft.fft(ramp_spatial)))
    freq_domain = np.linspace(-np.pi, np.pi, data_length)

    if filter_function == 'ramp':
        def filter_function(*args): return 1

    elif filter_function == 'cosine':
        def filter_function(*args): return np.abs(np.cos(args[0]/2/args[1]))

    elif filter_function == 'shepp-logan':
        def filter_function(*args): return np.abs(np.sin(args[0] / 2 / args[1])/args[0]/2/args[1])

    elif type(filter_function) == str:
        raise FilterFunctionError("No filter defined as " + filter_function)

    elif callable(filter_function):
        try:
            filter_function(freq_domain[(freq_domain != 0)],cutoff)
        except (TypeError,IndexError):
            raise FilterFunctionError("Filter function takes only 2 parameters.")

    else:
        raise FilterFunctionError("filter_function takes only str or callable parameter.")

    freq_filter[freq_domain != 0] *= filter_function(freq_domain[(freq_domain != 0)],cutoff)

    freq_filter = np.array([0 if np.abs(arg) > np.pi*cutoff or arg == 0
                            else val for arg,val in zip(freq_domain,freq_filter)])
    plt.plot(freq_filter)
    plt.show()

    return freq_filter


def transformReferenceFramePoint(t, s, angle: float, x_offset: float, y_offset: float,
                                 back_translation: bool = True) -> Tuple[float, float]:
    '''Function that transforms points between the patient's reference frame and detector's reference frame,
    transformation consists of translation, rotation and back translation'''

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


def evaluateForParallelRays(lines_t_size: int, lines_s_size: int, image_shape: tuple,
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
            x, y = transformReferenceFramePoint(t, s, angle, width / 2,
                                                height / 2)  # Transforming between reference frames

            if int(np.floor(x)) - 1 >= width or int(np.floor(y)) - 1 >= height \
                    or int(np.floor(x)) - 1 < 0 or int(np.floor(y)) - 1 < 0:  # Mapping coordinates to pixel value

                integral += 0

            else:
                integral += np.sum(image[int(np.floor(x)) - 1, int(np.floor(y)) - 1, 0:2]) / 3

        sinogram_row[t_index] = integral

    image_mem.close()

    return sinogram_row


def padMatrix(matrix: np.ndarray, new_size: Tuple[int, int], pad_color: float):
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


def cropCenterMatrix(matrix: np.ndarray, new_matrix_size: Tuple[int, int]) -> np.ndarray:
    width, height = matrix.shape
    left_offset = width // 2 - new_matrix_size[0] // 2
    top_offset = height // 2 - new_matrix_size[1] // 2
    return matrix[top_offset:top_offset + new_matrix_size[1], left_offset:left_offset + new_matrix_size[0]]


def evaluateForFanRays(initial_source_position: tuple, resolution: int, path_resolution: int,
                       cone_angle: float, radius: float, image_shape: tuple,
                       dtype: type, memory_block_name: str, frame_angle: float) -> np.ndarray:
    image_mem = sm.SharedMemory(name=memory_block_name)
    image = np.ndarray(shape=image_shape, dtype=dtype, buffer=image_mem.buf)
    width, height, _ = image_shape
    sinogram_row = np.zeros([resolution, 1])

    angles_rays = np.linspace((np.pi - cone_angle) / 2,
                              np.pi - (np.pi - cone_angle) / 2, resolution)  # Angles for each xray in initial position

    xray_source_position = rotate(initial_source_position, frame_angle)

    for row_index, ray_ang in enumerate(angles_rays):  # Generating initial coordinates for each xray
        domain = generateDomain(ray_ang, radius, initial_source_position, path_resolution)
        ray = np.tan(ray_ang) * domain + initial_source_position[1]

        T, S = transformReferenceFramePoint(domain, ray, frame_angle, 0,
                                            initial_source_position[1] + height / 2,
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


def generateDomain(angle: float, radius: float, source_coords: tuple, points_num: int) -> np.ndarray:
    if angle <= np.pi / 2:
        start_point = -radius * np.cos(angle)
        domain = np.linspace(start_point, source_coords[0], points_num)
        return domain
    else:
        end_point = radius * np.cos(np.pi - angle)
        domain = np.linspace(source_coords[0], end_point, points_num)
        return domain


class Scan:

    def __init__(self):
        self.image = None
        self.sinogram = None
        self.width = None
        self.height = None
        self._filter = None

    def generateSinogram(self, resolution: int, path_resolution: int, processes: int = 0) -> None:

        '''This function generates a sinogram for a given image. Parameters (resolution,path_resolution,processes)
        resolution determines how many Xray beams are used to generate the sinogram as well as how many angular
        positions are registered. path_resolution determines how many datapoints are summed in the process
        of generating a sinogram for a given Xray. The bigger the values the more detailed sinogram will be generated
        and the longer the computation time. processes dictates how many processes can be used to generate the sinogram.
        If not specified, all but 2 CPU threads will remain unused by this process.'''

        # Generating a sinogram can obviously be done better and quicker, this implementation represents the real
        # measurement process and is pretty slow.

        if self.image is None:
            raise RuntimeError("Image required before generating a sinogram")

        number_of_rays = resolution  # How many X-ray beams/detector cells
        angle_resolution = resolution  # Angular step
        angles = np.linspace(np.pi / 2, np.pi * 3 / 2, angle_resolution + 1)
        image_memory_shared = sm.SharedMemory(create=True, size=self.image.nbytes)
        image_shared_copy = np.ndarray(self.image.shape, dtype=self.image.dtype, buffer=image_memory_shared.buf)
        image_shared_copy[:, :, :] = self.image[:, :, :]

        if processes:
            CPU_S = processes
        else:
            CPU_S = mp.cpu_count() - 2

        pool = Pool(processes=CPU_S)

        function = partial(evaluateForParallelRays, number_of_rays + 1, path_resolution + 1,
                           self.image.shape, self.image.dtype, image_memory_shared.name)

        results = pool.imap(function, (ang for ang in angles))
        pool.close()
        pool.join()

        sinogram = np.transpose(np.hstack([row for row in results]))

        image_memory_shared.close()
        image_memory_shared.unlink()

        self.sinogram = sinogram

    def fanBeamSinogram(self, resolution: int, path_resolution: int, cone_angle_deg: float, processes: int = 0) -> None:
        # Probably implement this method to generateSinogram with an additional bool parameter

        if cone_angle_deg >= 180 or cone_angle_deg <= 0:
            raise AttributeError("The angle of the fan beam must be between 0 and 180 degrees.")

        if self.image is None:
            raise RuntimeError("Image required before generating a sinogram")

        cone_angle = np.deg2rad(cone_angle_deg)

        xray_source_initial_position = (0, self.height + self.width / 2 / np.tan(cone_angle / 2))
        xray_radius = self.height * np.sqrt(2) + xray_source_initial_position[1] / 3
        #reference_frame_angles = np.linspace(np.pi / 2, np.pi * 3 / 2, resolution)
        reference_frame_angles = np.linspace(0, np.pi * 2, resolution)
        # Angles for rotating the source-detector reference frame
        image_memory_shared = sm.SharedMemory(create=True, size=self.image.nbytes)
        image_shared_copy = np.ndarray(self.image.shape, dtype=self.image.dtype, buffer=image_memory_shared.buf)
        image_shared_copy[:, :, :] = self.image[:, :, :]

        if processes:
            CPU_S = processes
        else:
            CPU_S = mp.cpu_count() - 2

        pool = Pool(processes=CPU_S)
        function = partial(evaluateForFanRays, xray_source_initial_position, resolution,
                           path_resolution, cone_angle, xray_radius, self.image.shape,
                           self.image.dtype, image_memory_shared.name)

        results = pool.imap(function, (ang for ang in reference_frame_angles))
        pool.close()
        pool.join()

        sinogram = np.transpose(np.hstack([row for row in results]))

        image_memory_shared.close()
        image_memory_shared.unlink()

        self.sinogram = sinogram

    def loadSinogram(self, path: str) -> None:
        sinogram = mpimg.imread(path)

        if sinogram.shape[0] != sinogram.shape[1]:
            raise AttributeError("Image has to be square")

        sinogram_sum = np.sum(sinogram, axis=2)
        self.sinogram = sinogram_sum

    def loadImage(self, path: str) -> None:
        img = mpimg.imread(path)

        if img.shape[0] != img.shape[1]:
            raise AttributeError("Image has to be square")

        self.image = img
        self.width, self.height, _ = self.image.shape

    def fourierReconstruction(self) -> Tuple[np.ndarray, np.ndarray]:

        # This reconstruction exploits the mathematical similarity of the definition of projection to 2D Fourier
        # transform of absorption coefficient.

        if self.sinogram is None:
            raise RuntimeError("Generate or load a sinogram before attempting a reconstruction.")

        fft_sinogram = scipy.fft.fftshift(scipy.fft(scipy.fft.ifftshift(self.sinogram, axes=1)), axes=1)
        fourier_size = fft_sinogram.shape[0]
        fourier_radius = len(fft_sinogram[0]) / 2
        transformed_to_radial = []
        ang_i = 0
        angles = np.linspace(0, np.pi, fft_sinogram.shape[1])

        for row in fft_sinogram:  # Converting cartesian to polar coordinates for calculated Fourier transform
            for x in range(len(row)):
                temp_x, temp_y = rotate(np.array([x - fourier_radius, 0]), angles[ang_i])
                transformed_to_radial.append([temp_x, temp_y])
            ang_i += 1

        x_data = [data[0] + fourier_radius for data in transformed_to_radial]
        y_data = [data[1] + fourier_radius for data in transformed_to_radial]

        X, Y = np.meshgrid(np.arange(fourier_size), np.arange(fourier_size))  # Preparing coordinates for interpolation
        X = X.flatten()
        Y = Y.flatten()

        interpolated_radial_fft = scipy.interpolate.griddata((x_data, y_data), fft_sinogram.flatten(), (X, Y),
                                                             fill_value=0.0, method='cubic').reshape(
            (fourier_size, fourier_size))

        reconstruction = scipy.fft.fftshift(scipy.fft.ifft2(scipy.fft.ifftshift(interpolated_radial_fft)))

        reconstruction = ndimage.rotate(np.abs(reconstruction), 180)

        return interpolated_radial_fft, reconstruction

    def backProjectionReconstruction(self,angle,filter_function:Union[Callable,None,str] = 'ramp',filter_cutoff=1)\
            -> Tuple[np.ndarray,np.ndarray]:
        '''Method for reconstructing an image using the filtered back projection algorithm. angle defines
         the max. angle at which the sinogram has been recorded. Filter can be defined via filter_function
          as one of following strings:'ramp','cosine','shepp-logan' which are commonly used when dealing with FBP.
          Filter can be also defined as None which means no filtration, or a custom lambda function which is later
           applied to the standard ramp filter.'''

        sample_projection = self.sinogram[0]
        sample_tile = np.tile(sample_projection, (len(sample_projection), 1))
        reconstruction_size = ndimage.rotate(sample_tile, 45).shape
        offset = (reconstruction_size[0] - len(self.sinogram[0])) // 2
        reconstruction = np.zeros(sample_tile.shape)
        angles = np.linspace(0, angle, len(self.sinogram))

        if filter_function is not None:
            self._filter = generateFilter(reconstruction_size[0],
                                          filter_function=filter_function,cutoff=filter_cutoff)
        else:
            self._filter = 1

        for ind,(ang, values) in enumerate(zip(angles, self.sinogram)):

            extended_projection = np.zeros(reconstruction_size[0])
            extended_projection[offset:offset+len(values)] = values-min(values)
            extended_projection_freq = scipy.fft.fftshift(scipy.fft.fft(extended_projection))

            projection = np.real(scipy.fft.ifft(scipy.fft.ifftshift(self._filter*extended_projection_freq)))
            back_projection = np.tile(projection / reconstruction_size[0], (len(projection), 1))
            rot_img = ndimage.rotate(back_projection, ang, cval=np.amin(back_projection), reshape=False)
            reconstruction += cropCenterMatrix(rot_img, sample_tile.shape)

        return reconstruction,self._filter
