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
from tools import *

"""
Useful sources: http://bioeng2003.fc.ul.pt/Conference%20Files/papers/De%20Francesco,%20Fourier.pdf - English
http://ncbj.edu.pl/zasoby/wyklady/ld_podst_fiz_med_nukl-01/med_nukl_10_v3.pdf - Polish

"""


# todo: Implement other reconstruction techniques.
# todo: Check for correct sinogram orientation in loadSinogram.
# todo: Crop/pad loaded image so it is square.
# todo: Add multiprocessing to reconstruction methods.
# todo: Adding GPU computation would be fun.
# todo: I want to try out Cython, find some usage.


class Scan:

    def __init__(self):
        self.image = None
        self.sinogram = None
        self.width = None
        self.height = None
        self._filter = None

    def generate_sinogram(self, resolution: int, path_resolution: int, processes: int = 0) -> None:

        """
        Generate a sinogram for a given image.

        :param resolution: How many Xray beams are used to generate the sinogram as well as how many angular
        positions are registered.

        :param path_resolution: Determines how many datapoints are summed in the process
        of generating a sinogram for a given Xray. The bigger the values the more detailed sinogram will be generated
        and the longer the computation time.

        :param processes: How many processes can be used to generate the sinogram. If not specified, all but 2 CPU
        threads will remain unused by this process.
        """
        # Generating a sinogram can obviously be done better and quicker by the use of linear algebra and matrix
        # multiplication. This implementation represents the "real" measurement process and is pretty slow.

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

        function = partial(evaluate_for_parallel_rays, number_of_rays + 1, path_resolution + 1,
                           self.image.shape, self.image.dtype, image_memory_shared.name)

        results = pool.imap(function, (ang for ang in angles))
        pool.close()
        pool.join()

        sinogram = np.transpose(np.hstack([row for row in results]))

        image_memory_shared.close()
        image_memory_shared.unlink()

        self.sinogram = sinogram

    def fan_beam_sinogram(self, resolution: int, path_resolution: int, cone_angle_deg: float, processes: int = 0) -> None:

        if cone_angle_deg >= 180 or cone_angle_deg <= 0:
            raise AttributeError("The angle of the fan beam must be between 0 and 180 degrees.")

        if self.image is None:
            raise RuntimeError("Image required before generating a sinogram")

        cone_angle = np.deg2rad(cone_angle_deg)

        xray_source_initial_position = (0, self.height + self.width / 2 / np.tan(cone_angle / 2))
        xray_radius = self.height * np.sqrt(2) + xray_source_initial_position[1] / 3
        reference_frame_angles = np.linspace(0, np.pi * 2, resolution)
        image_memory_shared = sm.SharedMemory(create=True, size=self.image.nbytes)
        image_shared_copy = np.ndarray(self.image.shape, dtype=self.image.dtype, buffer=image_memory_shared.buf)
        image_shared_copy[:, :, :] = self.image[:, :, :]

        if processes:
            CPU_S = processes
        else:
            CPU_S = mp.cpu_count() - 2

        pool = Pool(processes=CPU_S)
        function = partial(evaluate_for_fan_rays, xray_source_initial_position, resolution,
                           path_resolution, cone_angle, xray_radius, self.image.shape,
                           self.image.dtype, image_memory_shared.name)

        results = pool.imap(function, (ang for ang in reference_frame_angles))
        pool.close()
        pool.join()

        sinogram = np.transpose(np.hstack([row for row in results]))

        image_memory_shared.close()
        image_memory_shared.unlink()

        self.sinogram = sinogram

    def load_sinogram(self, path: str) -> None:
        sinogram = mpimg.imread(path)

        if sinogram.shape[0] != sinogram.shape[1]:
            raise AttributeError("Image has to be square")

        sinogram_sum = np.sum(sinogram, axis=2)
        self.sinogram = sinogram_sum

    def load_image(self, path: str) -> None:
        img = mpimg.imread(path)

        if img.shape[0] != img.shape[1]:
            raise AttributeError("Image has to be square")

        self.image = img
        self.width, self.height, _ = self.image.shape

    def fourier_reconstruction(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method for reconstructing an image using the fourier transformation method. This reconstruction exploits the
        mathematical similarity of the definition of projection to 2D Fourier transform of absorption coefficient.
        """

        if self.sinogram is None:
            raise RuntimeError("Generate or load a sinogram before attempting a reconstruction.")

        fft_sinogram = scipy.fft.fftshift(scipy.fft.fft(scipy.fft.ifftshift(self.sinogram, axes=1)), axes=1)
        fourier_size = fft_sinogram.shape[0]
        fourier_radius = len(fft_sinogram[0]) / 2
        transformed_to_radial = []
        ang_i = 0
        angles = np.linspace(0, np.pi, fft_sinogram.shape[1])

        for row in fft_sinogram:  # Converting cartesian to polar coordinates for calculated Fourier transform
            for x in range(len(row)):
                temp_x, temp_y = rotate_vector(np.array([x - fourier_radius, 0]), angles[ang_i])
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

    def fbp_reconstruction(self, angle, filter_function: Union[Callable, None, str] = 'ramp', filter_cutoff=1) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Method for reconstructing an image using the filtered back projection algorithm.

        :param angle: Max angle at which the sinogram has been recorded.

        :param filter_function: Specified filter function callable or an appropriate string: "ramp", "cosine",
        "shepp-logan" which are the most commonly used for FBP. None denotes lack of filtering.
        """

        if self.sinogram is None:
            raise RuntimeError("Generate or load a sinogram before attempting a reconstruction.")

        sample_projection = self.sinogram[0]
        sample_tile = np.tile(sample_projection, (len(sample_projection), 1))
        reconstruction_size = ndimage.rotate(sample_tile, 45).shape
        offset = (reconstruction_size[0] - len(self.sinogram[0])) // 2
        reconstruction = np.zeros(sample_tile.shape)
        angles = np.linspace(0, angle, len(self.sinogram))

        if filter_function is not None:
            self._filter = generate_filter(reconstruction_size[0],
                                           filter_function=filter_function, cutoff=filter_cutoff)
        else:
            self._filter = 1

        for ind, (ang, values) in enumerate(zip(angles, self.sinogram)):
            extended_projection = np.zeros(reconstruction_size[0])
            extended_projection[offset:offset + len(values)] = values - min(values)
            extended_projection_freq = scipy.fft.fftshift(scipy.fft.fft(extended_projection))

            projection = np.real(scipy.fft.ifft(scipy.fft.ifftshift(self._filter * extended_projection_freq)))
            back_projection = np.tile(projection / reconstruction_size[0], (len(projection), 1))
            rot_img = ndimage.rotate(back_projection, ang, cval=np.amin(back_projection), reshape=False)
            reconstruction += crop_matrix_center(rot_img, sample_tile.shape)

        return reconstruction, self._filter
