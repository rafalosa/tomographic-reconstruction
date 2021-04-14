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
import matplotlib.pyplot as plt
from typing import Union

"""
Useful sources: http://bioeng2003.fc.ul.pt/Conference%20Files/papers/De%20Francesco,%20Fourier.pdf - English
http://ncbj.edu.pl/zasoby/wyklady/ld_podst_fiz_med_nukl-01/med_nukl_10_v3.pdf - Polish

"""


def transformReferenceFramePoint(t,s,angle:float,x_offset:float,y_offset:float,
                                 back_translation:bool = True) -> tuple:
    '''Function that transforms points between the patient's reference frame and detector's reference frame,
    transformation consists of translation, rotation and back translation'''

    """The patient's reference frame is stationary, detector's reference frame
    rotates around the (0,0,0) point. Due to the way that matplotlib loads images (the image loads
    in the 1st quadrant of the coordinate system), translation transformation is first required before
    rotating the reference frame"""

    x = (t-x_offset)*np.cos(angle) - (s-y_offset)*np.sin(angle)
    y = (t-x_offset)*np.sin(angle) + (s-y_offset)*np.cos(angle)

    if back_translation:
        return x+x_offset,y+y_offset
    else:
        return x,y


def calculateForDetectorPosition(lines_t_size:int, lines_s_size:int,image_shape:tuple,
                                 dtype:type,memory_block_name:str,angle:float):

    width, height, _ = image_shape
    lines_t = np.linspace(0, width, lines_t_size)
    lines_s = np.linspace(0, height, lines_s_size)
    sinogram_row = np.zeros([len(lines_t),1])
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


def generateDomain(angle:float, radius:float, source_coords:tuple,points_num:int):

    if angle <= np.pi/2:
        start_point = -radius * np.cos(angle)
        domain = np.linspace(start_point,source_coords[0],points_num)
        return domain
    else:
        end_point = radius * np.cos(np.pi - angle)
        domain = np.linspace(source_coords[0],end_point, points_num)
        return domain


class Scan:

    # todo: The image should be square, so crop/pad the loaded image if it isn't square
    def __init__(self):
        self.image = None
        self.sinogram = None
        self.width = None
        self.height = None

    def generateSinogram(self,resolution:int,path_resolution:int,processes:int = 0):

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
        image_memory_shared = sm.SharedMemory(create=True,size=self.image.nbytes)
        image_shared_copy = np.ndarray(self.image.shape,dtype=self.image.dtype,buffer=image_memory_shared.buf)
        image_shared_copy[:,:,:] = self.image[:,:,:]

        if processes:
            CPU_S = processes
        else:
            CPU_S = mp.cpu_count() - 2

        pool = Pool(processes=CPU_S)
        function = partial(calculateForDetectorPosition,number_of_rays+1,path_resolution+1,
                           self.image.shape,self.image.dtype,image_memory_shared.name)

        results = pool.imap(function,(ang for ang in angles))
        pool.close()
        pool.join()

        sinogram = np.transpose(np.hstack([row for row in results]))

        image_memory_shared.close()
        image_memory_shared.unlink()

        self.sinogram = sinogram

    def fanBeamSinogram(self,resolution:int,path_resolution:int,cone_angle_deg:float):
        # Probably implement this method to generateSinogram with an additional bool parameter
        # todo: implement multiprocessing

        cone_angle = np.deg2rad(cone_angle_deg)

        xray_source_initial_position = (0, self.height + self.width/2/np.tan(cone_angle/2))
        xray_radius = self.height * np.sqrt(2) + xray_source_initial_position[1]/3
        angles_rays = np.linspace((np.pi-cone_angle)/2,np.pi-(np.pi-cone_angle)/2,resolution)
        reference_frame_angles = np.linspace(0,np.pi,resolution)
        domains = []
        rays = []
        rows = []

        for ray_ang in angles_rays:
            domain = generateDomain(ray_ang, xray_radius, xray_source_initial_position, path_resolution)
            domains.append(domain)
            rays.append(np.tan(ray_ang) * domain + xray_source_initial_position[1])

        for ref_angle in reference_frame_angles:

            xray_source_position = rotate(xray_source_initial_position,ref_angle)
            sinogram_row = np.zeros([resolution, 1])

            for row_index,(domain,ray) in enumerate(zip(domains,rays)):
                T,S = transformReferenceFramePoint(domain,ray,ref_angle,0,
                                                   xray_source_initial_position[1]+self.height/2,back_translation=False)
                T += xray_source_position[0] + self.width/2
                S += xray_source_position[1] + self.height/2
                integral = 0
                for t,s in zip(T,S):
                    if int(np.floor(t)) - 1 >= self.width or int(np.floor(s)) - 1 >= self.height \
                            or int(np.floor(t)) - 1 < 0 or int(np.floor(s)) - 1 < 0:
                        # Mapping coordinates to pixel value

                        integral += 0

                    else:
                        integral += np.sum(self.image[int(np.floor(t)) - 1, int(np.floor(s)) - 1, 0:2]) / 3

                sinogram_row[row_index] = integral
                rows.append(sinogram_row[row_index])

        self.sinogram = np.transpose(np.reshape(rows,(resolution,resolution)))

    def loadSinogram(self,path:str): # todo: detect if the sinogram has a correct orientation, if not rotate it
        sinogram = mpimg.imread(path)
        sinogram_sum = np.sum(sinogram,axis=2)
        self.sinogram = sinogram_sum

    def loadImage(self,path:str):
        self.image = mpimg.imread(path)
        self.width,self.height,_ = self.image.shape

    def fourierReconstruction(self) -> tuple:

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


def rotate(vector:Union[np.ndarray,tuple],angle:float) -> np.ndarray:
    '''Function rotates a given vector counterclockwise by a given angle in radians (vector,angle)'''

    rot_matrix = [[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    return np.dot(rot_matrix,vector)





