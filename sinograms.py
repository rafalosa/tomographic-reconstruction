import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import seaborn as sns
import multiprocessing as mp
from multiprocessing import shared_memory as sm
from multiprocessing import Pool
from functools import partial


def transformReferenceFramePoint(t:float,s:float,angle:float,x_offset:float,y_offset:float):
    # Function that transforms points between the patient's reference frame and detector's reference frame,
    # transformation consists of translation, rotation and back translation

    x = (t-x_offset)*np.cos(angle) - (s-y_offset)*np.sin(angle)
    y = (t-x_offset)*np.sin(angle) + (s-y_offset)*np.cos(angle)

    return x+x_offset,y+y_offset


def showSinogram(sinogram_data):
    fig, ax = plt.subplots()
    ax = sns.heatmap(sinogram_data, cmap=cm.gray)
    plt.show()
    return fig,ax


def calculateForDetectorPosition(lines_t_size, lines_s_size,image_shape,dtype,memory_block_name,angle):

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
                    or int(np.floor(x)) - 1 < 0 or int(np.floor(y)) - 1 < 0:

                integral += 0

            else:
                integral += np.sum(image[int(np.floor(x)) - 1, int(np.floor(y)) - 1, 0:2]) / 3

        sinogram_row[t_index] = integral

    image_mem.close()

    return sinogram_row


class Scan:

    # todo: The image should be square, so crop/pad the loaded image if it isn't square
    def __init__(self,img_path:str):
        self.image = mpimg.imread(img_path)
        self.width = self.image.shape[0]
        self.height = self.image.shape[1]

    def generateSinogram(self):
        '''The patient's reference frame is stationary, detector's reference frame
        rotates around the (0,0,0) point. Due to the way that matplotlib loads images (the image loads
        in the 1st quadrant of the coordinate system), translation transformation is first required before
        rotating the reference frame'''

        # Generating a sinogram can obviously be done better and quicker, this implementation represents the real
        # measurement process and is pretty slow.

        resolution = 301
        number_of_rays = resolution  # How many X-ray beams/detector cells
        path_resolution = 200  # How many data points for each beam/detector cell
        angle_resolution = resolution  # Angular step
        angles = np.linspace(np.pi / 2, np.pi * 3 / 2, angle_resolution + 1)
        image_memory_shared = sm.SharedMemory(create=True,size=self.image.nbytes)
        image_shared_copy = np.ndarray(self.image.shape,dtype=self.image.dtype,buffer=image_memory_shared.buf)
        image_shared_copy[:,:,:] = self.image[:,:,:]

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

        return sinogram
