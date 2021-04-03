import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import seaborn as sns


def transformReferenceFramePoint(t:float,s:float,angle:float,x_offset:float,y_offset:float):
    # Function that transforms points between the patient's reference frame and detector's reference frame,
    # transformation consists of translation, rotation and back translation

    x = (t-x_offset)*np.cos(angle) - (s-y_offset)*np.sin(angle)
    y = (t-x_offset)*np.sin(angle) + (s-y_offset)*np.cos(angle)

    return x+x_offset,y+y_offset


def showSinogram(sinogram_data):
    fig, ax = plt.subplots()
    ax = sns.heatmap(sinogram_data[:, :, 2], cmap=cm.gray)
    plt.show()
    return fig,ax


class Scan:

    def __init__(self,img_path:str):  # todo: The image should be square, so crop/pad the loaded image if it isn't square
        self.image = mpimg.imread(img_path)
        self.width = self.image.shape[0]
        self.height = self.image.shape[1]

    def coordinatesToPixelValue(self,x:float, y:float):
        # Converting coordinates to corresponding pixel value

        # Due to the nature of the reference frames transformations some points end up outside the original image
        # if that happens, set the pixel to 0 (empty space)
        if int(np.floor(x))-1 >= self.width or int(np.floor(y))-1 >= self.height\
                or int(np.floor(x))-1 < 0 or int(np.floor(y))-1 < 0:

            return .0

        else:
            return np.sum(self.image[int(np.floor(x))-1,int(np.floor(y))-1,0:2])/3  # Fourth channel ignored

    #todo: Parallelize the process of generating a sinogram
    def generateSinogram(self):  # The patient's reference frame is stationary, detector's reference frame
        # rotates around the (0,0,0) point. Due to the way that matplotlib loads images (the image loads
        # in the 1st quadrant of the coordinate system), translation transformation is first required before
        # rotating the reference frame

        # Generating a sinogram can obviously be done better and quicker, this implementation represents the real
        # measurement process and is pretty slow.

        number_of_rays = 50  # How many X-ray beams/detector cells
        path_resolution = 50  # How many data points for each beam/detector cell
        angle_resolution = 180  # Angular step
        angles = np.linspace(np.pi / 2, np.pi * 3 / 2, angle_resolution + 1)
        lines_t = np.linspace(0,self.width,number_of_rays+1)
        lines_s = np.linspace(0,self.height,path_resolution+1)
        sinogram_data = np.zeros([number_of_rays+1,angle_resolution+1,3])
        for angle_index,angle in enumerate(angles):
            for t_index,t in enumerate(lines_t):
                integral = 0
                for s in lines_s: # Computing the Radon transform for each generated X-ray
                    x,y = transformReferenceFramePoint(t,s,angle,self.width/2,self.height/2) # Transforming between reference frames
                    integral += self.coordinatesToPixelValue(x,y)
                sinogram_data[t_index,angle_index] = [t,angle,integral]
        return sinogram_data
