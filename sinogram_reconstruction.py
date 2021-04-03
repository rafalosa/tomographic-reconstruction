from scipy import fft
import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy import ndimage

'''Useful sources: http://bioeng2003.fc.ul.pt/Conference%20Files/papers/De%20Francesco,%20Fourier.pdf - English
                   http://ncbj.edu.pl/zasoby/wyklady/ld_podst_fiz_med_nukl-01/med_nukl_10_v3.pdf - Polish '''
def rotate(vector,angle:float):

    rot_matrix = [[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    return np.dot(rot_matrix,vector)

def fourierReconstruction(sinogram_data): # t, angle, data

    # This reconstruction exploits the mathematical similarity of the definition of projection to 2D Fourier transform.
    value_boundries = 60
    fft_sinogram = scipy.fft.fftshift(scipy.fft(scipy.fft.ifftshift(sinogram_data,axes=1)),axes=1)
    fourier_size = fft_sinogram.shape[0]
    plt.imshow(np.imag(fft_sinogram), vmin=-value_boundries, vmax=value_boundries,cmap='hsv')
    fourier_radius = len(fft_sinogram[0])/2
    transformed_to_radial = []
    ang_i = 0
    angs = np.linspace(0,np.pi,fft_sinogram.shape[1])

    for row in fft_sinogram:
        for x,value in enumerate(row):
            temp_x,temp_y = rotate([x-fourier_radius,0],angs[ang_i])
            transformed_to_radial.append([temp_x,temp_y])
        ang_i += 1

    fig,ax = plt.subplots()
    x_data = [data[0] + fourier_radius for data in transformed_to_radial]
    y_data = [data[1] + fourier_radius for data in transformed_to_radial]
    ax.scatter(x_data,y_data,c=np.abs(fft_sinogram.flatten()),marker='.',cmap='hsv',vmin=-value_boundries, vmax=value_boundries,s=1)

    X, Y = np.meshgrid(np.arange(fourier_size), np.arange(fourier_size)) # Preparing coordinates for interpolation
    X = X.flatten()
    Y = Y.flatten()

    interpolated_radial_fft = scipy.interpolate.griddata((x_data,y_data),fft_sinogram.flatten(),(X,Y),fill_value=0.0,method='cubic').reshape((fourier_size,fourier_size))

    fig2,ax2 = plt.subplots()
    ax2.imshow(np.abs(interpolated_radial_fft),cmap='hsv',vmin=-value_boundries, vmax=value_boundries)

    reconstruction = scipy.fft.fftshift(scipy.fft.ifft2(scipy.fft.ifftshift(interpolated_radial_fft)))

    fig3,ax3 = plt.subplots()
    rotated_recon = ndimage.rotate(np.abs(reconstruction), 180)
    ax3.imshow(rotated_recon,vmin=0.0,vmax=1.0,cmap='gray')

    plt.show()

    return 0
