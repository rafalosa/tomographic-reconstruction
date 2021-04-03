from scipy import fft
import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

'''Useful sources: http://bioeng2003.fc.ul.pt/Conference%20Files/papers/De%20Francesco,%20Fourier.pdf - English
                   http://ncbj.edu.pl/zasoby/wyklady/ld_podst_fiz_med_nukl-01/med_nukl_10_v3.pdf - Polish '''
def rotate(vector,angle:float):

    rot_matrix = [[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    return np.dot(rot_matrix,vector)

def fourierReconstruction(sinogram_data): # t, angle, data

    # This reconstruction exploits the mathematical similarity of the definition of projection to 2D Fourier transform.
    value_boundries = 100
    angles = sinogram_data[0,:,1]
    detector_domain = sinogram_data[:,0,0]
    fft_sinogram = scipy.fft.fftshift(scipy.fft(sinogram_data[:, :, 2]),axes=1)
    print(len(fft_sinogram))
    plt.imshow(np.abs(fft_sinogram), vmin=-value_boundries, vmax=value_boundries,cmap='cubehelix')
    fourier_radius = len(fft_sinogram[0])/2
    transformed_to_radial = []
    ang_i = 0

    for row in fft_sinogram:
        for x,value in enumerate(row):
            temp_x,temp_y = rotate([x-fourier_radius,0],angles[ang_i])
            transformed_to_radial.append([temp_x,temp_y])
        ang_i += 1

    fig,ax = plt.subplots()
    x_data = [data[0] + fourier_radius for data in transformed_to_radial]
    y_data = [data[1] + fourier_radius for data in transformed_to_radial]
    ax.scatter(x_data,y_data,c=np.abs(fft_sinogram.flatten()),marker='.',cmap='cubehelix',vmin=-value_boundries, vmax=value_boundries)

    X, Y = np.meshgrid(np.arange(199), np.arange(199))
    X = X.flatten()
    Y = Y.flatten()

    interpolated_radial_fft = scipy.interpolate.griddata((x_data,y_data),fft_sinogram.flatten(),(X,Y),fill_value=0.0).reshape((199,199))

    fig2,ax2 = plt.subplots()
    ax2.imshow(np.abs(interpolated_radial_fft),cmap='cubehelix',vmin=-value_boundries, vmax=value_boundries)

    reconstruction = scipy.fft.fftshift(scipy.fft.ifft2(scipy.fft.ifftshift(interpolated_radial_fft,)))

    fig3,ax3 = plt.subplots()
    ax3.imshow(np.real(reconstruction),vmin=0.0,vmax=1.0,cmap='gray')

    plt.show()

    ''' 
   for angle_index,angle in enumerate(angles):

        projection_transform = fft.fft(sinogram_data[:,angle_index,2])
        transform_domain = fft.fftfreq(detector_domain.size,detector_step)
        for ksi_index,ksi in enumerate(transform_domain):

            transform_map[ksi_index,angle_index] = ksi*np.cos(angle),ksi*np.sin(angle),angle,np.imag(projection_transform[ksi_index])

    inverse_transform = fft.ifft2(transform_map[:,:,3])
    
    for angle_index, angle in enumerate(angles):
             for ksi_index, ksi in enumerate(transform_domain):
                 absorption_map[ksi_index, angle_index] = ksi*np.cos(angle),ksi*np.sin(angle), angle, np.imag(inverse_transform[angle_index][ksi_index])'''

    return 0
