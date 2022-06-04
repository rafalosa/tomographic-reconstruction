from tomography import Scan
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

if __name__ == '__main__':
    img_path = "images/objects/sample5.png"
    image = mpimg.imread(img_path)
    scan_obj = Scan()
    scan_obj.load_image(img_path)
    scan_obj.generate_sinogram(201, 200)  # <- adjust the number of available processes if the program crashes.
    fourier, fourier_recon = scan_obj.fourier_reconstruction()
    fbp_recon, filt = scan_obj.fbp_reconstruction(180, filter_function='cosine')
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Original image')
    axs[1, 0].imshow(scan_obj.sinogram, cmap='gray')
    axs[1, 0].set_title('Sinogram of the image')
    axs[0, 1].imshow(np.abs(fourier), cmap='hsv', vmin=-60, vmax=60)
    axs[0, 1].set_title('FFT of the sinogram converted\nto polar coordinates (abs)')
    axs[1, 1].imshow(fourier_recon, cmap='gray')
    axs[1, 1].set_title('Reconstructed image FR')
    axs[1, 2].imshow(fbp_recon, cmap='gray')
    axs[1, 2].set_title('Reconstructed image BP')
    axs[0, 2].plot(filt)
    axs[0, 2].set_title("Filter function in frequency domain")
    fig.tight_layout(pad=1.0)
    plt.show()
