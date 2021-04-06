import sinograms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

if __name__ == '__main__':

    img_path = "images/sample5.png"
    image = mpimg.imread(img_path)
    scan_obj = sinograms.Scan()
    scan_obj.loadSinogram(img_path)
    #scan_obj.generateSinogram(101,100)
    fourier,recon = scan_obj.fourierReconstruction()

    fig,axs = plt.subplots(2,2)
    axs[0,0].imshow(image)
    axs[0,0].set_title('Original image')
    axs[0,1].imshow(scan_obj.sinogram,cmap='gray')
    axs[0, 1].set_title('Sinogram of the image')
    axs[1,0].imshow(np.abs(fourier),cmap='hsv',vmin=-60,vmax=60)
    axs[1, 0].set_title('FFT of the sinogram converted\nto polar coordinates (abs)')
    axs[1,1].imshow(recon,cmap='gray')
    axs[1, 1].set_title('Reconstructed image')
    fig.tight_layout(pad=1.0)

    plt.show()