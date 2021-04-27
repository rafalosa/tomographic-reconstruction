import sinograms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

if __name__ == '__main__':

    img_path = "images/sinograms/sample6.png"
    image = mpimg.imread(img_path)
    scan_obj = sinograms.Scan()
    #scan_obj.loadImage(img_path)
    scan_obj.loadSinogram(img_path)
    #scan_obj.generateSinogram(250,250,8)
    #scan_obj.fanBeamSinogram(301,200,60,4)
    #scan_obj.generateSinogram(301,200,8)
    #fourier,recon = scan_obj.fourierReconstruction()
    scan_obj.backProjectionReconstruction(180)
    """fig,axs = plt.subplots(2,3)
    axs[0,0].imshow(image)
    axs[0,0].set_title('Original image')
    axs[0,1].imshow(scan_obj.sinogram,cmap='gray')
    axs[0, 1].set_title('Sinogram of the image')
    axs[1,0].imshow(np.abs(fourier),cmap='hsv',vmin=-60,vmax=60)
    axs[1, 0].set_title('FFT of the sinogram converted\nto polar coordinates (abs)')
    axs[1,1].imshow(recon,cmap='gray')
    axs[1, 1].set_title('Reconstructed image FR')
    #axs[1, 2].imshow(scan_obj.backProjectionReconstruction(), cmap='gray')
    #axs[1, 2].set_title('Reconstructed image BP')
    fig.tight_layout(pad=1.0)"""
    plt.show()

