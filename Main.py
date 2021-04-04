import sinograms
import sinogram_reconstruction
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

if __name__ == '__main__':

    im = sinograms.Scan("images/censor.png")
    sinogram = im.generateSinogram()
    _ = sinograms.showSinogram(sinogram)
    sinogram_reconstruction.fourierReconstruction(sinogram)
    plt.show()
