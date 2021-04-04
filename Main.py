import sinograms
import sinogram_reconstruction
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

if __name__ == '__main__':

    im = sinograms.Scan("sample2.png")
    sinogram = im.generateSinogram()
    sinograms.showSinogram(sinogram)
    sinogram_reconstruction.fourierReconstruction(sinogram)


    #fig,ax = plt.subplots()
    #sb.heatmap(trans[:,:,3])

    plt.show()
