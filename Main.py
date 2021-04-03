import sinograms
import sinogram_reconstruction
import matplotlib.pyplot as plt
import seaborn as sb


im = sinograms.Scan("sample1.png")
sinogram = im.generateSinogram()
sinograms.showSinogram(sinogram)
sinogram_reconstruction.fourierReconstruction(sinogram)


#fig,ax = plt.subplots()
#sb.heatmap(trans[:,:,3])

plt.show()
