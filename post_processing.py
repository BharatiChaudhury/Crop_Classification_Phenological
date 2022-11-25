from skimage.morphology import binary_dilation,binary_closing, rectangle,disk, square, binary_erosion, binary_opening
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from skimage.measure import label, regionprops
from scipy import misc,ndimage
from skimage import segmentation
from skimage.morphology import skeletonize
from skimage.morphology import disk, erosion, dilation, reconstruction


# Image Thresholding and Morphological Filtering
def postprocessing(edge):
    c=np.sqrt(edge[:,:])
    thresh = threshold_local(c,31,method='gaussian')
    #thresh = threshold_otsu(c,nbins=151)
    d=(c>thresh)

    aa=binary_opening(d,square(1))
    final=binary_closing(aa,square(2))
    indx = edge.shape

    clos=binary_closing(d,rectangle(3,5))
    skl = skeletonize(clos)
    final=dilation(skl, disk(1))

    bw=1-(final*1).astype('int')
    bw[:,0:1]=0
    bw[0:1,:]=0
    bw[indx[0]-1,:]=0
    bw[:,indx[1]-1]=0

    bwi=1-bw
    bwi[:,0:1]=0
    bwi[0:1,:]=0
    bwi[indx[0]-1,:]=0
    bwi[:,indx[1]-1]=0

    labels = label(bw,background=0)
    regions = regionprops(labels)

    # Marker locations for the watershed segmentation; Markers will - 
    # be the centroids of the different connected regions in the image
    markers = np.array([r.centroid for r in regions]).astype(np.uint16)
    marker_image = np.zeros_like(bw, dtype=np.int64)
    marker_image[markers[:, 0], markers[:, 1]]  = np.arange(len(markers))+1

    # segmentation
    distance_map = ndimage.distance_transform_edt(bwi);plt.figure()
    ndistance=1-(np.array(distance_map,dtype='int'))

    # Compute the watershed segmentation; it will over-segment the image
    filled = segmentation.watershed(ndistance, markers=marker_image,mask=1-bw)

    ## In the over-segmented image, combine touching regions
    filled_connected = label(filled ==0, background=0)+1
    # In this optional step, filter out all regions that are < 25% the size
    # of the mean region area found
    filled_regions = regionprops(filled_connected)
    mean_area = np.mean([r.area for r in filled_regions])
    std_area = np.std([r.area for r in filled_regions])

    filled_filtered = filled_connected.copy()
    for r in filled_regions:
        if r.area < .1 * mean_area:
            coords = np.array(r.coords).astype(int)
            filled_filtered[coords[:, 0], coords[:, 1]] = 0
            
    plt.figure(figsize=(8, 6))
    plt.subplot(131)
    plt.imshow(edge,cmap='gray')
    plt.title('Input Image')
    plt.subplot(132)
    plt.imshow(final,cmap='gray')
    #plt.title()
    plt.subplot(133)
    plt.imshow(filled_filtered,cmap='jet')
    #plt.title('Segmented Image')