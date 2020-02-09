import os 
import numpy as np
from datetime import datetime
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from skimage.segmentation import inverse_gaussian_gradient, circle_level_set, morphological_geodesic_active_contour
from skimage.filters import unsharp_mask
from skimage.transform import resize
from skimage.util import crop
from skimage.exposure import equalize_hist
from scipy.spatial.distance import directed_hausdorff

coords = []

# Function adding coordinates of clicked points to global 'coords' tuple.

def onclick(event):
    ix, iy = event.xdata, event.ydata
    global coords
    coords.append((iy,ix))
    ax = plt.gca()
    ax.plot(ix, iy, '.r-')
    plt.draw()
	
    if len(coords) == 2:
        plt.close()
        return
    
# Function implementing anisotropic diffusion filter.
# Authors specified at the bottom of the script.

def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1.,1.), option=1):
    
    img = img.astype('float32')
    imgout = img.copy()
 
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()
 
    for ii in range(niter):
 
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)
 
        if option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
 
        E = gE*deltaE
        S = gS*deltaS
 
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]
 
        imgout += gamma*(NS+EW)
                
    return imgout

# Function converting an image with a contour into binary mask of the contour.
# Arguments: filename - a file containing an image with a contour, 
#            mask_shape - final shape of a mask
# Returns: binary_contour - Binary mask, shape specified in mask_shape

def contour_to_binary_mask(filename, mask_shape):
    
    current_path = os.path.abspath(os.path.dirname(__file__))
    image_path = os.path.join(current_path, filename)
    image = io.imread(image_path)
    image_gray = color.rgb2gray(image)
    
    # Deleting white spaces around certain contour images
    binary_image = image_gray < 0.996
    y_crop = np.nonzero(binary_image)[0][0]
    x_crop = np.nonzero(binary_image)[1][0]
    cropped_image = crop(image, ((y_crop, y_crop), (x_crop, x_crop), (0,0)))
    
    cropped_image = resize(cropped_image, mask_shape, preserve_range = True).astype(np.uint8)
    
    cropped_gray = color.rgb2gray(cropped_image)
    edge_mask = np.zeros(cropped_gray.shape)
    size_y = np.shape(cropped_image)[0]
    size_x = np.shape(cropped_image)[1]
    
    # A condition for thresholding found empirically
    for y in range(size_y):
        for x in range(size_x):
            if abs(int(cropped_image[y,x][0]) - int(cropped_image[y,x][1])) > 10.0:
                edge_mask[y,x] = 1
                
    binary_contour = nd.morphology.binary_fill_holes(edge_mask).astype(int)
    
    return binary_contour

# Function calculating Dice coefficient between two binary masks

def dice(binary_mask1, binary_mask2):
    
    intersection = np.sum(np.logical_and(binary_mask1, binary_mask2))
    sum_1 = np.sum(binary_mask1)
    sum_2 = np.sum(binary_mask2)
    
    dice_coeff = (2*intersection)/(sum_1 + sum_2)
    
    return dice_coeff

# Function calculating Hausdorff distace between two binary masks

def hausdorff(binary_mask1, binary_mask2):
    return max(directed_hausdorff(binary_mask1, binary_mask2)[0], 
               directed_hausdorff(binary_mask2, binary_mask1)[0])

# Mean calculation

def mean(input_list):
    n = len(input_list)
    return sum(input_list)/n

# Standard deviation calculation

def stddev(input_list):
    xm = mean(input_list)
    xs = sum((x - xm)**2 for x in input_list)
    return xs/len(input_list)

# Main function

def main():
    
    current_path = os.path.abspath(os.path.dirname(__file__))
    dice_results = []
    hausdorff_results = []
    
    # Final report text file initialization
    
    summary_file = open('contour_report.txt', 'w+')
    summary_file.write('Script run: {a}\n'.format(a = datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    for image_number in range(1,110):
        
        image_path = os.path.join(current_path, 'input_images','{}_no_contour.jpg'.format(image_number))
        image = io.imread(image_path)
        image = color.rgb2gray(image)

        # Displaying the image to get the clicks coordinates
        
        fig = plt.figure()
        plt.imshow(image, cmap = 'gray')
        plt.axis('Off')

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        # Calculating the radius, used to initialize the level set algorithm
        
        radius = np.sqrt(np.square(coords[1][0]-coords[0][0])+
                         np.square(coords[1][1]-coords[0][1]))

        # Main part of the script - image processing and creating the contour:
        # 1. Histogram equalization
        # 2. Gaussian filtration
        # 3. Anisotropic diffusion filtration
        # 4. Inverse Gaussian Gradient
        # 5. Unsharp Masking
        # 6. Morphological snakes
        # Hyperparameter optimization were performed using grid search for every method. 
        
        equalized_image = equalize_hist(image)
        filtered_image = nd.gaussian_filter(equalized_image, sigma = 5)
        diff_image = anisodiff(filtered_image, niter = 50)
        gimage = inverse_gaussian_gradient(diff_image, alpha = 50)
        sharpened_image = unsharp_mask(gimage, radius = 100, amount = 1.0)
        init_level_set = circle_level_set(gimage.shape, coords[0], radius)
        level_set = morphological_geodesic_active_contour(sharpened_image, 250, 
                                                          init_level_set, smoothing = 4, 
                                                          balloon = 1)
        level_set = nd.morphology.binary_fill_holes(level_set).astype(int)

        contour_access = os.path.join('input_images', '{}_contour.jpg'.format(image_number))
        given_contour = contour_to_binary_mask(contour_access, level_set.shape)
        dice_result = dice(given_contour, level_set)
        hausdorff_result = hausdorff(given_contour, level_set)
        
        print('Dice =', dice_result)
        print('Hausdorff =', hausdorff_result)
        
        # Updating the report with new coefficients.
        
        summary_file.write('\n{a}:\nDice: {b}\nHausdorff: {c}\n'.format(a = image_number, 
                                                                        b = format(dice_result, '.2f'), 
                                                                        c = format(hausdorff_result,'.2f')))
        
        dice_results.append(dice_result)
        hausdorff_results.append(hausdorff_result)
                           
        # Displaying the result
        
        plt.figure()
        plt.imshow(image, cmap="gray")
        plt.axis('Off')
        plt.contour(level_set, [0.5], colors='r')
        plt.show()
                           
    # Calculating and appending mean and standard deviation of the test to the end of the report
    
    summary_file.write('\n\nDICE:\nMEAN: {d}\nSTD DEV: {e}'.format(d = format(mean(dice_results), '.2f'), 
                       e = format(stddev(dice_results), '.2f')))
    summary_file.write('\n\nHAUSDORFF:\nMEAN: {d}\nSTD DEV: {e}'.format(d = format(mean(hausdorff_results), '.2f'),
                       e = format(stddev(hausdorff_results), '.2f')))
    summary_file.close()
    
    
if __name__ == "__main__":
    main()

# -------------------------------------------------------------------   
# anisodiff function authors::

# Reference:
#       P. Perona and J. Malik.
#       Scale-space and edge detection using ansotropic diffusion.
#       IEEE Transactions on Pattern Analysis and Machine Intelligence,
#       12(7):629-639, July 1990.
 
#       Original MATLAB code by Peter Kovesi  
#       School of Computer Science & Software Engineering
#       The University of Western Australia
#       pk @ csse uwa edu au
#       <http://www.csse.uwa.edu.au>
 
#       Translated to Python and optimised by Alistair Muldal
#       Department of Pharmacology
#       University of Oxford
#       <alistair.muldal@pharm.ox.ac.uk>
 
#       June 2000  original version.      
#       March 2002 corrected diffusion eqn No 2.
#       July 2012 translated to Python
#       """

