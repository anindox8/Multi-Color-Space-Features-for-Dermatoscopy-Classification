### Libraries
import numpy as np
import cv2
from skimage import filters as skifilters
from scipy import ndimage
import skimage
from skimage import filters


### Preprocessing Functions
def color_constant(img):
    # Extract Color Channels
    img_R = img[:,:,0]
    img_G = img[:,:,1]
    img_B = img[:,:,2]    
    
    # Calculate Channel Averages
    avg_R = np.mean(img_R)
    avg_G = np.mean(img_G)
    avg_B = np.mean(img_B)
    avg_all = np.mean(img)
    
    # Calculate Scaling Factor for White-Balance
    scale_R = (avg_all / avg_R)
    scale_G = (avg_all / avg_G)
    scale_B = (avg_all / avg_B)
    
    # Transform to White-Balance
    img_new = np.zeros(img.shape)
    img_new[:,:,0] = scale_R * img_R  
    img_new[:,:,1] = scale_G * img_G 
    img_new[:,:,2] = scale_B * img_B  
    
    # Normalize Images
    max_intensity = np.max(np.max(np.max(img_new)))
    min_intensity = np.min(np.min(np.min(img_new)))
    
    img_normalized = (((img_new - min_intensity) / (max_intensity - min_intensity))*255).astype(np.uint8)

    # Illuminant Profile (Gray World Color Constancy) 
    illuminance = [avg_R, avg_G, avg_B]
    
    return img_normalized, illuminance




def white_balance(img):
    # White Balance in LAB Color Space
    result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)                                             
    avg_a  = np.average(result[:, :, 1])
    avg_b  = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)      
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)      
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    return result




def clahe_LAB(img,clip=0.9,tile=1):
    # Contrast Limited Adaptive Histogram Equalization in LAB Color Space
    lab        = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)                          
    clahe      = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile))     
    lab[:,:,0] = clahe.apply(lab[:,:,0])                                       
    output     = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)                          
    return output




def occlusion_removal(img,threshold=20,SE_radius=13,minArea=50000):
    # Remove Dark Hair Occlusions in Dermatoscopic Images via LUV Color Space
    luv       = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)

    # Morphological Closing via Spherical SE
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(SE_radius,SE_radius))
    closing   = cv2.morphologyEx(luv, cv2.MORPH_CLOSE, kernel)

    # Generate Masks via Hysteresis Thresholding Difference Image in L Channel
    diffc      = closing[:,:,0]-luv[:,:,0]
    maskc      = (skifilters.apply_hysteresis_threshold(diffc,threshold,80)).astype(np.uint8)*255
    
    # Remove Side Components
    label_im, nb_labels = ndimage.label(maskc)
    sizes               = ndimage.sum(maskc, label_im, range(nb_labels + 1))
    temp_mask           = sizes > minArea
    maskc               = (temp_mask[label_im]*255).astype(np.uint8)
  
    mask_3dc   = maskc[:,:,None] * np.ones(3,dtype=np.uint8)[None, None, :]
    basec      = cv2.bitwise_not(maskc)
    base_3dc   = basec[:,:,None] * np.ones(3,dtype=np.uint8)[None, None, :]

    # Restitch Preprocessed Image
    preimagec  = ((base_3dc/255)*luv).astype(np.uint8)
    postimagec = ((mask_3dc/255)*closing).astype(np.uint8)
    fullc      = preimagec + postimagec
    outputc    = cv2.cvtColor(fullc, cv2.COLOR_Luv2RGB)

    return outputc, maskc



def maxrgb_filter(image):
    (B, G, R) = cv2.split(image)
    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0
    return cv2.merge([B, G, R])




def grey_edge(image, njet=0, mink_norm=1, sigma=1):
    """
    Estimates the light source of an input_image as proposed in:
    J. van de Weijer, Th. Gevers, A. Gijsenij
    "Edge-Based Color Constancy"
    IEEE Trans. Image Processing, accepted 2007.
    Depending on the parameters the estimation is equal to Grey-World, Max-RGB, general Grey-World,
    Shades-of-Grey or Grey-Edge algorithm.
    :param image: rgb input image (NxMx3)
    :param njet: the order of differentiation (range from 0-2)
    :param mink_norm: minkowski norm used (if mink_norm==-1 then the max
           operation is applied which is equal to minkowski_norm=infinity).
    :param sigma: sigma used for gaussian pre-processing of input image
    :return: illuminant color estimation
    :raise: ValueError
    
    Ref: https://github.com/MinaSGorgy/Color-Constancy
    """
    gauss_image = filters.gaussian(image, sigma=sigma, multichannel=True)

    if njet == 0:
        deriv_image = [gauss_image[:, :, channel] for channel in range(3)]
    else:   
        if njet == 1:
            deriv_filter = filters.sobel
        elif njet == 2:
            deriv_filter = filters.laplace
        else:
            raise ValueError("njet should be in range[0-2]! Given value is: " + str(njet))     
        deriv_image = [np.abs(deriv_filter(gauss_image[:, :, channel])) for channel in range(3)]

    for channel in range(3):
        deriv_image[channel][image[:, :, channel] >= 255] = 0.

    if mink_norm == -1:  
        estimating_func = np.max 
    else:
        estimating_func = lambda x: np.power(np.sum(np.power(x, mink_norm)), 1 / mink_norm)
    illum = [estimating_func(channel) for channel in deriv_image]
    som   = np.sqrt(np.sum(np.power(illum, 2)))
    illum = np.divide(illum, som)

    return illum


def correct_image(image, illum):
    """
    Corrects image colors by performing diagonal transformation according to 
    given estimated illumination of the image.
    
    :param image: rgb input image (NxMx3)
    :param illum: estimated illumination of the image
    :return: corrected image
    
    Ref: https://github.com/MinaSGorgy/Color-Constancy
    """
    correcting_illum = illum * np.sqrt(3)
    corrected_image = image / 255.
    for channel in range(3):
        corrected_image[:, :, channel] /= correcting_illum[channel]
    return np.clip(corrected_image, 0., 1.)