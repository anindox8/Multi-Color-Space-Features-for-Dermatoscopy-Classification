### Libraries
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import local_binary_pattern







### Color Features
def color_moments(image, channel=3):
    if (channel==3):        
        mean_0 = np.mean(image[:,:,0])
        mean_1 = np.mean(image[:,:,1])
        mean_2 = np.mean(image[:,:,2])
        std_0  = np.std(image[:,:,0])
        std_1  = np.std(image[:,:,1])
        std_2  = np.std(image[:,:,2])
        skew_0 = skew(image[:,:,0].reshape(-1))
        skew_1 = skew(image[:,:,1].reshape(-1))
        skew_2 = skew(image[:,:,2].reshape(-1))
        kurt_0 = kurtosis(image[:,:,0].reshape(-1))
        kurt_1 = kurtosis(image[:,:,1].reshape(-1))
        kurt_2 = kurtosis(image[:,:,2].reshape(-1))
        return mean_0, std_0, skew_0, kurt_0, mean_1, std_1, skew_1, kurt_1, mean_2, std_2, skew_2, kurt_2
    
    elif (channel==1):        
        mean_0 = np.mean(image)
        std_0  = np.std(image)
        skew_0 = skew(image.reshape(-1))
        return mean_0, std_0, skew_0
    
    else:
        assert False, "ERROR: The function supports only 1 or 3-channel image formats."
    


    
def GLCM(image, channel=3, bit_depth=8):
    if (channel==3):  
        GLCM_0  = greycomatrix(image[:,:,0],  [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=2**bit_depth)
        GLCM_1  = greycomatrix(image[:,:,1],  [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=2**bit_depth)
        GLCM_2  = greycomatrix(image[:,:,2],  [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=2**bit_depth)
        contrast_0  = greycoprops(GLCM_0,  'contrast').mean()
        contrast_1  = greycoprops(GLCM_1,  'contrast').mean()
        contrast_2  = greycoprops(GLCM_2,  'contrast').mean()
        dissim_0    = greycoprops(GLCM_0,  'dissimilarity').mean()
        dissim_1    = greycoprops(GLCM_1,  'dissimilarity').mean()
        dissim_2    = greycoprops(GLCM_2,  'dissimilarity').mean()
        correl_0    = greycoprops(GLCM_0,  'correlation').mean()
        correl_1    = greycoprops(GLCM_1,  'correlation').mean()
        correl_2    = greycoprops(GLCM_2,  'correlation').mean()
        homo_0      = greycoprops(GLCM_0,  'homogeneity').mean()
        homo_1      = greycoprops(GLCM_1,  'homogeneity').mean()
        homo_2      = greycoprops(GLCM_2,  'homogeneity').mean()
        return [ contrast_0, dissim_0, correl_0, homo_0, contrast_1, dissim_1,
                 correl_1, homo_1, contrast_2, dissim_2, correl_2, homo_2 ]
    
    elif (channel==1):
        GLCM_0  = greycomatrix(image,  [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=2**bit_depth)
        contrast_0  = greycoprops(GLCM_0,  'contrast').mean()
        dissim_0    = greycoprops(GLCM_0,  'dissimilarity').mean()
        correl_0    = greycoprops(GLCM_0,  'correlation').mean()
        homo_0      = greycoprops(GLCM_0,  'homogeneity').mean()
        return [ contrast_0, dissim_0, correl_0, homo_0 ]
    
    else:
        assert False, "ERROR: The function supports only 1 or 3-channel image formats."




def melanoma_color_markers(image,mask):
    CM_black      = np.count_nonzero(((image[:,:,0].astype(float)/255)<0.20)
                                    &((image[:,:,1].astype(float)/255)<0.20)
                                    &((image[:,:,2].astype(float)/255)<0.20))*(100/np.sum(mask))
    CM_red        = np.count_nonzero(((image[:,:,0].astype(float)/255)>0.80)
                                    &((image[:,:,1].astype(float)/255)<0.20)
                                    &((image[:,:,2].astype(float)/255)<0.20))*(100/np.sum(mask))
    CM_bluegray   = np.count_nonzero(((image[:,:,0].astype(float)/255)<0.20)
                                    &((image[:,:,1].astype(float)/255)<0.72)
                                    &((image[:,:,1].astype(float)/255)>0.30)
                                    &((image[:,:,2].astype(float)/255)<0.74)
                                    &((image[:,:,2].astype(float)/255)>0.34))*(100/np.sum(mask))
    CM_white      = np.count_nonzero(((image[:,:,0].astype(float)/255)>0.80)
                                    &((image[:,:,1].astype(float)/255)>0.80)
                                    &((image[:,:,2].astype(float)/255)>0.80))*(100/np.sum(mask))
    CM_lightbrown = np.count_nonzero(((image[:,:,0].astype(float)/255)<1.00)
                                    &((image[:,:,0].astype(float)/255)>0.60)
                                    &((image[:,:,1].astype(float)/255)<0.72)
                                    &((image[:,:,1].astype(float)/255)>0.32)
                                    &((image[:,:,2].astype(float)/255)<0.45)
                                    &((image[:,:,2].astype(float)/255)>0.05))*(100/np.sum(mask))
    CM_darkbrown  = np.count_nonzero(((image[:,:,0].astype(float)/255)<0.60)
                                    &((image[:,:,0].astype(float)/255)>0.20)
                                    &((image[:,:,1].astype(float)/255)<0.46)
                                    &((image[:,:,1].astype(float)/255)>0.06)
                                    &((image[:,:,2].astype(float)/255)<0.33))*(100/np.sum(mask))
    return CM_black, CM_red, CM_bluegray, CM_white, CM_lightbrown, CM_darkbrown




def LBP(image, channel=3, P=8, R=2, bins=10):
    if (channel==3):
        lbp       = local_binary_pattern(image[:,:,0], P=P, R=R, method="uniform")
        lbp_0, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
        lbp       = local_binary_pattern(image[:,:,1], P=P, R=R, method="uniform")
        lbp_1, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
        lbp       = local_binary_pattern(image[:,:,2], P=P, R=R, method="uniform")
        lbp_2, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
        return lbp_0, lbp_1, lbp_2
    
    elif (channel==1):
        lbp       = local_binary_pattern(image, P=P, R=R, method="uniform")
        lbp_0, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
        return lbp_0
    else:
        assert False, "ERROR: The function supports only 1 or 3-channel image formats."

        

def entropyplus(image):   
    histogram         = np.histogram(image, bins=2**8, range=(0,(2**8)-1), density=True)
    histogram_prob    = histogram[0]/sum(histogram[0])    
    single_entropy    = np.zeros((len(histogram_prob)), dtype = float)
    for i in range(len(histogram_prob)):
        if(histogram_prob[i] == 0):
            single_entropy[i] = 0;
        else:
            single_entropy[i] = histogram_prob[i]*np.log2(histogram_prob[i]);
    smoothness   = 1- 1/(1 + np.var(image/2**8))            
    uniformity   = sum(histogram_prob**2);        
    entropy      = -(histogram_prob*single_entropy).sum()
    return smoothness, uniformity, entropy



def entropyplus_3(image):
    smoothness_0, uniformity_0, entropy_0 = entropyplus(image[:,:,0])
    smoothness_1, uniformity_1, entropy_1 = entropyplus(image[:,:,1])
    smoothness_2, uniformity_2, entropy_2 = entropyplus(image[:,:,2])
    return [ smoothness_0, uniformity_0, entropy_0, smoothness_1, uniformity_1, 
             entropy_1, smoothness_2, uniformity_2, entropy_2 ]