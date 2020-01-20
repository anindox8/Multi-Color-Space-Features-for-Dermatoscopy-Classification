import tqdm
import time
import numpy as np
import cv2
from sklearn.metrics import (roc_curve, auc, accuracy_score, f1_score, precision_score, 
                             recall_score, classification_report, confusion_matrix)
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold
from dataio import *
from preprocess import *
from segment import *
from colorfeatures import *
from classify import *





### Feature Extraction
def extract_features(image,mask=None):    
    # Color Spaces: I/O ------------------------------------------------------------------------------------------------------------------------------------------------------
    img_RGB               = image
    img_GL                = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
    img_HSV               = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    img_LAB               = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Lab)
    img_YCrCb             = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)
    img_luv               = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Luv)
    circa_mask            = create_circular_mask(image.shape[0], image.shape[1], radius = 300).astype(bool)
    
    masked_lesion_GL      = np.ma.array(np.multiply(img_GL,    circa_mask)  ,mask=~circa_mask)
    masked_lesion_RGB     = np.ma.array(np.multiply(img_RGB,   np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    masked_lesion_HSV     = np.ma.array(np.multiply(img_HSV,   np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    masked_lesion_LAB     = np.ma.array(np.multiply(img_LAB,   np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    masked_lesion_YCrCb   = np.ma.array(np.multiply(img_YCrCb, np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    masked_lesion_luv     = np.ma.array(np.multiply(img_luv,   np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    # Color Constancy Spaces: I/O ---------------------------------------------------------------------------------------------------------------------------------------------
    img_ccRGB,_           = color_constant(image)
    img_ccGL              = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
    img_ccHSV             = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    img_ccLAB             = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Lab)
    img_ccYCrCb           = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)
    img_ccluv             = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Luv)
    
    masked_lesion_ccGL    = np.ma.array(np.multiply(img_ccGL,    circa_mask)  ,mask=~circa_mask)
    masked_lesion_ccRGB   = np.ma.array(np.multiply(img_ccRGB,   np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    masked_lesion_ccHSV   = np.ma.array(np.multiply(img_ccHSV,   np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    masked_lesion_ccLAB   = np.ma.array(np.multiply(img_ccLAB,   np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    masked_lesion_ccYCrCb = np.ma.array(np.multiply(img_ccYCrCb, np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    masked_lesion_ccluv   = np.ma.array(np.multiply(img_ccluv,   np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    
    
    img_mxRGB             = (correct_image(image, grey_edge(image, njet=0, mink_norm=-1, sigma=0))*255).astype(np.uint8)
    img_mxGL              = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
    img_mxHSV             = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    img_mxLAB             = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Lab)
    img_mxYCrCb           = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)
    img_mxluv             = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Luv)
    
    masked_lesion_mxGL    = np.ma.array(np.multiply(img_mxGL,    circa_mask)  ,mask=~circa_mask)
    masked_lesion_mxRGB   = np.ma.array(np.multiply(img_mxRGB,   np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    masked_lesion_mxHSV   = np.ma.array(np.multiply(img_mxHSV,   np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    masked_lesion_mxLAB   = np.ma.array(np.multiply(img_mxLAB,   np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    masked_lesion_mxYCrCb = np.ma.array(np.multiply(img_mxYCrCb, np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    masked_lesion_mxluv   = np.ma.array(np.multiply(img_mxluv,   np.dstack((circa_mask,circa_mask,circa_mask))), mask=~np.dstack((circa_mask,circa_mask,circa_mask)))
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    
    # Color Moments ----------------------------------------------------------------------------------------------------------------------------------------------------------------
    mean_R, std_R, skew_R, kurt_R, mean_G,  std_G,  skew_G,  kurt_G,  mean_B,  std_B,  skew_B,  kurt_B   = color_moments(masked_lesion_RGB,     channel=3)
    mean_H, std_H, skew_H, kurt_H, mean_S,  std_S,  skew_S,  kurt_S,  mean_V,  std_V,  skew_V,  kurt_V   = color_moments(masked_lesion_HSV,     channel=3)
    mean_L, std_L, skew_L, kurt_L, mean_A,  std_A,  skew_A,  kurt_A,  mean_b,  std_b,  skew_b,  kurt_b   = color_moments(masked_lesion_LAB,     channel=3)
    mean_Y, std_Y, skew_Y, kurt_Y, mean_Cr, std_Cr, skew_Cr, kurt_Cr, mean_Cb, std_Cb, skew_Cb, kurt_Cb  = color_moments(masked_lesion_YCrCb,   channel=3)
    mean_l, std_l, skew_l, kurt_l, mean_u,  std_u,  skew_u,  kurt_u,  mean_v,  std_v,  skew_v,  kurt_v   = color_moments(masked_lesion_luv,     channel=3)
    
    mean_ccR, std_ccR, skew_ccR, kurt_ccR, mean_ccG,  std_ccG,  skew_ccG,  kurt_ccG,  mean_ccB,  std_ccB,  skew_ccB,  kurt_ccB   = color_moments(masked_lesion_ccRGB,   channel=3)
    mean_ccH, std_ccH, skew_ccH, kurt_ccH, mean_ccS,  std_ccS,  skew_ccS,  kurt_ccS,  mean_ccV,  std_ccV,  skew_ccV,  kurt_ccV   = color_moments(masked_lesion_ccHSV,   channel=3)
    mean_ccL, std_ccL, skew_ccL, kurt_ccL, mean_ccA,  std_ccA,  skew_ccA,  kurt_ccA,  mean_ccb,  std_ccb,  skew_ccb,  kurt_ccb   = color_moments(masked_lesion_ccLAB,   channel=3)
    mean_ccY, std_ccY, skew_ccY, kurt_ccY, mean_ccCr, std_ccCr, skew_ccCr, kurt_ccCr, mean_ccCb, std_ccCb, skew_ccCb, kurt_ccCb  = color_moments(masked_lesion_ccYCrCb, channel=3)
    mean_ccl, std_ccl, skew_ccl, kurt_ccl, mean_ccu,  std_ccu,  skew_ccu,  kurt_ccu,  mean_ccv,  std_ccv,  skew_ccv,  kurt_ccv   = color_moments(masked_lesion_ccluv,   channel=3)

    mean_mxR, std_mxR, skew_mxR, kurt_mxR, mean_mxG,  std_mxG,  skew_mxG,  kurt_mxG,  mean_mxB,  std_mxB,  skew_mxB,  kurt_mxB   = color_moments(masked_lesion_mxRGB,   channel=3)
    mean_mxH, std_mxH, skew_mxH, kurt_mxH, mean_mxS,  std_mxS,  skew_mxS,  kurt_mxS,  mean_mxV,  std_mxV,  skew_mxV,  kurt_mxV   = color_moments(masked_lesion_mxHSV,   channel=3)
    mean_mxL, std_mxL, skew_mxL, kurt_mxL, mean_mxA,  std_mxA,  skew_mxA,  kurt_mxA,  mean_mxb,  std_mxb,  skew_mxb,  kurt_mxb   = color_moments(masked_lesion_mxLAB,   channel=3)
    mean_mxY, std_mxY, skew_mxY, kurt_mxY, mean_mxCr, std_mxCr, skew_mxCr, kurt_mxCr, mean_mxCb, std_mxCb, skew_mxCb, kurt_mxCb  = color_moments(masked_lesion_mxYCrCb, channel=3)
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # Graylevel Co-Occurrence Matrix -------------------------------------------------------------------------------------------------------------------------
    GLCM_RGB   = GLCM(masked_lesion_RGB,   channel=3)
    GLCM_HSV   = GLCM(masked_lesion_HSV,   channel=3)
    
    GLCM_mxHSV = GLCM(masked_lesion_mxHSV, channel=3)
    
    GLCM_LAB   = GLCM(masked_lesion_LAB,   channel=3)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------

    
    # Color Markers ----------------------------------------------------------------------------------------------------------------
    CM_black, CM_red, CM_bluegray, CM_white, CM_lightbrown, CM_darkbrown = melanoma_color_markers(masked_lesion_RGB, circa_mask)  
    #-------------------------------------------------------------------------------------------------------------------------------
    
    
    # Local Binary Patterns --------------------------------------------------------------------------------------
    lbp_R, lbp_G, lbp_B    = LBP(masked_lesion_RGB,   channel=3)
    lbp_H, lbp_S, lbp_V    = LBP(masked_lesion_HSV,   channel=3)
    lbp_Y, lbp_Cr, lbp_Cb  = LBP(masked_lesion_YCrCb, channel=3)
    
    lbp_mxH, lbp_mxS, lbp_mxV    = LBP(masked_lesion_mxHSV, channel=3)
    
    LBP_CGLF  = np.concatenate((lbp_R,lbp_G,lbp_B,lbp_H,lbp_S,lbp_V,lbp_Y,lbp_Cr,lbp_Cb),axis=0)
    #--------------------------------------------------------------------------------------------------------------
    
    
    
    # Smoothness, Uniformity, Entropy -----------------------------------------------------------------------------
    entropyplus_RGB  = entropyplus_3(masked_lesion_RGB)
    entropyplus_HSV  = entropyplus_3(masked_lesion_HSV)
    
    entropyplus_mxHSV  = entropyplus_3(masked_lesion_mxHSV)
    
    #--------------------------------------------------------------------------------------------------------------    
    
    
    features = [ mean_R, std_R, skew_R, mean_G,  std_G,  skew_G,  mean_B,  std_B,  skew_B,   
                 mean_H, std_H, skew_H, mean_S,  std_S,  skew_S,  mean_V,  std_V,  skew_V,   
                 mean_L, std_L, skew_L, mean_A,  std_A,  skew_A,  mean_b,  std_b,  skew_b,   
                 mean_Y, std_Y, skew_Y, mean_Cr, std_Cr, skew_Cr, mean_Cb, std_Cb, skew_Cb, 
                 mean_l, std_l, skew_l, mean_u,  std_u,  skew_u,  mean_v,  std_v,  skew_v,   
                
                 CM_black, CM_white, CM_lightbrown, CM_darkbrown,
               
                 mean_ccR, std_ccR, skew_ccR, mean_ccG,  std_ccG,  skew_ccG,  mean_ccB,  std_ccB,  skew_ccB, 
                 mean_ccH, std_ccH, skew_ccH, mean_ccS,  std_ccS,  skew_ccS,  mean_ccV,  std_ccV,  skew_ccV, 
                 mean_ccL, std_ccL, skew_ccL, mean_ccA,  std_ccA,  skew_ccA,  mean_ccb,  std_ccb,  skew_ccb, 
                 mean_ccY, std_ccY, skew_ccY, mean_ccCr, std_ccCr, skew_ccCr, mean_ccCb, std_ccCb, skew_ccCb, 
                 mean_ccl, std_ccl, skew_ccl, mean_ccu,  std_ccu,  skew_ccu,  mean_ccv,  std_ccv,  skew_ccv,
               
                 mean_mxR, std_mxR, skew_mxR, mean_mxG,  std_mxG,  skew_mxG,  mean_mxB,  std_mxB,  skew_mxB, 
                 mean_mxH, std_mxH, skew_mxH, mean_mxS,  std_mxS,  skew_mxS,  mean_mxV,  std_mxV,  skew_mxV, 
                 mean_mxL, std_mxL, skew_mxL, mean_mxA,  std_mxA,  skew_mxA,  mean_mxb,  std_mxb,  skew_mxb, 
                 mean_mxY, std_mxY, skew_mxY, mean_mxCr, std_mxCr, skew_mxCr, mean_mxCb, std_mxCb, skew_mxCb ]
 
    features = np.concatenate((features, GLCM_RGB, GLCM_HSV, GLCM_LAB, LBP_CGLF, entropyplus_RGB, entropyplus_HSV),axis=0)

    return features
