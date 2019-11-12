# The *Classical* Approach: Color/Texture Features with Support Vector Machine and Random Forest
**Problem Statement**: Fully supervised binary classification of skin lesions from dermatoscopic images. 

**Data**: *Class A*: Nevus; *Class B:* Other (Melanoma, Dermatofibroma, Pigmented Bowen's, Basal Cell Carcinoma, Vascular, Pigmented Benign Keratoses). 
 
**Directories**  
  ● Convert DICOM to NIfTI Volumes: `preprocess/prime/DICOM_NIFTI.py`  
  ● Resample NIfTI Volume Resolutions: `preprocess/prime/resampleRes.py`  
  ● Infer StFA/DyFA Segmentation Sub-Model (DenseVNet): `python net_segment.py inference -c '../config.ini'`  
  ● Preprocess Full Dataset to Optimized I/O HDF5 Training Patch-Volumes: `preprocess/prime/preprocess_alpha.py`               


## Color Constancy  

![Color_Constancy](reports/images/pre_wbcc.png) 
   
    
## Occlusion Removal  

![Hair_Removal](reports/images/occlusion_clahe.png) 
  
    
## Unsupervised Segmentation 

![Unsupervised_Segmentation](reports/images/segmentation_ac.png) 


## Color Space 
![Color_Space](reports/images/colorspace.png) 


## Gabor Filter Features
![Gabor_Filter_Features](reports/images/gabor.png) 


## HOG Features 
![HOG_Features](reports/images/hog.png) 


## Feature Selection

![Feature_Selection](reports/images/feature_selection.png) 


## Experimental Results

![Experimental_Results](reports/images/results.png) 

