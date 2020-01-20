# Color/Texture Features with Support Vector Machine and Random Forest for Disease Classification of Skin Lesions
**Problem Statement**: Fully supervised binary classification of skin lesions from dermatoscopic images. 

**Note**: The following approach won 1st place at the **2019 Computer-Aided Diagnosis: Machine Learning in Dermascopy Challenge** at [Universitat de Girona](https://www.udg.edu), during the 2018-20 Joint Master of Science in [Medical Imaging and Applications (MaIA)](https://maiamaster.udg.edu) program. 

**Acknowledgments**: Mina Sami for the Python implementation of [Shades of Gray Color Constancy](https://github.com/MinaSGorgy/Color-Constancy). 

**Data**: *Class A*: Nevus; *Class B:* Other (Melanoma, Dermatofibroma, Pigmented Bowen's, Basal Cell Carcinoma, Vascular, Pigmented Benign Keratoses) [4800/1200/1000 : Train/Val/Test Ratio]
 
**Directories**  
  ● Data I/O Functions: `scripts/dataio.py`  
  ● Preprocessing Functions: `scripts/preprocess.py`  
  ● Unsupervised Segmentation Functions: `scripts/segment.py`  
  ● Feature Computation Functions: `scripts/colorfeatures.py`               
  ● Final Feature Extraction Function: `scripts/feature_extraction.py`               
  ● Classifier Support Functions: `scripts/classify.py`  
  ● Inference Pipeline Notebook: `scripts/predict.ipynb`  
  ● Training-Validation Pipeline Notebook: `scripts/train-val.ipynb` 


## Color Constancy  

![Color Constancy](reports/images/pre_wbcc.png) 
   
    
## Occlusion Removal  

![Hair Removal](reports/images/occlusion_clahe.png) 
  
    
## Unsupervised Segmentation 

![Unsupervised Segmentation](reports/images/segmentation_ac.png) 


## Color Space 
![Color Space](reports/images/colorspace.png) 


## Gabor Filter Features
![Gabor Filter Features](reports/images/gabor.png) 


## HOG Features 
![HOG Features](reports/images/hog.png) 


## Feature Selection

![Feature Selection](reports/images/feature_selection.png) 


## Experimental Results

![Experimental Results](reports/images/results.png) 

