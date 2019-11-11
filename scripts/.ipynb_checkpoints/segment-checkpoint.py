### Libraries
import numpy as np
import cv2
from scipy import ndimage
from skimage import img_as_float
from sklearn.cluster import KMeans
from skimage.measure import label as skilabel
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from skimage.morphology.convex_hull import convex_hull_image
from scipy.stats import multivariate_normal









### Unsupervised Segmentation
def segment_lesion(image,mode="KMeans"):
    ## Unsupervised Segmentation
    # K-Means Clustering
    if (mode=="KMeans"):
        # K-Means Clustering
        deim    = denoise(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), weight=150)
        kmeans  = KMeans(n_clusters=2, random_state=0, n_jobs=-1).fit(deim.reshape(-1,1))
        mask    = (kmeans.labels_).reshape(deim.shape[0],deim.shape[1]).astype('uint8')
    
    # Expectation-Maximization Gaussian Mixture Model    
    elif (mode=="EM"):
        # K-Means Clustering
        feature_vector             = (denoise(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), weight=150)).reshape(-1,1)
        kmeans                     = KMeans(n_clusters=2, random_state=0, n_jobs=-1).fit(feature_vector)
        KMpredict                  = kmeans.predict(feature_vector)

        # Expectation Step: Initialization with K-Means
        KM_BG        = feature_vector[KMpredict==0]
        KM_FG        = feature_vector[KMpredict==1]

        # Expectation Step: Mean and Covariance
        mean_BG      = np.mean(KM_BG, axis = 0)
        mean_FG      = np.mean(KM_FG, axis = 0)
        covar_BG     = np.cov(KM_BG,  rowvar = False)
        covar_FG     = np.cov(KM_FG,  rowvar = False)

        # Expectation Step: Prior Probabilities
        prob_BG      = KM_BG.shape[0] / feature_vector.shape[0]
        prob_FG      = KM_FG.shape[0] / feature_vector.shape[0]

        # Iterative Update
        min_change   = 0.01
        max_steps    = 5

        for i in range(max_steps):
            # Expectation Step: Probability Density Function
            PDF_BG       = multivariate_normal.pdf(feature_vector, mean=mean_BG, cov=covar_BG)
            PDF_FG       = multivariate_normal.pdf(feature_vector, mean=mean_FG, cov=covar_FG)
            weights_BG   = (prob_BG * PDF_BG)/((prob_FG * PDF_FG) + (prob_BG * PDF_BG))
            weights_FG   = (prob_FG * PDF_FG) /((prob_FG * PDF_FG) + (prob_BG * PDF_BG))
            weights      = np.concatenate((weights_BG.reshape(-1,1),weights_FG.reshape(-1,1)),axis=1)
            log_B        = sum((np.log(sum(weights))))

            # Maximization Step: New Probabilities
            _,counts     = np.unique(np.argmax(weights,axis=1), return_counts=True)
            prob_BG      = counts[0] / feature_vector.shape[0]
            prob_FG      = counts[1] / feature_vector.shape[0]

            # Maximization Step: New Mean and Covariance
            mean_BG      = (1/counts[0]) * (weights[:,0] @ feature_vector)
            mean_FG      = (1/counts[1]) * (weights[:,1] @ feature_vector)
            covar_BG     = (1/counts[0]) * (weights[:,0] * np.transpose(feature_vector - mean_BG)) @ (feature_vector - mean_BG)
            covar_FG     = (1/counts[1]) * (weights[:,1] * np.transpose(feature_vector - mean_FG)) @ (feature_vector - mean_FG)

            # Maximization Step: Probability Density Function
            PDF_BG       = multivariate_normal.pdf(feature_vector, mean=mean_BG, cov=covar_BG)
            PDF_FG       = multivariate_normal.pdf(feature_vector, mean=mean_FG, cov=covar_FG)
            weights_BG   = (prob_BG * PDF_BG)/((prob_FG * PDF_FG) + (prob_BG * PDF_BG))
            weights_FG   = (prob_FG * PDF_FG) /((prob_FG * PDF_FG) + (prob_BG * PDF_BG))
            weights      = np.concatenate((weights_BG.reshape(-1,1),weights_FG.reshape(-1,1)),axis=1)
            log_N        = sum((np.log(sum(weights))))

            # Update Trackers, Verify Conditions
            change_log   = np.linalg.norm(log_N-log_B)
            if (change_log <= min_change):
                continue
            else:
                break
        
        # Output Image Reconstruction
        mask        = (np.argmax(weights,axis=1)).reshape(-1,1)
        mask        = np.reshape(mask,(image.shape[0],image.shape[1])).astype('uint8')
            
    # Chan-Vese Active Contours   
    elif (mode=="active_contours"):
        # Morphological ACWE
        image     = img_as_float(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))

        # Initial level Set
        init_ls   = checkerboard_level_set(image.shape, 6)

        # List with Intermediate Results for Plotting the Evolution
        evolution = []
        callback  = store_evolution_in(evolution)
        mask      = morphological_chan_vese(image, 5, init_level_set=init_ls, smoothing=10, iter_callback=callback).astype(np.uint8)
    
    else:
        print("ERROR: Undefined Segmentation Mode")
    
    ## Segmentation Label Selection
    # Define Target Scope
    candidate_mask_1 = mask.copy() * create_circular_mask(image.shape[0], image.shape[1], radius = 230)
    candidate_mask_2 = (np.invert(mask.copy()) + 2) * create_circular_mask(image.shape[0], image.shape[1], radius = 230)

    # Compute Area of Convex Hulls
    convex_hull_1 = convex_hull_image(candidate_mask_1)
    convex_hull_2 = convex_hull_image(candidate_mask_2)

    # Set Segmentation Component Labels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    if (np.sum(convex_hull_1) < np.sum(convex_hull_2)):
        dilation           = cv2.dilate(candidate_mask_1,kernel,iterations = 1)
        mask               = ndimage.binary_fill_holes(dilation, structure=np.ones((5,5)))
    else:
        dilation           = cv2.dilate(candidate_mask_2,kernel,iterations = 1)
        mask               = ndimage.binary_fill_holes(dilation, structure=np.ones((5,5)))
    if (np.sum(mask)<5000):
        mask = create_circular_mask(image.shape[0], image.shape[1], radius = 230)
    return mask




def getLargestCC(segmentation):
    labels    = skilabel(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC




def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask




def store_evolution_in(lst):
    def _store(x):
        lst.append(np.copy(x))
    return _store




def denoise(img, weight=0.1, eps=1e-3, num_iter_max=200):
    # Denoising Gray-Level Image via PDE-ROF Model 
    u   = np.zeros_like(img)
    px  = np.zeros_like(img)
    py  = np.zeros_like(img)
    nm  = np.prod(img.shape[:2])
    tau = 0.125

    i = 0
    while (i < num_iter_max):
        u_old = u

        # X and Y Components of U's Gradient
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u

        # Update Dual Variable
        px_new   = px + (tau / weight) * ux
        py_new   = py + (tau / weight) * uy
        norm_new = np.maximum(1, np.sqrt(px_new **2 + py_new ** 2))
        px       = px_new / norm_new
        py       = py_new / norm_new

        # Calculate Divergence
        rx    = np.roll(px, 1, axis=1)
        ry    = np.roll(py, 1, axis=0)
        div_p = (px - rx) + (py - ry)

        # Update Image
        u = img + weight * div_p

        # Calculate Error
        error = np.linalg.norm(u - u_old) / np.sqrt(nm)
        if i == 0:
            err_init = error
            err_prev = error
        else:
            # Break for Small Error
            if np.abs(err_prev - error) < eps * err_init:
                break
            else:
                err_prev = error
        i += 1
    return u