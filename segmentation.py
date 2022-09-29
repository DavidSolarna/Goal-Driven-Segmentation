# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:03:57 2020

@author: David
"""

import utils

import numpy as np
from tqdm import tqdm
import math
import random
import itertools
from skimage import measure
import skimage.morphology as morph
from skimage.filters import threshold_otsu
import logging

import matplotlib.pyplot as plt

#import maxflow
from maxflow import fastmin
from enum import Enum, unique, auto
from sklearn.mixture import BayesianGaussianMixture
from scipy import special
from scipy.optimize import newton
from scipy.ndimage.morphology import binary_closing


#-----------------------------------
#--- UTILITIES
#-----------------------------------


@unique
class Type_Model(Enum):
    GAUSSIAN = auto()
    GAUSSIAN_MIXTURE = auto()
    LOGNORMAL = auto()
    WEIBULL = auto()
    GAMMA = auto()
    
    @staticmethod
    def to_list():
        return [element.name for element in Type_Model]    
    @staticmethod
    def to_string():
        return str([element.name for element in Type_Model])
    
    
#-----------------------------------
#--- SEGMENTATION
#-----------------------------------


class Segmentation:
    
    #---- Initialization
    
    def __init__(self, beta=5, gamma=0.5, thr_memb=0.7, filter_memb_ci_value=None, preproc_level=5):
        self.image_orig = None
        self.image_preproc = None
        self.size_r = None
        self.size_c = None
        # Graph
        self.seeds = []
        # self.seedCoord = []
        self.weight_images = []
        self.unique_levels = []
        self.cost_images = []
        self.propagations = []
        # Membership
        self.membership = []
        self.membership_unprocessed = []
        # self.compactMembership = []
        # Models
        self.type_model = None
        self.num_param = None
        self.func_model_estimation = None
        self.func_model = None
        self.models = []
        self.likelihood_images = []
        # Spatial model
        self.spatial_centroids = []
        self.compactness_indices = []
        self.distance_unary = None
        # Graph Cut
        self.initialization = None
        # Result
        self.segmented_image = None
        # Options
        self.threshold_membership_list = []
        self.gc_max_iter = 100
        self.gc_pairwise_weight = 5
        # Parameters
        self.beta = beta
        self.gamma = gamma
        self.threshold_membership = thr_memb # (0-1) / otsu / ci
        self.filter_memb_ci_value = filter_memb_ci_value
        self.preproc_level = preproc_level
        
        # Misc
        self.log = logging.getLogger(__name__)
        
        
    def _preprocess_image(self, image):
        image = np.asarray(image)
        image_preproc = utils.imadjust(image, tol=self.preproc_level)
        image_preproc = np.floor(utils.normalize_datacube_0_1(image_preproc)*255)
        return image_preproc
    
    
    def set_image(self, image_orig, image_preproc=None):
        if np.ndim(image_orig) != 2:
            raise ValueError("Only 2D images are supported.")
        self.image_orig = np.floor(utils.normalize_datacube_0_1(image_orig)*255)
        self.size_r, self.size_c = np.shape(image_orig)
        if image_preproc is None:
            self.image_preproc = self._preprocess_image(image_orig)
        else:
            if np.shape(image_orig) != np.shape(image_preproc):
                raise ValueError("The image shapes must match.")
            self.image_preproc = image_preproc
        
        
    def set_model_type(self, type_model):
        if type_model == Type_Model.GAUSSIAN:
            self.type_model = Type_Model.GAUSSIAN
            self.num_param = 2
            self.func_model_estimation = self._gaussian_model_estimation
            self.func_model = utils.gaussian_model
        elif type_model == Type_Model.GAUSSIAN_MIXTURE:
            self.type_model = Type_Model.GAUSSIAN_MIXTURE
            self.num_param = 3
            self.func_model_estimation = self._gaussian_mixture_model_estimation
            self.func_model = None
        elif type_model == Type_Model.LOGNORMAL:
            self.type_model = Type_Model.LOGNORMAL
            self.num_param = 2
            self.func_model_estimation = self._lognormal_model_estimation
            self.func_model = utils.lognormal_model
        elif type_model == Type_Model.WEIBULL:
            self.type_model = Type_Model.WEIBULL
            self.num_param = 2
            self.func_model_estimation = self._weibull_model_estimation
            self.func_model = utils.weibull_model
        elif type_model == Type_Model.GAMMA:
            self.type_model = Type_Model.GAMMA
            self.num_param = 2
            self.func_model_estimation = self._gamma_model_estimation
            self.func_model = utils.gamma_model
        else:
            raise ValueError("No support for model {0}. Supported types are: \
                             {1}".format(type_model.name, Type_Model.to_string()))
                             
                             
    #---- Seed Logic
    
                    
    def _is_smooth_area(self, seed, max_std, window_size):
        image = self.image_preproc
        r0, c0 = seed
        l = int(window_size/2)
        try:
            window = image[r0-l:r0+l, c0-l:c0+l]
            std = np.std(window)
            if std < max_std:
                return True
            else:
                return False
        except:
            return False
        
        
    def _get_mean_in_window(self, seed, window_size):
        image = self.image_preproc
        r0, c0 = seed
        l = int(window_size/2)
        seed_value = image[r0, c0]
        try:
            window = image[r0-l:r0+l, c0-l:c0+l]
            return np.mean(window)
        except:
            return seed_value
                             
    
    def _randomize_seed(self, seed, stepR, stepC):
        image = self.image_preproc
        # Extract
        (sr, sc) = seed
        # ROW -> Y
        sr = int(sr + (random.uniform(-0.5,0.5)*stepR))
        sr = min(image.shape[0]-5, sr)
        sr = max(5, sr)
        # COL -> Y
        sc = int(sc + (random.uniform(-0.5,0.5)*stepC))
        sc = min(image.shape[1]-5, sc)
        sc = max(5, sc)
        # Ret
        return (sr, sc)
                        
        
    def add_seed_rc(self, seed):
        if isinstance(seed, tuple) and len(seed) == 2:
            if seed > (0, 0) and seed < (self.size_r, self.size_c):
                self.seeds.append(seed)
            else:
                raise ValueError("The seed must be inside the image.")
        else:
            raise ValueError("The seed must be a tuple of 2 elements.")
            
            
    def add_seed_xy(self, seed):
        seed_rc = (seed[1], seed[0])
        self.add_seed_rc(seed_rc)
                    
                    
    def _seed_selection(self, seed_list_rc, min_value, max_value, max_std, window_size):
        for idx, seed in enumerate(seed_list_rc):
            m = self._get_mean_in_window(seed, window_size)
            if m > min_value and m < max_value:
                if self._is_smooth_area(seed, max_std, window_size):
                    self.add_seed_rc(seed)
        
        
    def add_seed_auto(self, startX, startY, stepX, stepY, 
                      min_value=0, max_value=255, max_std=35, window_size=5):
        # Fixed grid
        listR = np.arange(start=startY, stop=self.image_orig.shape[0], step=stepY)
        listC = np.arange(start=startX, stop=self.image_orig.shape[1], step=stepX)
        stepR = stepY
        stepC = stepX
        # Iterate
        seed_list_rc = []
        for sr, sc in list(itertools.product(listR, listC)):
            s = self._randomize_seed((sr, sc), stepR, stepC)
            seed_list_rc.append(s)
        self._seed_selection(seed_list_rc, min_value, max_value, max_std, window_size)
        
        
    def show_seed_location(self):
        fig = plt.figure()
        plt.imshow(self.image_preproc)
        for seed in self.seeds:
            plt.plot(seed[1], seed[0], 'ro')
        plt.show()
        self.seed_image = fig
        
        
    #---- Reset Logic
    
    
    def reset_flood_fill(self):
        self.weight_images = []
        self.unique_levels = []
        self.cost_images = []
        self.propagations = []
        self.membership = []
        self.membership_unprocessed = []
        
        
    def reset_models(self):
        self.models = []
        self.likelihood_images = []
        
        
    def _set_random_initialization(self):
        self.initialization = None
        

    def reset_graph_cut(self):
        self._set_random_initialization()
        self.D = None
        self.V = None
        
        
    def reset_all(self):
        self._reset_flood_fill()
        self._reset_models()
        self._reset_graph_cut()
        
        
    #---- FloodFill Logic
        
        
    def _compute_compact_index(self, area, perimeter):
        return (area * 4*math.pi) / (perimeter * perimeter)
    
    
    def _is_valid_prop(self, properties):
        if len(properties) == 1 and 30 < properties[0].area < 0.5*(self.size_r*self.size_c):
            return True
        return False
        
        
    def _optimize_threshold_membership_ci(self, membership):
        m = membership.copy()
        thresholds = np.linspace(80, 70, 80-69, endpoint=True)/100
        ci = np.zeros([len(thresholds)])
        for idx, t in enumerate(thresholds):
            m_b = m.copy()
            m_b[m>=t] = 1
            m_b[m<t] = 0
            m_b = m_b.astype('int8')
            m_b = self._bin_and_morph_membership(m_b)
            properties = measure.regionprops(m_b.astype('int8'))
            if self._is_valid_prop(properties):
                ci[idx] = self._compute_compact_index(properties[0].area, properties[0].perimeter)
        best_idx = np.argmax(ci)
        best_threshold = thresholds[best_idx]
        self.threshold_membership_list.append(best_threshold)
        return best_threshold
    
    
    def _optimize_threshold_membership_otsu(self, membership):
        memb = membership.copy()
        best_threshold = threshold_otsu(memb)
        return best_threshold
    
    
    def _bin_and_morph_membership(self, membership):
        closed_memb_mask = binary_closing(membership)
        c_memb = np.zeros(np.shape(membership))
        c_memb[closed_memb_mask] = 1
        return c_memb
    
    
    def _flood_fill_core_v0(self, seed):
        
        # Use the preproc image for the Flood Fill
        image = self.image_preproc.copy()
        
        # Compute the weight image for a seed 
        weight = np.absolute(image - image[seed])
        self.weight_images.append(weight)
        # Determine the unique level for a seed
        unique_lev = np.unique(weight)
        self.unique_levels.append(unique_lev)
        
        # Initialization
        cost_img = np.zeros_like(image) - 1
        
        # Run flood fill
        prop = np.zeros([*np.shape(self.image),len(unique_lev)])
        for idx, level in enumerate(tqdm(reversed(unique_lev), total=len(unique_lev))):
            prop[:,:,idx] = utils.flood_fill_core(
                prop[:,:,idx].copy(), weight.copy(), cost_img, seed, level)
        self.cost_images.append(cost_img)
        self.propagations.append(prop)
        
        # Membership computation
        max_diff = np.amax(unique_lev)
        norm_cost = cost_img / max_diff
        memb = np.ones_like(norm_cost) - norm_cost
        self.membership_unprocessed.append(memb.copy())
        
        # Thresholding
        if 0 < self.threshold_membership < 1:
            memb[memb<self.threshold_membership] = 0
            memb = self._bin_and_morph_membership(memb)
            self.membership.append(memb)
        elif self.threshold_membership == "ci":
            threshold = self._optimize_threshold_membership_ci(memb)
            self.threshold_membership_list.append(threshold)
            memb[memb<threshold] = 0
        elif self.threshold_membership == "otsu":
            threshold = self._optimize_threshold_membership_otsu(memb)
            self.threshold_membership_list.append(threshold)
            memb[memb<threshold] = 0
        else:
            self.log.error("Invalid threshold.")
            raise ValueError("Invalid threshold.")
            
        # Save
        self.membership.append(memb)
        
    
    def _flood_fill_core(self, seed):
        
        # Use the preproc image for the Flood Fill
        image = self.image_preproc.copy()
        
        # Compute the weight image for a seed 
        weight = np.absolute(image - image[seed])
        self.weight_images.append(weight)
        # Determine the unique level for a seed
        unique_lev = np.unique(weight)
        self.unique_levels.append(unique_lev)
        
        # Initialization
        cost_img = np.zeros_like(image) - 1
        
        # Run flood fill
        for idx, level in enumerate(tqdm(reversed(unique_lev), total=len(unique_lev))):
            iterCost = morph.flood_fill(image.copy(), seed, 9999, tolerance = level)            
            cost_img[iterCost == 9999] = level
            
        # Membership computation
        max_diff = np.amax(unique_lev)
        norm_cost = cost_img / max_diff
        memb = np.ones_like(norm_cost) - norm_cost
        self.membership_unprocessed.append(memb.copy())
        
        # Thresholding
        error_message = "Invalid threshold. Must be either in range (0,1) or \"ci\" or \"otsu\"."
        if isinstance(self.threshold_membership, float):
            if 0 < self.threshold_membership < 1:
                memb[memb<self.threshold_membership] = 0
                memb = self._bin_and_morph_membership(memb)
            else:
                self.log.error(error_message)
                raise ValueError(error_message)
        elif isinstance(self.threshold_membership, str):
            if self.threshold_membership == "ci":
                threshold = self._optimize_threshold_membership_ci(memb)
                self.threshold_membership_list.append(threshold)
                memb[memb<threshold] = 0
            elif self.threshold_membership == "otsu":
                threshold = self._optimize_threshold_membership_otsu(memb)
                self.threshold_membership_list.append(threshold)
                memb[memb<threshold] = 0
            else:
                self.log.error(error_message)
                raise ValueError(error_message)
        else:
            self.log.error(error_message)
            raise ValueError(error_message)
                
            
        # Save
        self.membership.append(memb)
    
    
    def _flood_fill(self):
        [self._flood_fill_core(seed) for seed in self.seeds]

        
    def _filter_membership(self):
        # Filter logic
        idx_to_remove = []
        for idx, m in enumerate(self.membership):
            aux = np.copy(m)
            aux[aux>0] = 1
            aux = aux.astype('int8')
            properties = measure.regionprops(aux)
            if len(properties) == 0 or (self._is_valid_prop(properties)==False):
                idx_to_remove.append(idx)
            else:
                if self.filter_memb_ci_value is not None:
                    comp_ind = self._compute_compact_index(properties[0].area, properties[0].perimeter)
                    if comp_ind < self.filter_memb_ci_value:
                        idx_to_remove.append(idx)
        # Update
        new_seeds = [s for i, s in enumerate(self.seeds) if i not in idx_to_remove]
        new_memb = [m for i, m in enumerate(self.membership) if i not in idx_to_remove]
        new_memb_unp = [m for i, m in enumerate(self.membership_unprocessed) 
                        if i not in idx_to_remove]
        new_weight_images = [wi for i, wi in enumerate(self.weight_images) 
                             if i not in idx_to_remove]
        new_unique_levels = [ul for i, ul in enumerate(self.unique_levels) 
                             if i not in idx_to_remove]
        new_cost_images = [ci for i, ci in enumerate(self.cost_images) 
                           if i not in idx_to_remove]
        new_propagations = [p for i, p in enumerate(self.propagations) 
                            if i not in idx_to_remove]
        # Save
        self.seeds = new_seeds
        self.membership = new_memb
        self.membership_unprocessed = new_memb_unp
        self.weight_images = new_weight_images
        self.unique_levels = new_unique_levels
        self.cost_images = new_cost_images
        self.propagations = new_propagations
        
        
    #---- Model Estimation Logic
 
       
    def _gaussian_model_estimation(self, membership):
        binMemb = np.zeros_like(membership)
        binMemb[membership>0] = 1 
        filteredImage = np.multiply(self.image_orig, binMemb)
        mean = np.sum(filteredImage.flatten()) / np.sum(membership.flatten())
        var = np.sqrt(np.sum(np.square(filteredImage.flatten()-mean)) / np.sum(membership.flatten()))
        return np.asarray([mean, var])
    
    
    def _gaussian_mixture_model_estimation(self, membership):
        binMemb = np.zeros_like(membership)
        binMemb[membership>0] = 1 
        filteredImage = np.multiply(self.image_orig, binMemb)
        estimator = BayesianGaussianMixture()
        samples = filteredImage[filteredImage>0].flatten()
        estimator.fit(np.reshape(samples, [len(samples),1]))
        return estimator
    
    
    def _lognormal_model_estimation(self, membership):
        binMemb = np.zeros_like(membership)
        binMemb[membership>0] = 1 
        filteredImage = np.multiply(self.image_orig, binMemb)
        samples = filteredImage[filteredImage>0].flatten()
        k1 = np.sum(np.log(samples)) / np.sum(membership.flatten())
        k2 = np.sum(np.square(np.log(samples)-k1)) / np.sum(membership.flatten())
        return np.asarray([k1, k2])
    
    
    def _weibull_model_estimation(self, membership):
        binMemb = np.zeros_like(membership)
        binMemb[membership>0] = 1 
        filteredImage = np.multiply(self.image_orig, binMemb)
        samples = filteredImage[filteredImage>0].flatten()
        k1 = np.sum(np.log(samples)) / np.sum(membership.flatten())
        k2 = np.sum(np.square(np.log(samples)-k1)) / np.sum(membership.flatten())
        aux_mu = k1 - ((np.sqrt(k2)*special.polygamma(0, 1))/(np.sqrt(special.polygamma(1, 1))))
        mu = np.exp(aux_mu)
        eta = np.sqrt(special.polygamma(1, 1) / k2)
        return np.asarray([eta, mu])
    
    
    def _gamma_model_estimation(self, membership):
        binMemb = np.zeros_like(membership)
        binMemb[membership>0] = 1 
        filteredImage = np.multiply(self.image_orig, binMemb)
        samples = filteredImage[filteredImage>0].flatten()
        k1 = np.sum(np.log(samples)) / np.sum(membership.flatten())
        k2 = np.sum(np.square(np.log(samples)-k1)) / np.sum(membership.flatten())
        L = newton(utils.f,x0=5,args=[k2],maxiter=200) # initial approximation = 5
        aux_mu = k1 - special.polygamma(0,L) + math.log(L)
        mu = math.exp(aux_mu)
        return np.asarray([L, mu])
    
    
    #---- Unary Logic
    
       
    def _estimate_models(self):
        for seed_membership in self.membership:
            single_model = self.func_model_estimation(seed_membership)
            self.models.append(single_model)
            
            
    def _compute_likelihood(self):
        for idx, model_params in enumerate(self.models):
            if self.type_model != Type_Model.GAUSSIAN_MIXTURE:
                likelihood = self.func_model(self.image_orig, model_params)
            else:
                image_flatten = self.image_orig.copy().flatten()
                image_flatten = np.reshape(image_flatten, [self.size_c*self.size_r, 1])
                likelihood = model_params.score_samples(image_flatten)
                likelihood = np.reshape(likelihood, [self.size_r, self.size_c])
            self.likelihood_images.append(likelihood)
            

    def _estimate_spatial_models(self):
        for m in self.membership:
            m[m>0] = 1
            m = m.astype('int8')
            props = measure.regionprops(m)
            self.spatial_centroids.append(props[0].centroid)
            ci = self._compute_compact_index(props[0].area,props[0].perimeter)
            self.compactness_indices.append(ci)
            
            
    #---- GC Logic
        
        
    def _set_initialization_from_flood_fill(self):
        if len(self.membership)==0:
            raise ValueError("Compute the membership before initializing the labels.")
        membership_array = np.asarray(self.membership)
        membership_array = np.moveaxis(membership_array, 0, 2)
        self.initialization = np.argmax(membership_array, axis=2)
            
    
    def _compute_distances_unary(self):
        self.distance_unary = np.zeros([*np.shape(self.image_orig), len(self.seeds)])
        for idx, centroid in enumerate(self.spatial_centroids):
            rr, cc = np.meshgrid(range(self.size_r), range(self.size_c), indexing='ij')
            d_squared = np.square((rr[:,:] - centroid[0]))+np.square(cc[:,:] - centroid[1])
            distance_unary = np.sqrt(d_squared)
            distance_unary[distance_unary==0] = 1
            self.distance_unary[:,:,idx] = distance_unary
            
            
    def _compute_bg_weight(self, unary):
        all_values = unary.flatten()
        return np.percentile(all_values, 1), np.percentile(all_values, 99)
            
    
    def _run_graph_cut(self):
        # Unary
        num_labels = len(self.likelihood_images)
        D = np.ndarray(shape=(self.size_r, self.size_c, num_labels+1))
        for idx, seed_likelihood in enumerate(self.likelihood_images):
            D[:,:,idx] = - seed_likelihood
            dist_unary = 1 / self.distance_unary[:,:,idx]
            dist_unary = (np.max(D[:,:,idx]) / np.max(dist_unary)) * dist_unary
            D[:,:,idx] = D[:,:,idx] - (self.gamma * dist_unary)
        # Add background
        low_energy, high_energy = self._compute_bg_weight(D[:,:,:-1])
        bg = np.sum(np.asarray(self.membership), axis=0)==0
        bg_unary = np.ones_like(self.image_orig) * high_energy # 10
        bg_unary[bg] = low_energy # -5
        D[:,:,num_labels] = bg_unary
        self.log.info("BG_VALUES: {}, {}".format(low_energy, high_energy))
        # Pairwise
        V = np.ndarray(shape=(num_labels+1, num_labels+1))
        for l1 in range(0, num_labels+1):
            for l2 in range(0, num_labels+1):
                if l1 == l2:
                    V[l1, l2] = 0
                else:
                    V[l1, l2] = self.beta
        # Initialization
        self._set_initialization_from_flood_fill()
        init_labels = self.initialization        
        # GC
        max_cycles = self.gc_max_iter
        self.segmented_image = fastmin.abswap_grid(D, V, max_cycles, init_labels)
        
        
    #---- RUN Interface
        
    
    def run(self):
        if self.image_orig is None:
            raise ValueError("Set the image before running the segmentation.")
        if len(self.seeds) == 0:
            raise ValueError("Add at least one seed before running the segmentation.")
        if self.type_model is None:
            raise ValueError("Set the model before running the segmentation.")
        
        self._flood_fill()
        self._filter_membership()
        self._estimate_models()
        self._estimate_spatial_models()
        self._compute_distances_unary()
        self._compute_likelihood()
        self._run_graph_cut()
        
        
    def get_segmentation(self):
        if self.segmented_image is not None:
            return self.segmented_image
        else:
            raise ValueError("Run the segmentation method first.")