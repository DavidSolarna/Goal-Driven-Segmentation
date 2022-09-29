# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:28:31 2020

@author: David
"""

from segmentation import Segmentation
from segmentation import Type_Model
import utils

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFilter

import sys
import logging
from datetime import datetime
import pathlib
import json

# import pydicom


#%% Image Manipulation

def create_image():
    
    image = Image.new("L", (300, 300), 50)
    
    draw = ImageDraw.Draw(image)
    draw.ellipse([(25, 50), (100, 120)], fill=180)
    draw.ellipse([(120, 120), (200, 220)], fill=190)
    draw.rectangle([(150, 20), (280, 80)], fill=160)
    draw.rectangle([(10, 160), (90, 290)], fill=200)
    draw.ellipse([(220, 160), (280, 290)], fill=210)
    
    image_arr = np.asarray(image)
    noise = np.random.randint(-20,20,(300,300),dtype=np.dtype('int8'))
    image_out = Image.fromarray(image_arr+noise)
    
    return image_out


#%% Setup

data_folder = pathlib.Path("data")
data_folder.mkdir(parents=True, exist_ok=True)
result_folder = pathlib.Path("results")
result_folder.mkdir(parents=True, exist_ok=True)
log_level = "info"


#%% Logging

# Change root logger level from WARNING (default) to NOTSET in order for all messages to be delegated.
logging.getLogger().setLevel(logging.NOTSET)
# log_filename.mkdir(parents=True, exist_ok=True)
log_filename = data_folder / 'log_{}.log'.format(datetime.now().strftime("%Y-%m-%d-%H-%M"))

# Add stdout handler, with level INFO
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.getLevelName(log_level.upper()))
formatter_1 = logging.Formatter('%(asctime)s [%(name)s-%(levelname)s]: %(message)s')
console.setFormatter(formatter_1)
logging.getLogger().addHandler(console)

# Add file rotating handler, with level DEBUG
fileHandler = logging.FileHandler(filename=log_filename)
fileHandler.setLevel(logging.getLevelName(log_level.upper()))
formatter_2 = logging.Formatter('%(asctime)s [%(name)s-%(levelname)s]: %(message)s')
fileHandler.setFormatter(formatter_2)
logging.getLogger().addHandler(fileHandler)

log = logging.getLogger(__name__)
log.info("Configuration Complete.")


#%% Load and preprocess image

#---- Parsing
    
parser = utils.get_parser()
args = parser.parse_args()
config_file_path = args.config
# config_file = utils.format_path_string_on_platform(config_file_path)
config_file = pathlib.Path(config_file_path)

# Load
with open(config_file) as json_file:
    config = json.load(json_file)
image_path = config["image_path"]
min_value_seed = config["min_value_seed"]
max_value_seed = config["max_value_seed"]
beta = config["beta"]
gamma = config["gamma"]
thr_memb = config["thr_memb"]
preproc_level = config["preproc_level"]
step_x = config["step_x"]
step_y = config["step_y"]
max_std = config["max_std"]
window_size = config["window_size"]
model_type = config["model_type"]
seed_file = config["seed_file"]

#---- Loading

# image_path = pathlib.Path(utils.format_path_string_on_platform(image_path))
image_path = pathlib.Path(image_path)

if image_path.is_file():    
    image_pil = Image.open(image_path).convert('L') 
else:
    logging.warning("The image passed as argument does not exists. Creating a test image.")
    image_pil = create_image()

if model_type in Type_Model.to_list():
    model = utils.get_valid_model(model_type,Type_Model)
else:
    logging.warning("The selected model is not valid. Select a model among 'GAUSSIAN', 'GAUSSIAN_MIXTURE', 'LOGNORMAL', 'WEIBULL', 'GAMMA'.")

image_original = np.array(image_pil)
image_original = image_original.astype('float64')
image_original = np.floor(utils.normalize_datacube_0_1(image_original)*255)


#%% Segmentation

segm = Segmentation(beta=beta, gamma=gamma, thr_memb=thr_memb, 
                    filter_memb_ci_value=None, preproc_level=preproc_level)

segm.set_image(image_original)
segm.set_model_type(model)

if seed_file == -1:
    segm.add_seed_auto(startX=int(step_x/2), startY=int(step_y/2), stepX=step_x, stepY=step_y,  
                    min_value=min_value_seed, max_value=max_value_seed, 
                    max_std=max_std, window_size=window_size)
elif isinstance(seed_file,str):
    if seed_file[-4:]=='.txt':
        seed_path = pathlib.Path(seed_file)
        list_of_seeds_array = np.loadtxt(seed_path, dtype=int)
        list_of_seeds = tuple(map(tuple, list_of_seeds_array))
        for j in range(0,len(list_of_seeds)):
            print(list_of_seeds[j])
            segm.add_seed_xy(list_of_seeds[j])
    else:
        logging.warning("The seed list file is supposed to be a json file.")
        segm.add_seed_auto(startX=int(step_x/2), startY=int(step_y/2), stepX=step_x, stepY=step_y,  
                    min_value=min_value_seed, max_value=max_value_seed, 
                    max_std=max_std, window_size=window_size)
else:
    logging.warning("The seed list file is supposed to be either a json file or set to -1.")
    segm.add_seed_auto(startX=int(step_x/2), startY=int(step_y/2), stepX=step_x, stepY=step_y,  
                    min_value=min_value_seed, max_value=max_value_seed, 
                    max_std=max_std, window_size=window_size)
    
segm.show_seed_location()

segm.run()
result = segm.segmented_image 
seed_img = segm.seed_image


#%% Output result

out_image = result_folder / "segmentation_result.png"
out_original = result_folder / "input_image.png"
out_seed = result_folder / "seed_image.png"
plt.imsave(out_image, result)
plt.imsave(out_original, image_original, cmap="gray")
seed_img.savefig(out_seed)


#%% Close Log
    
logging.info("Close all.")
logging.getLogger().handlers.clear()
logging.shutdown()