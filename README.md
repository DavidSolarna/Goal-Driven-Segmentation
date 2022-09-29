# Goal-Driven-Segmentation

When using this code, please cite the corresponding paper:

**Marco Trombini, David Solarna, Gabriele Moser, Silvana Dellepiane, "A goal-driven unsupervised image segmentation method combining graph-based processing and Markov random fields", Pattern Recognition, 2022.** 

## Run

The code is written in python and it has been developed and tested using the Anaconda virtual environment contained in the **environment.yml** file available in the repository.

After the environment is created, install the PyMaxflow package from pip with the command:

**pip install PyMaxflow**

The python code can be run using the command:

**python main.py -c config.json"**

where the config.json file contains the configuration parameters.

## Configuration file

Below you can find the list of parameters in the json configuration file. Some explanation is given here. Nevertheless, for the best understanding, please refer to the corresponding paper.

**image_path**
Path to the input image (from the main.py directory). If the path is empty (i.e., ""), a test image is generated and the code is run.
Examples are: 
1. "input/image.png" for the image.png file located in the input folder; 
2. "" for the test image.

**min_value_seed**
**max_value_seed**
The thresholds used in one of the predicates for the automatic positioning of the initial seeds.

**beta**
**gamma**
The two parameters defined within the Markovian energy function.

**thr_memb**
Threshold for performing the graph cut at the end of the cost computation step. Between 0 and 1.

**preproc_level**
The amount of preprocessing to apply to the image before the graph-based processing. Generally, a value between 0 (no preprocessing) and 5 is enough.

**start_x**
**start_y**
**end_x**
**end_y**
**step_x**
**step_y**
The parameters define the grid on which the initial seed placement is performed. 
If start_x, start_y, end_x, and end_y are set to null, then the grid starts and ends at the border of the image (the whole image is used).
Conditions are:
1. start_x >= 0 
2. end_x <= number of columns of the input image
3. start_x < end_x
4. start_y >= 0
5. end_y <= number of rows of the input image
6. start_y < end_y 

**max_std**
**window_size**
Parameters used in one of the predicates for the automatic positioning of the initial seeds.

**model_type**
Type of model used used in the parametric models estimation steps.
Possible values are:
1. GAUSSIAN
2. GAUSSIAN_MIXTURE
3. LOGNORMAL
4. WEIBULL
5. GAMMA

**seed_file**
The path to a txt file that contains the position of the seeds to be used instead of the automatic placement (if the user wants to set them by hands).
If set to -1, then the automatic seed placement is performed.
If set to a file, then the file must contain, on each row, the pair of X and Y coordinates of the seeds. An example is provided within the data folder.

## Docker

It is possible to create the docker container with the following command (note the -d is used to run it in background):

**docker-compose up -d**

Then, to run the code passing the configuration parameters via the config.json file, type:

**docker exec -it goal_driven_segm_cont conda run -n goal_driven_segm python main.py -c config.json**
