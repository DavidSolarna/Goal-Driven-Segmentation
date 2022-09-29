FROM continuumio/miniconda3

WORKDIR /app

# Make RUN commands use 'bash --login':
SHELL ["/bin/bash", "--login", "-c"]

# Create the environment:
COPY environment.yml .
RUN conda env create --name "goal_driven_segm" -f environment.yml

# Install gcc
RUN apt-get update
RUN apt-get -y install gcc
RUN apt-get -y install --reinstall build-essential

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "goal_driven_segm", "/bin/bash", "-c"]

# Install PyMaxflow
RUN pip install PyMaxflow

# Copy all the required data to the image
COPY main.py .
COPY segmentation.py .
COPY utils.py .

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "-n", "goal_driven_segm", "python", "main.py", "-c", "data/config.json"]