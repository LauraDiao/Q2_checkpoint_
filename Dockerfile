FROM ucsdets/datascience-notebook:2021.2-stable

#change to root to install packages
USER root

# update base system and install packages with apt-get
# RUN apt-get update && \
# apt-get upgrade -y 

# 3) install packages using notebook user
USER jovyan

#install following libraries with conda or pip: 

#Only use pip after conda, RUN conda install --yes <package1> <package2>
#pip install --no-cache-dir <package>
RUN pip install --no-cache-dir numpy \
                               scipy \
                               pandas \
                               pyyaml \
                               sklearn \
                               notebook \
                               matplotlib \
                               seaborn 