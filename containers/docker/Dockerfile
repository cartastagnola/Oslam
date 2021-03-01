# Image: ubuntOslam
# run the image: docker run --rm -it --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 <image-name>

FROM ubuntu:20.04

# Dependencies for glvnd and X11 and nvidia stuffs
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
# non interactive stuff for some install
ENV DEBIAN_FRONTEND noninteractive

# setup a permanent build directory
RUN mkdir /builder
WORKDIR /builder/

# Install minimal prerequisites (Ubuntu 18.04 as reference)
RUN apt -y update
RUN apt install -y software-properties-common
RUN add-apt-repository universe
RUN add-apt-repository multiverse
#RUN DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration

RUN apt update && apt install -y git cmake g++ wget unzip \
                                pkg-config sudo vim-gtk3 libgtk2.0-dev
# SDL 
RUN apt install -y libcanberra-gtk-module libcanberra-gtk3-module libsdl2-dev libsdl2-image-dev libsdl2-gfx-dev gdb

# install raylib
RUN apt install -y libasound2-dev mesa-common-dev libx11-dev libxrandr-dev libxi-dev xorg-dev libgl1-mesa-dev libglu1-mesa-dev

RUN git clone https://github.com/raysan5/raylib.git raylib 
WORKDIR /builder/raylib/src/ 
RUN make PLATFORM=PLATFORM_DESKTOP RAYLIB_LIBTYPE=SHARED 
# PLATFORM_DESKTOP is To make the dynamic shared version. - Recommended 
# need sudo?
RUN make install RAYLIB_LIBTYPE=SHARED # Dynamic shared version.
# END - install raylib


# go into the builder directory in the image
WORKDIR /builder/

######## INSTALL open cv
# Download and unpack sources
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip
RUN unzip opencv.zip
RUN unzip opencv_contrib.zip
    # Create build directory and switch into it \
RUN mkdir -p build
WORKDIR /builder/build
    # Configure \
RUN cmake  -DOPENCV_ENABLE_NONFREE=ON -DOPENCV_GENERATE_PKGCONFIG=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules ../opencv-master
    # Build \
RUN make -j10 
RUN make install

RUN apt install -y libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6 freeglut3 freeglut3-dev

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libglvnd-dev libglvnd-dev \
        libgl1-mesa-dev libgl1-mesa-dev \
        libegl1-mesa-dev libegl1-mesa-dev \
        libgles2-mesa-dev libgles2-mesa-dev

WORKDIR /builder/raylib/examples/
RUN make

WORKDIR /home/
