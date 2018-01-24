# Parts of this Dockerfile were taken from other sources:
#
# For compiling dlib, a key face_recogition dependency, I borrowed from: 
#     https://github.com/ageitgey/face_recognition/blob/master/Dockerfile
#
# For compiling OpenCV, I relied on:
#     https://github.com/janza/docker-python3-opencv/blob/master/Dockerfile

FROM python:3.4-slim

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjasper-dev \
    libjpeg-dev \
    liblapack-dev \
    libpng-dev \
    libpq-dev \
    libswscale-dev \
    libtbb2 \  
    libtbb-dev \
    libtiff-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    unzip \
    yasm \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.7' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS

RUN pip install numpy==1.14.0

WORKDIR /
ENV OPENCV_VERSION="3.4.0"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
RUN unzip ${OPENCV_VERSION}.zip
RUN mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=OFF \
  -DENABLE_AVX=ON \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python3.4 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python3.4) \
  -DPYTHON_INCLUDE_DIR=$(python3.4 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python3.4 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
&& make install
#&& rm /${OPENCV_VERSION}.zip \
#&& rm -r /opencv-${OPENCV_VERSION}

RUN pip install face_recognition==1.0.0

COPY . /root/sketch_face

CMD cd /root/sketch_face && \
    python sketch_face.py
