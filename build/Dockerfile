FROM python:3.9-bullseye

RUN python -m pip install --upgrade pip

RUN apt install -y make wget unzip

RUN wget -O /tmp/corretto.tar.gz --progress=bar:force:noscroll "https://corretto.aws/downloads/latest/amazon-corretto-21-x64-linux-jdk.tar.gz"
RUN wget -O /tmp/slc.zip --progress=bar:force:noscroll "https://www.silabs.com/documents/login/software/slc_cli_linux.zip"
RUN wget -O /tmp/gcc-arm-none-eabi.tar.bz2 --progress=bar:force:noscroll "https://developer.arm.com/-/media/Files/downloads/gnu-rm/10.3-2021.10/gcc-arm-none-eabi-10.3-2021.10-x86_64-linux.tar.bz2?rev=78196d3461ba4c9089a67b5f33edf82a&hash=5631ACEF1F8F237389F14B41566964EC"
RUN wget -O /tmp/git-lfs.tar.gz --progress=bar:force:noscroll "https://github.com/git-lfs/git-lfs/releases/download/v3.3.0/git-lfs-linux-amd64-v3.3.0.tar.gz"
RUN wget -O /tmp/cmake.tar.gz --progress=bar:force:noscroll "https://cmake.org/files/v3.25/cmake-3.25.1-linux-x86_64.tar.gz"
RUN git clone https://github.com/SiliconLabs/gecko_sdk /opt/gecko_sdk

WORKDIR /opt
RUN ls /tmp
RUN tar -xf /tmp/corretto.tar.gz
RUN unzip /tmp/slc.zip
RUN tar -xjvf /tmp/gcc-arm-none-eabi.tar.bz2
RUN tar -xf /tmp/git-lfs.tar.gz
RUN tar -xf /tmp/cmake.tar.gz

RUN CORRETTO_DIRNAME="$(ls . | grep corretto | head -n1)"; \
    SLC_DIRNAME="$(ls . | grep slc | head -n1)"; \
    ARM_GCC_DIRNAME="$(ls . | grep gcc-arm-none-eabi | head -n1)"; \
    GIT_LFS_DIRNAME="$(ls . | grep lfs | head -n1)"; \
    CMAKE_DIRNAME="$(ls . | grep cmake | head -n1)"; \
    mv $CORRETTO_DIRNAME corretto; \
    mv $SLC_DIRNAME slc; \
    mv $ARM_GCC_DIRNAME gcc-arm-none-eabi; \
    mv $GIT_LFS_DIRNAME git-lfs; \
    mv $CMAKE_DIRNAME cmake

ENV PATH="/opt/corretto/bin:$PATH"
ENV PATH="/opt/slc:$PATH"
ENV PATH="/opt/gcc-arm-none-eabi/bin:$PATH"
ENV PATH="/opt/cmake/bin:$PATH"

RUN cd git-lfs && ./install.sh

RUN cd gecko_sdk && git checkout v4.2.0
RUN cd gecko_sdk && git lfs install && git lfs pull --include="lib*.a"

RUN chmod +x /opt/slc/slc

RUN slc configuration --sdk /opt/gecko_sdk
RUN slc signature trust --sdk /opt/gecko_sdk/
RUN slc configuration --gcc-toolchain /opt/gcc-arm-none-eabi

WORKDIR /tmp
RUN rm -r *
RUN mkdir -p /home/build

WORKDIR /home/build
COPY tests tests
COPY application application
COPY component component
COPY machine_learning_applications.slce machine_learning_applications.slce
COPY machine_learning_applications.slsdkext machine_learning_applications.slsdkext
COPY demos.xml demos.xml
COPY templates.xml templates.xml