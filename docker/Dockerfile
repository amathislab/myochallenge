FROM python:3.8-slim-buster

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    patchelf

RUN apt-get install -y --no-install-recommends -o APT::Immediate-Configure=false gcc g++ git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


ENV PYTHONUNBUFFERED 1

ADD docker/requirements.txt /

RUN pip install --upgrade pip
RUN pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt
RUN pip install myosuite==1.2.*
RUN pip install grpcio
RUN pip install grpcio-tools
RUN pip install --upgrade pickle5 cloudpickle
RUN export PYTHONPATH="./utils/:$PYTHONPATH"
RUN export PYTHONPATH="./agent/:$PYTHONPATH"


RUN mkdir /src
RUN mkdir /data
ADD ./data /data
ADD ./src /src
ADD ./definitions.py /

RUN cd src

CMD [ "python", "main_pose_elbow.py" ]