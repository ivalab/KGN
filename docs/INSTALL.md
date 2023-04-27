# Installation

Slighty modified from the CenterNet installation guide. Install the [DCNv2_latest](https://github.com/jinfagang/DCNv2_latest) externally instead of within the repository. Tested on Ubuntu 20.04 with Python3.8 and [PyTorch]((http://pytorch.org/)) 1.13.1. NVIDIA GPU(s) is(are) needed for both training and testing.


After install Anaconda:

0. [Optional but recommended] create a new environment. 

    ~~~
    conda create -n kgn python=3.8
    ~~~
    And activate the environment.
    
    ~~~
    conda activate kgn
    ~~~

1. Install pytorch. For pytorch 1.13.1:

    ~~~
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    ~~~
    
    Manually open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`. ??
    
    For the pytorch v0.4.0 and v0.4.1, disable cudnn batch normalization(Due to [this issue](https://github.com/xingyizhou/pytorch-pose-hg-3d/issues/16)).
    
     ~~~
    # PYTORCH=/path/to/pytorch # usually ~/anaconda3/envs/CenterNet/lib/python3.6/site-packages/
    # for pytorch v0.4.0
    sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
    # for pytorch v0.4.1
    sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
     ~~~
    
3. Clone this repo:

    ~~~
    KGN_ROOT=/path/to/clone/kgn
    git clone https://github.com/ $KGN_ROOT
    ~~~


4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
5. Install and Compile Deformable Convolutional Networks V2 from [DCNv2_latest](https://github.com/jinfagang/DCNv2_latest). 

    ~~~
    # DCNv2=/path/to/clone/DCNv2
    git clone https://github.com/jinfagang/DCNv2_latest $DCNv2
    cd $DCNv2
    python setup.py install --user
    ~~~
    
6. [Optional, only required if you are using extremenet or multi-scale testing] Compile NMS.

    ~~~
    cd ./src/lib/external
    make
    ~~~
