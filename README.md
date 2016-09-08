## SaliencyELD

Source code for our journal submission "ELD-Net: An efficient deep learning architecture for accurate saliency detection" which is an extended version of our CVPR 2016 paper "Deep Saliency with Encoded Low level Distance Map and High Level Features" by [Gayoung Lee](https://sites.google.com/site/gylee1103/), [Yu-Wing Tai](http://www.gdriv.es/yuwing) and [Junmo Kim](https://sites.google.com/site/siitkaist/professor). **([ArXiv paper link for CVPR paper] (http://arxiv.org/abs/1604.05495))**

![Image of our model](./figs/model_pic.png)

Acknowledgement : Our code uses various libraries: [Caffe](http://github.com/BVLC/caffe), [OpenCV](http://www.opencv.org) and [Boost](http://www.boost.org). We utilize [gSLICr](https://github.com/carlren/gSLICr) with minor modification for superpixel generation.

## Usage
1. **Dependencies**
    0. OS : Our code is tested on Ubuntu 14.04
    0. CMake : Tested on CMake 2.8.12
    0. Caffe : Caffe that we used is contained in this repository.
    0. OpenCV 3.0 : We used OpenCV 3.0, but the code may work with OpenCV 2.4.X version.
    0. g++ : Our code was tested with g++ 4.9.2.
    0. Boost : Tested on Boost 1.46

2. **Installation**
    0. Download our pretrained model from [(Link1)][https://www.dropbox.com/s/itv8donc7zifk8s/ELDNet_80K.caffemodel?dl=1] or [(Link2)][http://pan.baidu.com/s/1c27LKHy]
       
        **NOTE: If you cannot download our ELD model from dropbox, please download it from [this Baidu link](http://pan.baidu.com/s/1jI94TAu).**

        ```shell
        sh get_models.sh
        ```

    0. Build Caffe in the project folder using CMake:

        ```shell
        cd $(PROJECT_ROOT)/caffe/
        mkdir build
        cd build/
        cmake ..
        make -j4
        ```

    0. Change library paths in $(PROJECT_ROOT)/CMakeLists.txt for your custom environment and build our test code:

        ```shell
        cd $(PROJECT_ROOT)
        edit CMakeList.txt
        cmake .
        make
        ```

    0. Run the executable file with four arguments (arg1: prototxt path, arg2: model binary path, arg3: images directory, arg4: results directory)

        ```shell
        ./get_results ELDNet_test.prototxt ELDNet_80K.caffemodel test_images/ results/
        ```

    0. The results will be generated in the result directory.

## Results of datasets used in the paper

![visualization](./figs/visualization.png)

We provide our results of benchmark datasets used in the paper for convenience. Link1 is the link using dropbox and link2 is using baidu.

ECSSD results [(link1)](https://www.dropbox.com/s/zets1xsne570bgl/ECSSD_ELD.zip?dl=1) [(link2)](http://pan.baidu.com/s/1i4QslAP) (ECSSD dataset [site](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html))

PASCAL-S results [(link1)](https://www.dropbox.com/s/8f3d37h9rgx0b7i/PASCAL_ELD.zip?dl=1) [(link2)](http://pan.baidu.com/s/1boJcVx1) (PASCAL-S dataset [site](http://cbi.gatech.edu/salobj/))

DUT-OMRON results [(link1)](https://www.dropbox.com/s/s83aw290ndxxfub/DUTOMRON_ELD.zip?dl=1) [(link2)](http://pan.baidu.com/s/1pKX760z) (DUT-OMRON dataset [site](http://202.118.75.4/lu/DUT-OMRON/index.htm))

THUR15K results [(link1)](https://www.dropbox.com/s/zqtvo62lpdv7wc5/THUR15K_ELD.zip?dl=1) [(link2)](http://pan.baidu.com/s/1nuXB96T) (THUR15K dastaset [site](http://mmcheng.net/gsal/))

HKU-IS results [(link1)](https://www.dropbox.com/s/rchzoze80gg7116/HKUIS_ELD.zip?dl=1) [(link2)](http://pan.baidu.com/s/1geW1RQb) (ASD dataset [site](http://ivrlwww.epfl.ch/supplementary_material/RK_CVPR09/index))

## Citing our work
Please kindly cite our work if it helps your research:

    @inproceedings{lee2016saliency,
        title = {Deep Saliency with Encoded Low level Distance Map and High Level Features},
        author={Gayoung, Lee and Yu-Wing, Tai and Junmo, Kim},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2016}
    }

