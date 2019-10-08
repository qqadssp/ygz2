# ygz2

stereoVO(ygz-stereo) and stereoVIO(ygz-stereoVIO). This repo is a simplied version of [ygz-stereo-inertial](https://github.com/gaoxiang12/ygz-stereo-inertial), optical flow is used to track between left image of different frames and between left-right images in one stereo image frame. Stereo image is used to triangulate keypoint then create mappoints when keyframe is created. G2O is used to optimize keyframe poses and mappoints in the backend.

### Environment

 - Ubuntu 18.04
 - Cmake 3.10.2  
 - g++ 7.4.0  
 - OpenCV 4.1.0  
 - Pangolin
 - Eigen3
 - g2o 
 - SuiteSparse

### Build and run

1. Download this repo and build

```
    $git clone git@github.com:qqadssp/ygz2  

    $cd ygz2/ygz-stereo/Thirdparty/fast  
    $mkdir build  
    $cd build  
    $cmake ..  
    $make  

    $cd ..
    $cd ..
    $cd ..
    $mkdir build  
    $cd build  
    $cmake ..  
    $make  
```

2. Download Eurco dataset and modify the path in `./data/EuroStereo.yaml`  

3. Run  

```
    $cd ygz2/ygz-stereo  
    $./bin/ygz_stereo ./data/EuroStereo.yaml  
```

4. For stereo inertial VO, it is the same as stereo VO.

### Demo
