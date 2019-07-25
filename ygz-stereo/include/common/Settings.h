#ifndef YGZ_SETTINGS_H_
#define YGZ_SETTINGS_H_

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <stdexcept>
#include <glog/logging.h>

#include "se3.hpp"

namespace ygz {

    namespace setting {

        void initSettings();

        void destroySettings();

        extern size_t numPyramid;

        extern float scalePyramid;
        extern float *scaleFactors;
        extern float *invScaleFactors;
        extern float *levelSigma2;
        extern float *invLevelSigma2;

        extern int FRAME_GRID_SIZE;     // default: 20
        extern int FRAME_GRID_ROWS;     // 480/20
        extern int FRAME_GRID_COLS;     // 752/20

        extern float GridElementWidthInv;
        extern float GridElementHeightInv;

        extern Sophus::SE3d TBC;

        extern int imageWidth;
        extern int imageHeight;

        extern int boarder;

        extern int initTHFAST;
        extern int minTHFAST;
        extern int extractFeatures;
        const float minShiTomasiScore = 20.0;
        const float featureDistanceGFTT = 30.0;   // in GFTT we fix the feature distance (in pixels), default 30 px

        const int PATCH_SIZE = 31;
        const int HALF_PATCH_SIZE = 15;
        const int EDGE_THRESHOLD = 19;

        const float stereoMatchingTolerance = 5.0;

        /*** Tracking ***/
        const int minTrackLastFrameFeatures = 10;
        const int minTrackRefKFFeatures = 10;
        const int minPoseOptimizationInliers = 10;
        const int minTrackLocalMapInliers = 10;

        const float minPointDis = 0.5;
        const float maxPointDis = 30;
        const bool useTempMapPoints = true;
        
        

        extern double keyframeTimeGapInit;
        extern double keyframeTimeGapTracking;

        const size_t minStereoInitFeatures = 50;
        const size_t minValidInitFeatures = 20;
        const int minInitKFs = 5;
        extern bool trackerUseHistBalance;


        extern int numBackendKeyframes/* = 5*/;

        const float minNewMapPointInvD = 1 / maxPointDis;  // D = 20m
        const float maxNewMapPointInvD = 1 / minPointDis;   // D = 0.5m

        const float cameraSize = 0.1;

    }
}

#endif
