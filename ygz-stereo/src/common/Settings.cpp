#include "common/Settings.h"

namespace ygz {

    namespace setting {

        size_t numPyramid = 5;
        float scalePyramid = 2;
        float *scaleFactors = nullptr;
        float *invScaleFactors = nullptr;
        float *levelSigma2 = nullptr;
        float *invLevelSigma2 = nullptr;

        Eigen::Matrix3d Rbc = [] {
            Eigen::Matrix3d mat;
            mat << 0.0148655429818, -0.999880929698, 0.00414029679422,
                    0.999557249008, 0.0149672133247, 0.025715529948,
                    -0.0257744366974, 0.00375618835797, 0.999660727178;
            return mat;
        }();
        Eigen::Vector3d tbc(-0.0216401454975, -0.064676986768, 0.00981073058949);
        Sophus::SE3d TBC(Rbc, tbc);

        int FRAME_GRID_SIZE = 40;
        int FRAME_GRID_ROWS = 0;
        int FRAME_GRID_COLS = 0;

        Eigen::Vector3d biasAccePrior(-0.025, 0.136, 0.075);

        int imageWidth = 752;
        int imageHeight = 480;

        int boarder = 20;

        int initTHFAST = 20;
        int minTHFAST = 5;
        int extractFeatures = 100;

        float GridElementWidthInv;
        float GridElementHeightInv;

        bool trackerUseHistBalance = true;

        int numBackendKeyframes = 10;
        double keyframeTimeGapInit = 0.5; 
        double keyframeTimeGapTracking = 0.5;

        void initSettings() {
            // compute the scale factors in each frame
            scaleFactors = new float[numPyramid];
            levelSigma2 = new float[numPyramid];
            scaleFactors[0] = 1.0f;
            levelSigma2[0] = 1.0f;
            for (size_t i = 1; i < numPyramid; i++) {
                scaleFactors[i] = scaleFactors[i - 1] * scalePyramid;
                levelSigma2[i] = scaleFactors[i] * scaleFactors[i];
            }

            invScaleFactors = new float[numPyramid];
            invLevelSigma2 = new float[numPyramid];
            for (size_t i = 0; i < numPyramid; i++) {
                invScaleFactors[i] = 1.0f / scaleFactors[i];
                invLevelSigma2[i] = 1.0f / levelSigma2[i];
            }

            FRAME_GRID_ROWS = ceil(imageHeight / FRAME_GRID_SIZE);
            FRAME_GRID_COLS = ceil(imageWidth / FRAME_GRID_SIZE);
            GridElementWidthInv = 1.0 / float(FRAME_GRID_SIZE);
            GridElementHeightInv = 1.0 / float(FRAME_GRID_SIZE);
        }

        void destroySettings() {
            delete[] scaleFactors;
            delete[] levelSigma2;
            delete[] invScaleFactors;
            delete[] invLevelSigma2;
        }

    }
}
