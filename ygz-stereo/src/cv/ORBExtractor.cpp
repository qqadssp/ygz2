#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fast/fast.h>

#include "common/Frame.h"
#include "common/Feature.h"
#include "cv/ORBExtractor.h"

using namespace cv;
using namespace std;

namespace ygz {

    ORBExtractor::ORBExtractor(const KeyPointMethod &method) : mMethod(method) {
        mnFeaturesPerLevel.resize(setting::numPyramid);
        float factor = 1.0f / setting::scaleFactors[1];
        float nDesiredFeaturesPerScale = setting::extractFeatures * (1 - factor) /
                                         (1 - (float) pow((double) factor, (double) setting::numPyramid));
        int sumFeatures = 0;
        for (int level = 0; level < setting::numPyramid - 1; level++) {
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[setting::numPyramid - 1] = std::max(int(setting::numPyramid - sumFeatures), int(0));

        const int npoints = 512;

        mnGridSize = -1;
    }

    const float factorPI = (float) (CV_PI / 180.f);


    void ORBExtractor::Detect(shared_ptr<Frame> frame, bool leftEye, bool computeRotAndDesc) {
        mpFrame = frame;
        mbComputeRotAndDesc = computeRotAndDesc;

        vector<vector<shared_ptr<Feature>>> allKeypoints;
        allKeypoints.resize(setting::numPyramid);
            // SINGLE LEVEL
        vector<shared_ptr<Feature> > single_level_features;
        LOG(INFO) << "Calling fast single level" << endl;
        if (leftEye) {
            ComputeKeyPointsFastSingleLevel(single_level_features, frame->mImLeft);
        } else {
            ComputeKeyPointsFastSingleLevel(single_level_features, frame->mImRight);
        }
        allKeypoints[0] = single_level_features;

        if (mbComputeRotAndDesc) {

        } else {
            for (size_t level = 0; level < setting::numPyramid; level++) {
                vector<shared_ptr<Feature>> &features = allKeypoints[level];
                if (features.empty())
                    continue;
                for (auto f: features) {
                    f->mPixel *= setting::scaleFactors[level];
                    if (leftEye) {
                        frame->mFeaturesLeft.push_back(f);
                    } else {
                        frame->mFeaturesRight.push_back(f);
                    }
                }
            }
        }
    }

    void ORBExtractor::ComputeKeyPointsFastSingleLevel(std::vector<shared_ptr<Feature>> &allKeypoints,
                                                       const cv::Mat &image) {
        mpFrame->AssignFeaturesToGrid();
        allKeypoints.reserve(setting::extractFeatures * 2);
        int cnt = 0;   //

        for (int i = 1; i < setting::FRAME_GRID_ROWS - 1; i++) {
            for (int j = 1; j < setting::FRAME_GRID_COLS - 1; j++) {
                if (mpFrame->mGrid[i * setting::FRAME_GRID_COLS + j].empty()) {

                    const uchar *data = image.ptr<uchar>(i * setting::FRAME_GRID_SIZE) + j * setting::FRAME_GRID_SIZE;
                    vector<fast::fast_xy> fast_corners;
#ifdef __SSE2__
                    fast::fast_corner_detect_10_sse2(data, setting::FRAME_GRID_SIZE, setting::FRAME_GRID_SIZE,
                                                     setting::imageWidth, setting::initTHFAST, fast_corners);
#else
                    fast::fast_corner_detect_10(data, setting::FRAME_GRID_SIZE, setting::FRAME_GRID_SIZE, setting::imageWidth, setting::initTHFAST, fast_corners);
#endif
                    if (fast_corners.empty()) {
                        // try lower threshold
#ifdef __SSE2__
                        fast::fast_corner_detect_10_sse2(data, setting::FRAME_GRID_SIZE, setting::FRAME_GRID_SIZE,
                                                         setting::imageWidth, setting::minTHFAST, fast_corners);
#else
                        fast::fast_corner_detect_10(data, setting::FRAME_GRID_SIZE, setting::FRAME_GRID_SIZE, setting::imageWidth, setting::minTHFAST, fast_corners);
#endif
                    }

                    if (fast_corners.empty())
                        continue;

                    // find the best one and insert as a feature
                    int x_start = j * setting::FRAME_GRID_SIZE;
                    int y_start = i * setting::FRAME_GRID_SIZE;

                    // sort the corners according to shi-tomasi score
                    int idxBest = 0;
                    float scoreBest = -1;
                    for (int k = 0; k < fast_corners.size(); k++) {
                        fast::fast_xy &xy = fast_corners[k];
                        xy.x += x_start;
                        xy.y += y_start;

                        if (xy.x < setting::boarder || xy.y < setting::boarder ||
                            xy.x >= setting::imageWidth - setting::boarder ||
                            xy.y >= setting::imageHeight - setting::boarder) {
                            continue;
                        }

                        float score = ShiTomasiScore(image, xy.x, xy.y);
                        if (/* score > setting::minShiTomasiScore && */ score > scoreBest) {
                            scoreBest = score;
                            idxBest = k;
                        }
                    }

                    if (scoreBest < 0)
                        continue;

                    fast::fast_xy &best = fast_corners[idxBest];
                    shared_ptr<Feature> feature(new Feature());
                    feature->mPixel = Vector2f(best.x, best.y);
                    allKeypoints.push_back(feature);
                }
            }
        }

    }
}
