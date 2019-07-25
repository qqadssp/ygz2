#ifndef YGZ_ORB_EXTRACTOR_H
#define YGZ_ORB_EXTRACTOR_H

#include "common/Settings.h"
#include "common/NumTypes.h"

#include <opencv2/core/core.hpp>
#include <list>

namespace ygz {

    struct Frame;
    struct Feature;

    class ORBExtractor {
    public:

        typedef enum {
            FAST_MULTI_LEVEL,
            FAST_SINGLE_LEVEL,
            ORB_SLAM2,
            OPENCV_ORB,
            OPENCV_GFTT
        } KeyPointMethod;

        ORBExtractor(const KeyPointMethod &method);

        void Detect(
                shared_ptr<Frame> frame,
                bool leftEye = true,
                bool computeRotAndDesc = true
        );

    private:

        void ComputeKeyPointsFastSingleLevel(std::vector<shared_ptr<Feature >> &allKeypoints, const cv::Mat &image);

        inline float ShiTomasiScore(const cv::Mat &img, const int &u, const int &v) const {
            float dXX = 0.0;
            float dYY = 0.0;
            float dXY = 0.0;
            const int halfbox_size = 4;
            const int box_size = 2 * halfbox_size;
            const int box_area = box_size * box_size;
            const int x_min = u - halfbox_size;
            const int x_max = u + halfbox_size;
            const int y_min = v - halfbox_size;
            const int y_max = v + halfbox_size;

            if (x_min < 1 || x_max >= img.cols - 1 || y_min < 1 || y_max >= img.rows - 1)
                return 0.0; // patch is too close to the boundary

            const int stride = img.step.p[0];
            for (int y = y_min; y < y_max; ++y) {
                const uint8_t *ptr_left = img.data + stride * y + x_min - 1;
                const uint8_t *ptr_right = img.data + stride * y + x_min + 1;
                const uint8_t *ptr_top = img.data + stride * (y - 1) + x_min;
                const uint8_t *ptr_bottom = img.data + stride * (y + 1) + x_min;
                for (int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom) {
                    float dx = *ptr_right - *ptr_left;
                    float dy = *ptr_bottom - *ptr_top;
                    dXX += dx * dx;
                    dYY += dy * dy;
                    dXY += dx * dy;
                }
            }

            // Find and return smaller eigenvalue:
            dXX = dXX / (2.0 * box_area);
            dYY = dYY / (2.0 * box_area);
            dXY = dXY / (2.0 * box_area);
            return 0.5 * (dXX + dYY - sqrt((dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY)));
        }

        // data
        std::vector<int> mnFeaturesPerLevel;

        shared_ptr<Frame> mpFrame = nullptr;

        int mnGridSize;
        int mnGridCols;
        int mnGridRows;

        KeyPointMethod mMethod;

        cv::Mat mOccupancy;

        std::vector<bool> mvbGridOccupancy;

        // options
        bool mbComputeRotAndDesc = true;
    };
}

#endif
