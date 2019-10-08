#include "cv/LKFlow.h"
#include "cv/Align.h"
#include "common/Feature.h"
#include "common/MapPoint.h"

#include <opencv2/video/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace ygz {

    bool LKFlowSinglePoint(
            const vector<Mat> &pyramid1,
            const vector<Mat> &pyramid2,
            const Vector2f &pixel1,
            Vector2f &pixel2
    ) {

        uchar patch[align_patch_area] = {0};

        uchar patch_with_border[(align_patch_size + 2) * (align_patch_size + 2)] = {0};

        // from coarse to fine
        Vector2f refPixel = pixel1;
        Vector2f trackedPos = pixel2;
        bool success = true;

        for (int lvl = setting::numPyramid - 1; lvl >= 2; lvl--) {

            float scale = setting::scaleFactors[lvl];
            float invScale = setting::invScaleFactors[lvl];

            Vector2f posLvl = trackedPos * invScale;
            Vector2f refPixelLvl = refPixel * invScale;
            const cv::Mat &img_ref = pyramid1[lvl];

            // copy the patch with boarder
            uchar *patch_ptr = patch_with_border;
            const int ix = floor(refPixelLvl[0]);
            const int iy = floor(refPixelLvl[1]);
            const float xx = refPixelLvl[0] - ix;
            const float yy = refPixelLvl[1] - iy;

            for (int y = 0; y < align_patch_size + 2; y++) {
                for (int x = 0; x < align_patch_size + 2; x++, ++patch_ptr) {
                    const int dx = x - align_halfpatch_size - 1;
                    const int dy = y - align_halfpatch_size - 1;
                    const int iix = ix + dx;
                    const int iiy = iy + dy;
                    if (iix < 0 || iiy < 0 || iix >= img_ref.cols - 1 || iiy >= img_ref.rows - 1) {
                        *patch_ptr = 0;
                    } else {
                        uchar *data = img_ref.data + iiy * img_ref.step + iix;
                        *patch_ptr =
                                (1 - xx) * (1 - yy) * data[0] +
                                xx * (1 - yy) * data[1] +
                                (1 - xx) * yy * data[img_ref.step] +
                                xx * yy * data[img_ref.step + 1];
                    }
                }
            }

            // remove the boarder
            uint8_t *ref_patch_ptr = patch;
            for (int y = 1; y < align_patch_size + 1; ++y, ref_patch_ptr += align_patch_size) {
                uint8_t *ref_patch_border_ptr = patch_with_border + y * (align_patch_size + 2) + 1;
                for (int x = 0; x < align_patch_size; ++x)
                    ref_patch_ptr[x] = ref_patch_border_ptr[x];
            }

            bool ret = Align2D(
                    pyramid2[lvl],
                    patch_with_border,
                    patch,
                    30,
                    posLvl
            );

            if (lvl == 2)
                success = ret;

            // set the tracked pos
            trackedPos = posLvl * scale;

            if (trackedPos[0] < setting::boarder || trackedPos[0] >= setting::imageWidth - setting::boarder ||
                trackedPos[1] < setting::boarder || trackedPos[1] >= setting::imageHeight - setting::boarder) {
                success = false;
                break;
            }
        }

        if (success) {
            // copy the results
            pixel2 = trackedPos;
        }
        return success;
    }

    int LKFlowCV(
            const shared_ptr<Frame> ref,
            const shared_ptr<Frame> current,
            VecVector2f &refPts,
            VecVector2f &trackedPts
    ) {

        if (refPts.size() == 0)
            return 0;

        vector<cv::Point2f> refPx, currPts;
        for (auto &px:refPts) {
            refPx.push_back(cv::Point2f(px[0], px[1]));
        }
        for (Vector2f &v: trackedPts) {
            currPts.push_back(cv::Point2f(v[0], v[1]));
        }

        vector<uchar> status;
        vector<float> err;

        cv::calcOpticalFlowPyrLK(ref->mImLeft, current->mImLeft, refPx, currPts, status, err,
                                 cv::Size(21, 21), 3,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        // reject with F matrix
        cv::findFundamentalMat(refPx, currPts, cv::FM_RANSAC, 3.0, 0.99, status);

        int successPts = 0;
        for (int i = 0; i < currPts.size(); i++) {
            if (status[i] && (currPts[i].x > setting::boarder && currPts[i].y > setting::boarder &&
                              currPts[i].x < setting::imageWidth - setting::boarder &&
                              currPts[i].y < setting::imageHeight - setting::boarder)) {
                // succeed
                // trackedPts.push_back(Vector2f(currPts[i].x, currPts[i].y));
                trackedPts[i] = Vector2f(currPts[i].x, currPts[i].y);
                successPts++;
            } else {
                // failed
                // trackedPts.push_back(Vector2f(-1, -1));
                trackedPts[i] = Vector2f(-1, -1);
            }
        }
        return successPts;
    }

}
