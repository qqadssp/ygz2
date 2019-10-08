#include "common/Feature.h"
#include "cv/ORBMatcher.h"
#include "common/Frame.h"
#include "common/MapPoint.h"
#include "cv/LKFlow.h"

#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

namespace ygz {

    void ORBMatcher::ComputeStereoMatches(shared_ptr<Frame> f, StereoMethod method) {

        if (method == ORB_BASED) {

        } else if (method == OPTIFLOW_BASED) {
            ComputeStereoMatchesOptiFlow(f);
        } else if (method == OPTIFLOW_CV) {
            ComputeStereoMatchesOptiFlowCV(f);
        }
    }

    void ORBMatcher::ComputeStereoMatchesOptiFlow(shared_ptr<Frame> f, bool only2Dpoints) {

        assert(!f->mFeaturesLeft.empty());
        if (f->mPyramidLeft.empty() || f->mPyramidRight.empty())
            f->ComputeImagePyramid();
        for (int i = 0; i < f->mFeaturesLeft.size(); i++) {
            auto &feat = f->mFeaturesLeft[i];
            if (feat == nullptr)
                continue;
            if (only2Dpoints && feat->mpPoint &&
                feat->mpPoint->Status() == MapPoint::GOOD)    // already have a good point
                continue;
            Vector2f pl = feat->mPixel;
            Vector2f pr = feat->mPixel;
            bool ret = LKFlowSinglePoint(f->mPyramidLeft, f->mPyramidRight, feat->mPixel, pr);

            if (ret) {
                // check the right one
                if (pl[0] < pr[0] || (fabs(pl[1] - pr[1]) > setting::stereoMatchingTolerance)) {
                    continue;
                } else {
                    float disparity = pl[0] - pr[0];
                    if (disparity > 1)    // avoid zero disparity
                        feat->mfInvDepth = disparity / f->mpCam->bf;
                }
            }
        }
    }

    void ORBMatcher::ComputeStereoMatchesOptiFlowCV(shared_ptr<Frame> f) {

        vector<cv::Point2f> leftPts, rightPts;
        vector<shared_ptr<Feature>> validFeats;
        for (auto feat: f->mFeaturesLeft) {
            if (feat->mpPoint == nullptr) {
                leftPts.push_back(cv::Point2f(feat->mPixel[0], feat->mPixel[1]));
                validFeats.push_back(feat);
            }
        }

        if (leftPts.empty())
            return;

        vector<uchar> status;
        vector<float> error;
        cv::calcOpticalFlowPyrLK(f->mImLeft, f->mImRight, leftPts, rightPts, status, error,
                                 cv::Size(21, 21), 3);

        for (size_t i = 0; i < rightPts.size(); i++) {
            if (status[i]) {
                // lk succeed
                shared_ptr<Feature> &feat = validFeats[i];
                cv::Point2f &pl = leftPts[i];
                cv::Point2f &pr = rightPts[i];
                if (pl.x < pr.x || (fabs(pl.y - pr.y) > 5))   // x or y is not right
                    continue;
                float disparity = pl.x - pr.x;
                feat->mfInvDepth = disparity / f->mpCam->bf;
            }
        }
    }

    int ORBMatcher::SearchByDirectProjection(shared_ptr<Frame> F, const std::set<shared_ptr<MapPoint>> &vpMapPoints) {

        // unique_lock<mutex> lock(F->mMutexFeature);
        F->AssignFeaturesToGrid();

        uchar patch[align_patch_area] = {0};

        uchar patch_with_border[(align_patch_size + 2) * (align_patch_size + 2)] = {0};

        int cntSucceed = 0;
        for (const shared_ptr<MapPoint> &mp: vpMapPoints) {
            if (mp == nullptr)
                continue;

            if (mp->mpRefKF.expired()) continue;
            shared_ptr<Frame> kf = mp->mpRefKF.lock();

            if (mp->mbTrackInView) continue;

            if (!F->GetFeaturesInArea(mp->mTrackProjX, mp->mTrackProjY, 20).empty()) {
                // there is already a matched point here
                continue;
            }

            // Try align this map point
            unique_lock<mutex> lock(kf->mMutexFeature);
            size_t idxFeatKF = mp->GetObsFromKF(kf);
            if (idxFeatKF < 0 || idxFeatKF >= kf->mFeaturesLeft.size())
                continue;

            shared_ptr<Feature> refFeat = kf->mFeaturesLeft[idxFeatKF];
            if (refFeat == nullptr)
                continue;

            Eigen::Matrix2d ACR;
            Vector2f px_ref = refFeat->mPixel;
            SE3d pose_ref = SE3d(kf->Rcw(), kf->Tcw());
            SE3d TCR = SE3d(F->Rcw(), F->Tcw()) * pose_ref.inverse();

            this->GetWarpAffineMatrix(kf, F, refFeat, refFeat->mLevel, TCR, ACR);

            int search_level = 0;

            WarpAffine(ACR, kf->mImLeft, px_ref.cast<double>(), 0, kf, 0, align_halfpatch_size + 1,
                       patch_with_border);

            // remove the boarder
            uint8_t *ref_patch_ptr = patch;
            for (int y = 1; y < align_patch_size + 1; ++y, ref_patch_ptr += align_patch_size) {
                uint8_t *ref_patch_border_ptr = patch_with_border + y * (align_patch_size + 2) + 1;
                for (int x = 0; x < align_patch_size; ++x)
                    ref_patch_ptr[x] = ref_patch_border_ptr[x];
            }

            Vector2f px_curr(mp->mTrackProjX, mp->mTrackProjY);
            Vector2f px_scaled = px_curr * setting::invScaleFactors[refFeat->mLevel];

            bool success = Align2D(F->mImLeft, patch_with_border, patch, 10, px_scaled);
            px_curr = px_scaled * setting::scaleFactors[search_level];

            if (success) {
                // Create a feature in current
                shared_ptr<Feature> feat(new Feature);
                feat->mPixel = px_curr;
                feat->mpPoint = mp;
                feat->mLevel = search_level;
                F->mFeaturesLeft.push_back(feat);
                cntSucceed++;
            }
        }

        return cntSucceed;

    }

    void ORBMatcher::GetWarpAffineMatrix(shared_ptr<Frame> ref, shared_ptr<Frame> curr, const shared_ptr<Feature> feat,
                                         int level, const SE3d &TCR, Eigen::Matrix2d &ACR) {

        float depth = 1.0 / feat->mfInvDepth;
        Vector3d pt_ref = ref->Pixel2Camera(feat->mPixel.cast<double>(), depth);

        const Vector3d pt_du_ref = ref->Pixel2Camera(
                feat->mPixel.cast<double>() + Vector2d(align_halfpatch_size, 0) * (double) setting::scaleFactors[level],
                depth);
        const Vector3d pt_dv_ref = ref->Pixel2Camera(
                feat->mPixel.cast<double>() + Vector2d(0, align_halfpatch_size) * (double) setting::scaleFactors[level],
                depth);

        const Vector2d px_cur = curr->World2Pixel(pt_ref, TCR);
        const Vector2d px_du = curr->World2Pixel(pt_du_ref, TCR);
        const Vector2d px_dv = curr->World2Pixel(pt_dv_ref, TCR);

        ACR.col(0) = (px_du - px_cur) /
                     align_halfpatch_size;
        ACR.col(1) = (px_dv - px_cur) /
                     align_halfpatch_size;
    }

    void ORBMatcher::WarpAffine(
            const Matrix2d &ACR, const Mat &img_ref,
            const Vector2d &px_ref, const int &level_ref, const shared_ptr<Frame> ref,
            const int &search_level, const int &half_patch_size, uint8_t *patch) {

        const int patch_size = half_patch_size * 2;
        const Eigen::Matrix2d ARC = ACR.inverse();

        // Affine warp
        uint8_t *patch_ptr = patch;
        const Vector2d px_ref_pyr = px_ref / setting::scaleFactors[level_ref];
        for (int y = 0; y < patch_size; y++) {
            for (int x = 0; x < patch_size; x++, ++patch_ptr) {
                Vector2d px_patch(x - half_patch_size, y - half_patch_size);
                px_patch *= setting::scaleFactors[search_level];
                const Vector2d px(ARC * px_patch + px_ref_pyr);
                if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1) {
                    *patch_ptr = 0;
                } else {
                    *patch_ptr = GetBilateralInterpUchar(px[0], px[1], img_ref);
                }
            }
        }
    }

}
