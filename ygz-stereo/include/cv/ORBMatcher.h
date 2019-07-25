#ifndef YGZ_ORB_MATCHER_H
#define YGZ_ORB_MATCHER_H

#include "common/Settings.h"
#include "common/NumTypes.h"
#include "cv/Align.h"

#include <set>

namespace ygz {

    struct Frame;
    struct MapPoint;

    class ORBMatcher {

    public:

        ORBMatcher(float nnratio = 0.6, bool checkOri = true) : mfNNratio(nnratio), mbCheckOrientation(checkOri) {}

        int SearchByDirectProjection(shared_ptr<Frame> F, const std::set<shared_ptr<MapPoint>> &vpMapPoints);

        enum StereoMethod {
            ORB_BASED, OPTIFLOW_BASED, OPTIFLOW_CV
        };

        void ComputeStereoMatchesOptiFlow(shared_ptr<Frame> f, bool only2Dpoints = false);

        void ComputeStereoMatchesOptiFlowCV(shared_ptr<Frame> f);

    private:

        void GetWarpAffineMatrix(
                shared_ptr<Frame> ref,
                shared_ptr<Frame> curr,
                const shared_ptr<Feature> px_ref,
                int level,
                const SE3d &TCR,
                Eigen::Matrix2d &ACR
        );

        // perform affine warp
        void WarpAffine(
                const Eigen::Matrix2d &ACR,
                const cv::Mat &img_ref,
                const Vector2d &px_ref,
                const int &level_ref,
                const shared_ptr<Frame> ref,
                const int &search_level,
                const int &half_patch_size,
                uint8_t *patch
        );

        inline uchar GetBilateralInterpUchar(
                const double &x, const double &y, const cv::Mat &gray) {
            const double xx = x - floor(x);
            const double yy = y - floor(y);
            uchar *data = &gray.data[int(y) * gray.step + int(x)];
            return uchar(
                    (1 - xx) * (1 - yy) * data[0] +
                    xx * (1 - yy) * data[1] +
                    (1 - xx) * yy * data[gray.step] +
                    xx * yy * data[gray.step + 1]
            );
        }

        float mfNNratio;
        bool mbCheckOrientation;
    };

}

#endif
