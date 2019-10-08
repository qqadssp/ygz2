#ifndef YGZ_ORB_MATCHER_H
#define YGZ_ORB_MATCHER_H

#include "common/Settings.h"
#include "common/NumTypes.h"
#include "cv/Align.h"

#include <set>

namespace ygz {

    struct Frame;
    struct MapPoint;

    struct Match {
        Match(int _index1 = -1, int _index2 = -1, int _dist = -1) : index1(_index1), index2(_index2), dist(_dist) {}

        int index1 = -1;
        int index2 = -1;
        int dist = -1;
    };

    class ORBMatcher {

    public:

        ORBMatcher(float nnratio = 0.6, bool checkOri = true) : mfNNratio(nnratio), mbCheckOrientation(checkOri) {}

        int SearchByDirectProjection(shared_ptr<Frame> F, const std::set<shared_ptr<MapPoint>> &vpMapPoints);

        enum StereoMethod {
            ORB_BASED, OPTIFLOW_BASED, OPTIFLOW_CV
        };

        void ComputeStereoMatches(shared_ptr<Frame> f, StereoMethod method = ORB_BASED);

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

        inline int GetBestSearchLevel(
                const Eigen::Matrix2d &ACR,
                const int &max_level) {
            int search_level = 0;
            float D = ACR.determinant();
            while (D > 3.0 && search_level < max_level) {
                search_level += 1;
                D *= setting::invLevelSigma2[1];
            }
            return search_level;
        }

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

        inline float RadiusByViewingCos(const float &viewCos) {
            if (viewCos > 0.998)
                return 2.5;
            else
                return 4.0;
        }


        float mfNNratio;
        bool mbCheckOrientation;
    };

}

#endif
