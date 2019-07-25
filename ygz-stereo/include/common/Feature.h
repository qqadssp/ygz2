#ifndef YGZ_FEATURE_H
#define YGZ_FEATURE_H

#include "common/NumTypes.h"
#include "common/Settings.h"

using namespace std;

namespace ygz {

    // forward declare
    struct MapPoint;

    struct Feature {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Vector2f mPixel = Vector2f(0, 0);        // the pixel position
        shared_ptr<MapPoint> mpPoint = nullptr;  // the corresponding map point, nullptr if not associated
        float mfInvDepth = -1;                   // inverse depth, invalid if less than zero.

        // data used in ORB
        size_t mLevel = 0;                       // the pyramid level

        // flags
        bool mbOutlier = false;                  // true if it is an outlier
    };
}


#endif
