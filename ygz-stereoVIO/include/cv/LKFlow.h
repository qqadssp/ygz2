#ifndef  YGZ_LK_FLOW_H
#define  YGZ_LK_FLOW_H

#include "common/Settings.h"
#include "common/Frame.h"

namespace ygz {

    bool LKFlowSinglePoint(
            const vector<Mat> &pyramid1,
            const vector<Mat> &pyramid2,
            const Vector2f &pixel1,
            Vector2f &pixel2
    );

    int LKFlowCV(
            const shared_ptr<Frame> ref,
            const shared_ptr<Frame> current,
            VecVector2f &refPts,
            VecVector2f &trackedPts
    );

    inline uchar GetBilateralInterpUchar(
            const float &x, const float &y, const Mat &gray) {
        const float xx = x - floor(x);
        const float yy = y - floor(y);
        uchar *data = gray.data + int(y) * gray.step + int(x);
        return uchar(
                (1 - xx) * (1 - yy) * data[0] +
                xx * (1 - yy) * data[1] +
                (1 - xx) * yy * data[gray.step] +
                xx * yy * data[gray.step + 1]
        );
    }

}

#endif
