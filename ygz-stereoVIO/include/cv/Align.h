#ifndef YGZ_ALIGN_H_
#define YGZ_ALIGN_H_

#include "common/Settings.h"
#include "common/NumTypes.h"

#include <opencv2/opencv.hpp>

// This part is moved from rpg_SVO with modification to support ygz

namespace ygz {

    const int align_halfpatch_size = 4;
    const int align_patch_size = 8;
    const int align_patch_area = 64;

    bool Align2D(
            const cv::Mat &cur_img,
            uint8_t *ref_patch_with_border,
            uint8_t *ref_patch,
            const int n_iter,
            Vector2f &cur_px_estimate);

}

#endif
