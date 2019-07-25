#ifndef  YGZ_EUROC_READER_H
#define  YGZ_EUROC_READER_H

#include "common/Settings.h"
#include "common/NumTypes.h"
//#include "common/IMUData.h"

// 一些用于EuRoC数据集的IO函数

namespace ygz {


    // Load the stereo image data
    // 输入：左眼图像目录，右眼图像目录，时间戳文件
    // 输出：排序后左眼图像文件路径、右眼图像文件路径、时间戳
    bool LoadImages(const std::string &strPathLeft, const std::string &strPathRight, const std::string &strPathTimes,
                    std::vector<std::string> &vstrImageLeft, std::vector<std::string> &vstrImageRight, std::vector<double> &vTimeStamps);

    // Load the IMU data
//    bool LoadImus(const string &strImuPath, VecIMU &vImus);

    /**
     * Load the ground truth trajectory
     * @param [in] trajPath the path to trajectory, in euroc will be xxx/state_groundtruth_estimate0/data.csv
     * @param [out] the loaded trajectory
     * @return true if succeed
     */
//    typedef map<double, SE3d, std::less<double>, Eigen::aligned_allocator<SE3d>> TrajectoryType;

//    bool LoadGroundTruthTraj(const std::string &trajPath,
//                             TrajectoryType &trajectory);
}

#endif
