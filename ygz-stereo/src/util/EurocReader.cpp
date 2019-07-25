#include "util/EurocReader.h"
#include <fstream>
#include <iomanip>

using namespace std;

namespace ygz {

    bool LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                    vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps) {
        ifstream fTimes(strPathTimes.c_str());
        if (!fTimes) {
            LOG(ERROR) << "cannot find timestamp file: " << strPathTimes << endl;
            return false;
        }
        vTimeStamps.reserve(5000);
        vstrImageLeft.reserve(5000);
        vstrImageRight.reserve(5000);

        while (!fTimes.eof()) {
            string s;
            getline(fTimes, s);
            if (!s.empty()) {
                stringstream ss;
                ss << s;
                vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
                vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
                double t;
                ss >> t;
                vTimeStamps.push_back(t / 1e9);
            }
        }
        fTimes.close();

        if (strPathLeft.empty()) {
            LOG(ERROR) << "No images in left folder!" << endl;
            return false;
        }

        if (strPathRight.empty()) {
            LOG(ERROR) << "No images in right folder!" << endl;
            return false;
        }
        return true;
    }
}
