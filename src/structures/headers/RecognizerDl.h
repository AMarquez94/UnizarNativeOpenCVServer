//
// Created by alejandro on 3/11/16.
//

#ifndef UNIZAR_MOBILECV_RECOGNIZERDL_H
#define UNIZAR_MOBILECV_RECOGNIZERDL_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "ObjectDl.h"
#include "Result.h"
#include "Timer.h"
#include <mutex>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/regex.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;

using namespace std;
using namespace cv;

class RecognizerDl {

private:
    vector<ObjectDl> objects;
    Net<float> *_sharedNet;
    boost::shared_ptr<Net<float> > shared_feature_extraction_net;
    Timer timer;


public:
    RecognizerDl();
    RecognizerDl(const vector<ObjectDl>& objects);

    vector<ObjectDl> getObjects() const;

    void createObjects(const String& path, const bool& add);
    void deleteObjects();
    int getObjectIndex(const String& name);
    Result recognizeObject(const Mat& sceneImgColour, Mat dstImg, mutex& mutex);
    void createNet(const string& pathProtoTxt);
    void getDescriptor(const Mat& image, vector<float>& descriptor);
    void printTimer();
};


#endif //UNIZAR_MOBILECV_RECOGNIZERDL_H
