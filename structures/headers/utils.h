//
// Created by alejandro on 3/10/16.
//

#ifndef ANDROIDNATIVEOPENCV_UTILS_H
#define ANDROIDNATIVEOPENCV_UTILS_H

#include <opencv2/core/core.hpp>
#include <string>
#include<iostream>
//#include <android/log.h>

using namespace std;
using namespace cv;

String intToString(const int& number);

String longToString(const long long int& number);

String floatToString(const float& number);

void log(const String& tag, const String& text);

void error(const String& tag,const String& text);

void printVectorFloat(const vector<float>& vf);

void printMatFloat(const Mat& matf);



#endif //ANDROIDNATIVEOPENCV_UTILS_H
