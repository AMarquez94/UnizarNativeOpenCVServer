//
// Created by alejandro on 3/10/16.
//

#include "headers/utils.h"

using namespace std;
using namespace cv;

String intToString(const int& number){
    string Result;//string which will contain the result

    stringstream convert; // stringstream used for the conversion

    convert << number;//add the value of Number to the characters in the stream

    return convert.str();
}

String longToString(const long long int& number){
    string Result;

    stringstream convert;

    convert << number;

    return convert.str();
}

String floatToString(const float& number){
    string Result;

    stringstream convert;

    convert << number;

    return convert.str();
}

void log(const String& tag,const String& text){
    //__android_log_print(ANDROID_LOG_DEBUG, tag.c_str(), text.c_str());
	cout << tag << " " << text << endl;
}

void error(const String& tag,const String& text){
	cerr << tag << " " << text << endl;
    //__android_log_print(ANDROID_LOG_ERROR, tag.c_str(), text.c_str());
}

void printVectorFloat(const vector<float>& vf){
	String vftext = "";
	for(unsigned int i = 0; i < vf.size(); i++){
		vftext = vftext + " " + floatToString(vf[i]);
	}
	error("VECTOR", vftext);
}

void printMatFloat(const Mat& matf){
	String vftext = "";
	for(int i = 0; i < matf.cols; i++){
		vftext = vftext + " " + floatToString(matf.at<float>(0,i));
	}
	log("VECTOR", vftext);
}
