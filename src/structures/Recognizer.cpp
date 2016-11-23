//
// Created by alejandro on 27/09/16.
//

#include "headers/Recognizer.h"
#include "headers/utils.h"
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>

using namespace chrono;
using namespace std;
using namespace cv;


/* Aux methods */
bool isConvex(const vector <Point2f>& scene_corners);


Recognizer::Recognizer() {
    this->objects = vector<ObjectKp>();
}

Recognizer::Recognizer(const String& detector,const String& extractor,const String& matcher){

    if(detector == "ORB"){
        this->detector = FeatureDetector::create("ORB");
    } else if(detector == "FAST"){
        this->detector = FeatureDetector::create("FAST");
    } else if(detector == "BRISK"){
        this->detector = FeatureDetector::create("BRISK");
    }

    if(extractor == "ORB"){
        this->extractor = DescriptorExtractor::create("ORB"/*500, 1.2f, 8, 31, 0, 4*/);
    } else if(extractor == "BRISK"){
        this->extractor = DescriptorExtractor::create("BRISK");
    }

    this->matcher = DescriptorMatcher::create(matcher);
    this->matcherDistanceFilter = 0.7;

    this->timer = Timer();
    this->timer.addTime("KEYPOINTS - Extract features");
    this->timer.addTime("KEYPOINTS - Compare features");
    this->timer.addTime("KEYPOINTS - Process result");
}

Recognizer::Recognizer(const String& detector,const String& extractor,const String& matcher,
                       const vector<ObjectKp>& objects){
    if(detector == "ORB"){
        this->detector = FeatureDetector::create("ORB");
    } else if(detector == "FAST"){
        this->detector = FeatureDetector::create("FAST");
    } else if(detector == "BRISK"){
        this->detector = FeatureDetector::create("BRISK");
    }

     if(extractor == "ORB"){
        this->extractor = DescriptorExtractor::create("ORB"/*500, 1.2f, 8, 31, 0, 4*/);
    } else if(extractor == "BRISK"){
        this->extractor = DescriptorExtractor::create("BRISK");
    }

    this->matcher = DescriptorMatcher::create(matcher);
    this->matcherDistanceFilter = 0.7;

    this->objects = objects;

    this->timer = Timer();
    this->timer.addTime("KEYPOINTS - Extract features");
    this->timer.addTime("KEYPOINTS - Compare features");
    this->timer.addTime("KEYPOINTS - Process result");
}

Recognizer::Recognizer(const String& detector,const vector<float>& detectorParams,const String& extractor,
                       const vector<float>& extractorParams,const String& matcher,const double& matcherDistance){

    /* ORB: int float int int int int int int int */
    /* BRISK: int int float */

    /* TODO: Implement */
}

Recognizer::Recognizer(const String& detector,const vector<float>& detectorParams,const String& extractor,
                       const vector<float>& extractorParams,const String& matcher,const double& matcherDistance,
                       const vector<ObjectKp>& objects){

    /* TODO: Implement */
}

/* Getters */

Ptr<FeatureDetector> Recognizer::getDescriptor() const{
    return this->getDescriptor();
}


Ptr<DescriptorExtractor> Recognizer::getExtractor() const{
    return this->getExtractor();
}

Ptr<DescriptorMatcher> Recognizer::getMatcher() const{
    return this->getMatcher();
}

vector<ObjectKp> Recognizer::getObjects() const{
    return this->objects;
}

/* Setters */

void Recognizer::setDetector(const String& detector) {
    if(detector == "ORB"){
        this->detector = FeatureDetector::create("ORB");
    } else if(detector == "FAST"){
        this->detector = FeatureDetector::create("FAST");
    } else if(detector == "BRISK"){
        this->detector = FeatureDetector::create("BRISK");
    }
}

void Recognizer::setExtractor(const String& extractor) {
     if(extractor == "ORB"){
        this->extractor = DescriptorExtractor::create("ORB"/*500, 1.2f, 8, 31, 0, 4*/);
    } else if(extractor == "BRISK"){
        this->extractor = DescriptorExtractor::create("BRISK");
    }
}

void Recognizer::setMatcher(const String& matcher) {
    this->matcher = DescriptorMatcher::create(matcher);
    this->matcherDistanceFilter = 0.7;
}

void Recognizer::setObjects(const vector<ObjectKp>& objects) {
    this->objects = objects;
}

ObjectKp Recognizer::createObject(const String& path,const bool& add) {
    ifstream file;
    file.open((path + "/info.txt").c_str());
    if (!file.is_open()){
        return ObjectKp(true);
    } else{
        String name;
        vector < vector<KeyPoint> > keypoints;
        vector <Mat> descriptors;
        vector < vector<Point2f> > corners;
        vector <String> viewNames;
        vector <String> allergens;
        bool easy;

        string word;
        int i = 0;

        int numVistas;
        int numAllergens;
        while (file >> word)
        {
            if(i == 0){

                /* Object name */
                name = word;

                log("NOMBRE", name);

            } else if(i == 1){

                /* Object easy or not */
                easy = (word == "true");
                log("EASY", intToString(easy));
            } else if(i == 2){

                numVistas = atoi(word.c_str());
                log("NUM_VISTAS", intToString(numVistas));

            } else if(i < (3 + 2 * numVistas) && i%2 == 1){
                /* Object image view*/
                keypoints.push_back(vector <KeyPoint>());
                descriptors.push_back(Mat());
                corners.push_back(vector <Point2f>(4));


                Mat image = imread(path + "/" + word, CV_LOAD_IMAGE_GRAYSCALE);
                this->detector->detect(image, keypoints[(i/2)-1]);
                this->extractor->compute(image, keypoints[(i/2)-1], descriptors[(i/2)-1]);
                corners[(i/2)-1][0] = cvPoint(0,0);
                corners[(i/2)-1][1] = cvPoint(image.cols,0);
                corners[(i/2)-1][2] = cvPoint(image.cols, image.rows);
                corners[(i/2)-1][3] = cvPoint(0, image.rows);
                log("VISTA", intToString((i/2)-1));
            } else if(i < (3 + 2 * numVistas) && i%2 == 0){

                /* Object name view */
                viewNames.push_back(word);

                String aux = word;
                log("VISTA NOMBRE", aux);
            } else{
                allergens.push_back(word);
                String aux = word;
                log("ALERGENO", aux);
            }

            i++;
        }
        ObjectKp obj = ObjectKp(name, keypoints, descriptors, corners, viewNames, allergens, easy);

        if(add){
            this->objects.push_back(obj);
        }

        file.close();

        return obj;
    }
}

void Recognizer::deleteObjects() {
    this->objects.clear();
}

int Recognizer::getObjectIndex(const String& name) {
    int i = 0;
    bool found = false;
    while(i < this->objects.size() && !found){
        if(this->objects[i].getName() == name){
            found = true;
        } else{
            i++;
        }
    }

    if(!found){
        return -1;
    } else{
        return i;
    }
}

Result Recognizer::recognizeObject(const Mat& sceneImgGray,const Mat& sceneImgColour, Mat dstImg){

    high_resolution_clock::time_point beginExtracting = high_resolution_clock::now();

    /* Extract features from the sceneImage */
    vector<KeyPoint> keypointsScene;
    Mat descriptorsScene;
    this->detector->detect(sceneImgGray, keypointsScene);
    this->extractor->compute(sceneImgGray, keypointsScene, descriptorsScene);

    /* Ends measuring time */
    high_resolution_clock::time_point endExtracting = high_resolution_clock::now();

    /* Added to total time */
    auto durationExtracting = duration_cast<microseconds>( endExtracting - beginExtracting ).count();
    this->timer.addDuration("KEYPOINTS - Extract features", durationExtracting);

    /* Extract the matches between the cameraImage and all the objects in the object list */

    high_resolution_clock::time_point beginMatching = high_resolution_clock::now();

    /* Data structures */
    vector < vector< vector<KeyPoint> > >matchedObjects, matchedScene;
    vector < vector< vector<Point2f> > >objs, scene;
    vector < vector< vector< vector<DMatch> > > >matches(this->objects.size());

    int bestMatchObject = -1;   //Best object recognized
    int bestMatchView = -1;     //Best view of the best object recognized
    int numberOfMatches = 0;    //Maximum number of matches found

    if(keypointsScene.size() != 0){

        for(int i = 0; i < this->objects.size(); i++){
            /* Iterating over objects */

            matchedObjects.push_back(vector < vector<KeyPoint> >(this->objects[i].getNumberOfViews()));
            matchedScene.push_back(vector < vector<KeyPoint> >(this->objects[i].getNumberOfViews()));
            objs.push_back(vector < vector<Point2f> >(this->objects[i].getNumberOfViews()));
            scene.push_back(vector < vector<Point2f> >(this->objects[i].getNumberOfViews()));
            matches[i] = vector< vector < vector<DMatch> > >(this->objects[i].getNumberOfViews());

            for(int j = 0; j < this->objects[i].getNumberOfViews(); j++){
                /* Iterating over object views */

                if(this->objects[i].getViewsKeyPoints()[j].size() != 0){

                    matcher->knnMatch(this->objects[i].getViewsDescriptors()[j],descriptorsScene,matches[i][j],2);

                    for(int k = 0; k < matches[i][j].size(); k++){

                        /* Iterating over matches */

                        if(matches[i][j][k][0].distance < this->matcherDistanceFilter * matches[i][j][k][1].distance){

                            /* Good match -> save matches points and info */
                            matchedObjects[i][j].push_back(this->objects[i].getViewsKeyPoints()[j][matches[i][j][k][0].queryIdx]);
                            matchedScene[i][j].push_back(keypointsScene[matches[i][j][k][0].trainIdx]);
                            objs[i][j].push_back(this->objects[i].getViewsKeyPoints()[j][matches[i][j][k][0].queryIdx].pt);
                            scene[i][j].push_back(keypointsScene[matches[i][j][k][0].trainIdx].pt);
                        }
                    }

                    if(numberOfMatches < matchedScene[i][j].size()){

                        /* At the moment, object with more matches -> object recognized */
                        bestMatchObject = i;
                        bestMatchView = j;
                        numberOfMatches = matchedScene[i][j].size();
                    }
                }
            }
        }
    }

    /* Ends measuring time */
    high_resolution_clock::time_point endMatching = high_resolution_clock::now();

    /* Added to total time */
    auto durationMatching = duration_cast<microseconds>( endMatching - beginMatching ).count();
    this->timer.addDuration("KEYPOINTS - Compare features", durationMatching);

    high_resolution_clock::time_point beginProcessing = high_resolution_clock::now();

    bool isMatch = false;

    vector<Point2f> scene_corners;

    if(bestMatchObject!=-1 && bestMatchView!=-1 && matchedScene[bestMatchObject][bestMatchView].size() >= 4){

        //Homography matrix
        Mat H = findHomography(objs[bestMatchObject][bestMatchView], scene[bestMatchObject][bestMatchView], RANSAC);

        if(!H.empty()){
            //Corners Object
            scene_corners = vector<Point2f>(4);

            perspectiveTransform(this->objects[bestMatchObject].getViewsCorners()[bestMatchView], scene_corners, H);

            /* Check if convex or concav */
            Scalar colour;
            if(isConvex(scene_corners)){
                colour = Scalar(0,255,0);
                isMatch = true;
            } else{
                colour = Scalar(0,0,255);
            }
        }
    }

    String objectName;
    String viewName;
    vector <String> allergens;

    if(isMatch){
        //-- Draw keypoints
        drawKeypoints( sceneImgColour, matchedScene[bestMatchObject][bestMatchView], dstImg, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

//        objectName = this->objects[bestMatchObject].getName() + "_" +
//            this->objects[bestMatchObject].getViewsNames()[bestMatchView];

        objectName = this->objects[bestMatchObject].getName();
        viewName = this->objects[bestMatchObject].getViewsNames()[bestMatchView];
        allergens = this->objects[bestMatchObject].getAllergens();
    }
    else{

        /* No match */
        sceneImgColour.copyTo(dstImg);
        objectName = "No object";
        viewName = "";
        allergens = vector<String>(0);
    }

    /* Write object name */
    Size textsize = getTextSize(objectName, FONT_HERSHEY_COMPLEX, 1, 2, 0);

    Point org((640 - textsize.width - 20), textsize.height + 20);
    putText( dstImg, objectName, org, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 0), 2);

    for(int i = 0; i < allergens.size(); i++){
        textsize = getTextSize(allergens[i], FONT_HERSHEY_COMPLEX, 0.75, 2, 0);
        Point org((640 - textsize.width - 20), textsize.height + 20 + 30*(i+1));

        /* Comprobar con lista de alergenos para pintar de verde o de rojo */
        putText( dstImg, allergens[i], org, FONT_HERSHEY_COMPLEX, 0.75, Scalar(0, 0, 255), 2);
    }

    /* Ends measuring time */
    high_resolution_clock::time_point endProcessing = high_resolution_clock::now();

    /* Added to total time */
    auto durationProcessing = duration_cast<microseconds>( endProcessing - beginProcessing ).count();
    this->timer.addDuration("KEYPOINTS - Process result", durationProcessing);

    return Result(objectName, viewName, keypointsScene.size(), numberOfMatches, scene_corners, allergens);
}

/**
 *
 * Return true if the cuadrilateral formed by the scene_corners is convex
 */
bool isConvex(const vector <Point2f>& scene_corners){

    bool sign = false;
    bool convex = true;

    int i = 0;
    while(i < 4 && convex){
        float dx1 = scene_corners[(i+2)%4].x - scene_corners[(i+1)%4].x;
        float dy1 = scene_corners[(i+2)%4].y - scene_corners[(i+1)%4].y;
        float dx2 = scene_corners[i].x - scene_corners[(i+1)%4].x;
        float dy2 = scene_corners[i].y - scene_corners[(i+1)%4].y;
        float z = dx1*dy2 - dy1*dx2;
        if(i == 0){
            sign = z > 0;
        } else{
            if( sign != (z>0)){
                convex = false;
            }
        }
        i++;
    }

    return convex;
}

void Recognizer::printTimer() {
    this->timer.printTimes();
}


