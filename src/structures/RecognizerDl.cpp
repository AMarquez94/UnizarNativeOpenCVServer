//
// Created by alejandro on 3/11/16.
//

#include "headers/RecognizerDl.h"
#include "headers/utils.h"
#include <fstream>
#include <iostream>
#include <string>
#include <limits>
#include <stdio.h>  // for snprintf
#include <chrono>

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;

using namespace std;
using namespace cv;
using namespace chrono;

void createAndResizeImages(const string& path, vector<Mat>& images, vector<int>& labels){
    ifstream file;
    file.open((path).c_str());
    if(!file.is_open()){
        cerr << "ERROR AL ABRIR FICHERO" << endl;
    } else{
        Size size(256,256);

        string word;
        int i = 0;
        while(file >> word){

            if(i%2 == 0){
                /* Imagen */
                Mat image = imread(word, CV_LOAD_IMAGE_COLOR);
                resize(image, image, size);
                images.push_back(image);
            } else{
                /* Label */
                labels.push_back(atoi(word.c_str()));
            }
            i++;
        }
    }
}

Mat getDescriptors(const string& imageprototxt, const vector<Mat>& images, const vector<int>& labels)
{
    ::google::InitGoogleLogging("nativeOpenCV");

    /* CPU */
    Caffe::SetDevice(0);
    Caffe::set_mode(Caffe::GPU);
    
    /* test*/
    caffe::Phase phase = caffe::TEST;
    
    /* bvlc_reference_caffenet.caffemodel */
    std::string pretrained_binary_proto("/opt/caffe-segnet/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"); // path to pretrained binary file

    /* imagenet_val.prototxt */
    std::string feature_extraction_proto(imageprototxt); // path to feature extraction proto file
    Net<float>* _net = new Net<float>(feature_extraction_proto, phase);
    boost::shared_ptr<Net<float> > feature_extraction_net(_net);
    feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);   /// load pretrained weights
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0])->AddMatVector(images,labels);
    
    /* fc7 */
    std::string blob_name;                        // get blob name
    blob_name = "fc7";
    
    // Check the given blob name
    CHECK(feature_extraction_net->has_blob(blob_name))         // check existence of blob
    << "Unknown feature blob name " << blob_name
    << " in the network " << feature_extraction_proto;
    
    /* num minibatches (1) */
    int num_mini_batches = 1;                 // parse number of mini batches

    /* opencv format */
    std::vector<Blob<float>*> input_vec;

    const boost::shared_ptr<Blob<float> > feature_blob = feature_extraction_net->blob_by_name(blob_name);  // obtain feature blob
    int batch_size = feature_blob->num();
    int dim_features = feature_blob->count() / batch_size;
    cv::Mat f_mat = Mat(dim_features, 0, CV_32F);
    int num_images_processed = batch_size * num_mini_batches;

    for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
    {
        feature_extraction_net->Forward(input_vec);                                 // feed forward
        const float* feature_blob_data;
        const boost::shared_ptr<Blob<float> > feature_blob = feature_extraction_net
        ->blob_by_name(blob_name);                // obtain feature blob
        int batch_size = feature_blob->num();
        
        const boost::shared_ptr<Blob<float> > label_blob = feature_extraction_net
        ->blob_by_name("label");                    // assumes there exists a 'label' blob
        const float* label_data = label_blob->cpu_data();
        
        for (int image_index = 0; image_index < batch_size; ++image_index)
        {
            feature_blob_data = feature_blob->cpu_data() +
            feature_blob->offset(image_index);
            
            vector<float> row;
            for (int feature_index = 0; feature_index < dim_features; ++feature_index)
            {
                if (feature_blob_data[feature_index] == 0)
                    row.push_back(0);
                else
                    row.push_back(feature_blob_data[feature_index]); 
            }
            f_mat.push_back(cv::Mat(row, CV_32F));
            
            if ((batch_index * batch_size + image_index) % 1000 == 0){
                LOG(INFO) << "Extracted features of " << batch_index * batch_size + image_index << " images";
        }
        }
    }

    cout << "SIZE " << f_mat.size() << endl; 
    return f_mat.reshape(1,num_images_processed);
}

void RecognizerDl::getDescriptor(const Mat& image, vector<float>& descriptor){
    // Size size(256,256);
    // Mat resizedImage;
    // resize(image, resizedImage, size);
    // vector<Mat> vm = {resizedImage};
    vector<Mat> vm = {image};
    vector<int> vl = {1};
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_sharedNet->layers()[0])->AddMatVector(vm,vl);

    /* fc7 */
    std::string blob_name;                        // get blob name
    blob_name = "fc7";
    
    // Check the given blob name
    CHECK(shared_feature_extraction_net->has_blob(blob_name))         // check existence of blob
    << "Unknown feature blob name " << blob_name
    << " in the network";
    
    /* num minibatches (1) */
    int num_mini_batches = 1;                 // parse number of mini batches

    /* opencv format */
    std::vector<Blob<float>*> input_vec;

    const boost::shared_ptr<Blob<float> > feature_blob = shared_feature_extraction_net->blob_by_name(blob_name);  // obtain feature blob
    int batch_size = feature_blob->num();
    int dim_features = feature_blob->count() / batch_size;
    cv::Mat f_mat = Mat(dim_features, 0, CV_32F);
    int num_images_processed = batch_size * num_mini_batches;

    for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
    {
        shared_feature_extraction_net->Forward(input_vec);                                 // feed forward
        const float* feature_blob_data;
        const boost::shared_ptr<Blob<float> > feature_blob = shared_feature_extraction_net
        ->blob_by_name(blob_name);                // obtain feature blob
        int batch_size = feature_blob->num();
        
        const boost::shared_ptr<Blob<float> > label_blob = shared_feature_extraction_net
        ->blob_by_name("label");                    // assumes there exists a 'label' blob
        const float* label_data = label_blob->cpu_data();
        
        for (int image_index = 0; image_index < batch_size; ++image_index)
        {
            feature_blob_data = feature_blob->cpu_data() +
            feature_blob->offset(image_index);
            
            vector<float> row;
            for (int feature_index = 0; feature_index < dim_features; ++feature_index)
            {
                if (feature_blob_data[feature_index] == 0)
                    row.push_back(0);
                else
                    row.push_back(feature_blob_data[feature_index]); 
            }
            f_mat.push_back(cv::Mat(row, CV_32F));
            
            if ((batch_index * batch_size + image_index) % 1000 == 0){
                LOG(INFO) << "Extracted features of " << batch_index * batch_size + image_index << " images";
        }
        }
    }

    cout << "SIZE " << f_mat.size() << endl;

    f_mat = f_mat.reshape(1,num_images_processed);
    descriptor.insert(descriptor.end(), f_mat.ptr<float>(0), f_mat.ptr<float>(0) + f_mat.cols);

}


RecognizerDl::RecognizerDl() {
    this->objects = vector<ObjectDl>();

    this->timer = Timer();
    this->timer.addTime("DEEP_LEARNING - Extract features");
    this->timer.addTime("DEEP_LEARNING - Compare features");
    this->timer.addTime("DEEP_LEARNING - Process result");
}

RecognizerDl::RecognizerDl(const vector <ObjectDl> &objects) {
    this->objects = objects;

    this->timer = Timer();
    this->timer.addTime("DEEP_LEARNING - Extract features");
    this->timer.addTime("DEEP_LEARNING - Compare features");
    this->timer.addTime("DEEP_LEARNING - Process result");
}

void RecognizerDl::createNet(const string& pathProtoTxt){
    
    /* test*/
    caffe::Phase phase = caffe::TEST;
    
    /* bvlc_reference_caffenet.caffemodel */
    std::string pretrained_binary_proto("/opt/caffe-segnet/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"); // path to pretrained binary file

    /* imagenet_val.prototxt */
    std::string feature_extraction_proto(pathProtoTxt); // path to feature extraction proto file
    _sharedNet = new Net<float>(feature_extraction_proto, phase);

    boost::shared_ptr<Net<float> > shared_feature_extraction_net(_sharedNet);
    this->shared_feature_extraction_net = shared_feature_extraction_net;
    shared_feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);   /// load pretrained weights
}

void RecognizerDl::createObjects(const String &path, const bool &add) {

     /* Leer fichero y cargar matriz de imagenes resizeadas */
    vector<Mat> images;
    vector<int> labels;

    /* Lista imagenes */
    createAndResizeImages("TFG/file_list.txt", images, labels);

    log("SIZE", intToString(images.size()));

    /* Read mat of floats */
    Mat objectsDescriptors = getDescriptors("TFG/imagenet_val.prototxt", images, labels);

    if(!objectsDescriptors.isContinuous()){
    	objectsDescriptors = objectsDescriptors.clone();
    }

    ifstream file;
    file.open((path + "/info_list.txt").c_str());
    if(!file.is_open()){
        error("PATH", path + "/info_list.txt");
        error("RECOGNIZER_DL", "error al abrir ficheros info");
    } else{

        string object_path;
        int row = 0;
        while(file >> object_path){

            ifstream object;
            object.open((path + "/" + object_path + "/info.txt").c_str());
            if(!object.is_open()){
                error("RECOGNIZER_DL", "error al abrir objeto");
            } else{
                String name;
                vector < vector<float> > descriptors;
                vector < String > viewNames;
                vector < String > allergens;
                bool easy;

                string word;
                int j = 0;

                int numVistas;
                while(object >> word){
                    if(j == 0){

                        /* Object name */
                        name = word;

                        log("NOMBRE", name);

                    } else if(j == 1){

                        /* Object easy or not */
                        easy = (word == "true");
                        log("EASY", intToString(easy));
                    } else if(j == 2){

                        numVistas = atoi(word.c_str());
                        log("NUM_VISTAS", intToString(numVistas));

                    } else if(j < (3 + 2 * numVistas) && j%2 == 1){
                    	vector<float> out;
                    	out.insert(out.end(), objectsDescriptors.ptr<float>(row), objectsDescriptors.ptr<float>(row) + objectsDescriptors.cols);
                        descriptors.push_back(out);
                        row++;
                        log("ROW", intToString(row));
                        log("VISTA", intToString((j/2)-1));
                    } else if(j < (3 + 2 * numVistas) && j%2 == 0){

                        /* Object name view */
                        viewNames.push_back(word);

                        String aux = word;
                        log("VISTA NOMBRE", aux);
                    } else{
                        allergens.push_back(word);
                        String aux = word;
                        log("ALERGENO", aux);
                    }
                    j++;
                }

                ObjectDl obj = ObjectDl(name, descriptors, viewNames, allergens, easy);
                this->objects.push_back(obj);
                object.close();
            }
        }

        file.close();
    }
}

void RecognizerDl::deleteObjects() {
    this->objects.clear();
}

int RecognizerDl::getObjectIndex(const String &name) {
    unsigned int i = 0;
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

Result RecognizerDl::recognizeObject(const Mat &sceneImgColour, Mat dstImg, mutex& mutex) {

    high_resolution_clock::time_point beginExtracting = high_resolution_clock::now();

    vector<float> sceneImgAttributes;

    mutex.lock();
    getDescriptor(sceneImgColour, sceneImgAttributes);
    mutex.unlock();

    /* Ends measuring time */
    high_resolution_clock::time_point endExtracting = high_resolution_clock::now();

    /* Added to total time */
    auto durationExtracting = duration_cast<microseconds>( endExtracting - beginExtracting ).count();
    this->timer.addDuration("DEEP_LEARNING - Extract features", durationExtracting);
	


    high_resolution_clock::time_point beginMatching = high_resolution_clock::now();

    int bestMatchObject = -1;
    int bestMatchview = -1;
    float bestDistance = numeric_limits<float>::max();

    if(sceneImgAttributes.size() == 4096) {

        for (unsigned int i = 0; i < this->objects.size(); i++) {
            /* Iterating over object array */


            for (int j = 0; j < this->objects[i].getNumberOfViews(); j++) {
                /* Iterating over object views */

                float euclideanDistance = 0;
                for(unsigned int k = 0; k < sceneImgAttributes.size(); k++){
                    float substraction = sceneImgAttributes[k] - this->objects[i].getViewsDescriptors()[j][k];
                    float squared = substraction * substraction;
                    euclideanDistance = euclideanDistance + squared;
                }
                euclideanDistance = sqrt(euclideanDistance);
                if(euclideanDistance < bestDistance){
                    bestDistance = euclideanDistance;
                    bestMatchObject = i;
                    bestMatchview = j;
                }
            }
        }
    }

    bool isMatch = bestDistance < numeric_limits<float>::max(); //TODO some kind of filter

    error("BEST DISTANCE ", floatToString(bestDistance));

    /* Ends measuring time */
    high_resolution_clock::time_point endMatching = high_resolution_clock::now();

    /* Added to total time */
    auto durationMatching = duration_cast<microseconds>( endMatching - beginMatching ).count();
    this->timer.addDuration("DEEP_LEARNING - Compare features", durationMatching);


    high_resolution_clock::time_point beginProcessing = high_resolution_clock::now();
   
    String objectName;
    String viewName;
    vector<String> allergens;
    vector<Point2f> scene_corners;

    sceneImgColour.copyTo(dstImg);

    if(isMatch){
        objectName = this->objects[bestMatchObject].getName();
        viewName = this->objects[bestMatchObject].getViewsNames()[bestMatchview];
        allergens = this->objects[bestMatchObject].getAllergens();
    } else{
        objectName = "No object";
        viewName = "";
        allergens = vector<String>(0);
    }

    /* Write object name */
    Size textsize = getTextSize(objectName, FONT_HERSHEY_COMPLEX, 1, 2, 0);

    Point org((640 - textsize.width - 20), textsize.height + 20);
    putText( dstImg, objectName, org, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 0), 2);

    for(unsigned int i = 0; i < allergens.size(); i++){
        textsize = getTextSize(allergens[i], FONT_HERSHEY_COMPLEX, 0.75, 2, 0);
        Point org((640 - textsize.width - 20), textsize.height + 20 + 30*(i+1));

        /* Comprobar con lista de alergenos para pintar de verde o de rojo */
        putText( dstImg, allergens[i], org, FONT_HERSHEY_COMPLEX, 0.75, Scalar(0, 0, 255), 2);
    }


    /* Ends measuring time */
    high_resolution_clock::time_point endProcessing = high_resolution_clock::now();

    /* Added to total time */
    auto durationProcessing = duration_cast<microseconds>( endProcessing - beginProcessing ).count();
    this->timer.addDuration("DEEP_LEARNING - Process result", durationProcessing);

    return Result(objectName, viewName, 0, 0, scene_corners, allergens);
}

void RecognizerDl::printTimer() {
    this->timer.printTimes();
}
