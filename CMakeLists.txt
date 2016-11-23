cmake_minimum_required(VERSION 2.8)
PROJECT(NativeOpenCV)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")   #Compilador necesario para cosas

set (CMAKE_MODULE_PATH /opt/caffe-segnet/cmake/Modules)
set (CMAKE_PREFIX_PATH /opt/caffe-segnet)
#set (Caffe_INCLUDE_DIRS /opt/caffe-segnet/include)
set(Caffe_INCLUDE_DIRS "/usr/include;/opt/caffe-segnet/include;/usr/include/atlas;/usr/local/include")
set (CUDA_INCLUDE /usr/local/cuda-7.5/include)
set (OpenCV_LIBS /usr/local/lib/)
#set (Caffe_LIBRARIES /opt/caffe-segnet/lib/)
set(Caffe_LIBRARIES "/opt/caffe-segnet/lib/libcaffe.so")
set (CUDA_LIBS /usr/local/cuda-7.5/lib64/)

include_directories( "${PROJECT_SOURCE_DIR}" )
include_directories( "${Caffe_INCLUDE_DIRS}" )
include_directories( "${CUDA_INCLUDE} ")

#add_subdirectory( src )

include(/opt/caffe-segnet/cmake/Utils.cmake)
include(/opt/caffe-segnet/cmake/Targets.cmake)
include(/opt/caffe-segnet/cmake/Misc.cmake)
include(/opt/caffe-segnet/cmake/Summary.cmake)
include(/opt/caffe-segnet/cmake/ConfigGen.cmake)

include(FindProtobuf)

find_package(Protobuf REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Caffe REQUIRED)

##Extract feats
#add_executable(nativeOpenCV src/extract_features_testing.cpp)


##Mio
include_directories(${PROTOBUF_INCLUDE_DIR})
include_directories ("${CMAKE_SOURCE_DIR}/src/structures/headers")

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cc)
add_executable(NativeOpenCV ${SRC_FILES} ${CMAKE_SOURCE_DIR}/src/Main.cpp)

target_link_libraries( NativeOpenCV ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${CUDA_LIBS} ${PROTOBUF_LIBRARY}) 
