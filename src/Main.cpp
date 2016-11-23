#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "structures/headers/Object.h"
#include "structures/headers/Recognizer.h"
#include "structures/headers/TestImage.h"
#include "structures/headers/utils.h"
#include "structures/headers/ProtoResult.pb.h"
#include "structures/headers/Result.h"
#include "structures/headers/Timer.h"
#include "structures/headers/RecognizerDl.h"
#include "structures/headers/Tester.h"

#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>
#include <string.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <mutex>

#include <chrono>

using std::string;
using std::cout;
using namespace std::chrono;
using namespace cv;

void *processImage(void *);
void *processInt(void *);
Recognizer recognizer;
RecognizerDl recognizerDl;
Timer timer;
bool deep_learning;
std::mutex myMutex;

int main( int argc, char** argv )
{
	deep_learning = (argc == 3) && (strcmp(argv[2],"dl") == 0);

	timer = Timer();
	timer.addTime("DEEP LEARNING - Loading images");
	timer.addTime("DEEP LEARNING - Creating net");
	timer.addTime("KEYPOINTS - Loading images");

    /* Creates recognizer */
	//Recognizer recognizer = Recognizer("ORB","ORB","BruteForce-Hamming");
	if(!deep_learning){
		recognizer = Recognizer("ORB","ORB","BruteForce-Hamming");
	} else{
		recognizerDl = RecognizerDl();

		high_resolution_clock::time_point beginObjectsDL = high_resolution_clock::now();
		recognizerDl.createObjects("./TFG", true);
		/* Ends measuring time */
	    high_resolution_clock::time_point endObjectsDL = high_resolution_clock::now();

	    /* Added to total time */
	    auto durationObjectsDL = duration_cast<microseconds>( endObjectsDL - beginObjectsDL ).count();
	    timer.addDuration("DEEP LEARNING - Loading images", durationObjectsDL);


		high_resolution_clock::time_point beginNet = high_resolution_clock::now();
		recognizerDl.createNet("./TFG/imagenet_val_rec.prototxt");
		/* Ends measuring time */
	    high_resolution_clock::time_point endNet = high_resolution_clock::now();

	    /* Added to total time */
	    auto durationNet = duration_cast<microseconds>( endNet - beginNet ).count();
	    timer.addDuration("DEEP LEARNING - Creating net", durationNet);
	}

	if(!deep_learning){
		/* Adds objects */
		high_resolution_clock::time_point begin = high_resolution_clock::now();
		recognizer.createObject("./TFG/BD/Asturiana", true);
		recognizer.createObject("./TFG/BD/Ariel", true);
		recognizer.createObject("./TFG/BD/Cafe", true);
		recognizer.createObject("./TFG/BD/Celta", true);
		recognizer.createObject("./TFG/BD/Nocilla", true);
		recognizer.createObject("./TFG/BD/Orlando", true);
		recognizer.createObject("./TFG/BD/Agua", true);
		recognizer.createObject("./TFG/BD/Ambar", true);
		recognizer.createObject("./TFG/BD/Licor43", true);
		recognizer.createObject("./TFG/BD/Manocao", true);
		recognizer.createObject("./TFG/BD/Nescafe", true);
		recognizer.createObject("./TFG/BD/Sal_Hacendado", true);
		recognizer.createObject("./TFG/BD/Aceite", true);
		recognizer.createObject("./TFG/BD/Axe", true);
		recognizer.createObject("./TFG/BD/Cesar_Heinz", true);
		recognizer.createObject("./TFG/BD/Enjuague", true);
		recognizer.createObject("./TFG/BD/Frenadol", true);
		recognizer.createObject("./TFG/BD/Gel", true);
		recognizer.createObject("./TFG/BD/KH7", true);
		recognizer.createObject("./TFG/BD/Levadura", true);
		recognizer.createObject("./TFG/BD/Mayonesa", true);
		recognizer.createObject("./TFG/BD/Tonica", true);
		recognizer.createObject("./TFG/BD/Tortitas_arroz", true);
		recognizer.createObject("./TFG/BD/Aceite_corporal", true);
		recognizer.createObject("./TFG/BD/Aftersun", true);
		recognizer.createObject("./TFG/BD/Anis", true);
		recognizer.createObject("./TFG/BD/Betadine", true);
		recognizer.createObject("./TFG/BD/Cafe_hacendado", true);
		recognizer.createObject("./TFG/BD/Caldo", true);
		recognizer.createObject("./TFG/BD/Champu", true);
		recognizer.createObject("./TFG/BD/Chromecast", true);
		recognizer.createObject("./TFG/BD/Cicaderma", true);
		recognizer.createObject("./TFG/BD/Compeed", true);
		recognizer.createObject("./TFG/BD/Crema", true);
		recognizer.createObject("./TFG/BD/Emulsion", true);
		recognizer.createObject("./TFG/BD/Espuma", true);
		recognizer.createObject("./TFG/BD/Fideos", true);
		recognizer.createObject("./TFG/BD/Film", true);
		recognizer.createObject("./TFG/BD/Galletas_danesas", true);
		recognizer.createObject("./TFG/BD/Peroxiben", true);
		recognizer.createObject("./TFG/BD/Talco", true);
		recognizer.createObject("./TFG/BD/Ultima", true);
		recognizer.createObject("./TFG/BD/Cereales", true);
		recognizer.createObject("./TFG/BD/Aceite_botella", true);
		recognizer.createObject("./TFG/BD/Aceitunas", true);
		recognizer.createObject("./TFG/BD/Atun", true);
		recognizer.createObject("./TFG/BD/Esparragos", true);
		recognizer.createObject("./TFG/BD/Leche_hacendado", true);
		recognizer.createObject("./TFG/BD/Miel", true);
		recognizer.createObject("./TFG/BD/Tomate_carrefour", true);
		/* Ends measuring time */
	    high_resolution_clock::time_point end = high_resolution_clock::now();

	    /* Added to total time */
	    auto duration = duration_cast<microseconds>( end - begin ).count();
	    timer.addDuration("KEYPOINTS - Loading images", duration);
	}

	bool endServer = false;

	int localSocket, remoteSocket, port;
	struct sockaddr_in localAddress, remoteAddress;
	pthread_t thread_id;

	int addressLength = sizeof(struct sockaddr_in);

	if( argc <= 1){
		port = 8888;
	} else{
		port = atoi(argv[1]);
	}

	localSocket = socket(AF_INET, SOCK_STREAM, 0);
	if (localSocket == -1){
		error("SERVER", "error al crear el socket");
	}

	int yes = 1;
	int resultNoDelay = setsockopt(localSocket, IPPROTO_TCP, TCP_NODELAY, (char *) &yes, sizeof(int));    // 1 - on, 0 - off

	if(resultNoDelay < 0){
        error("SERVER", "Error con TCP_NODELAY o TCP_QUICKACK");
	}
	localAddress.sin_family = AF_INET;	//IPv4
	localAddress.sin_addr.s_addr = INADDR_ANY;	//Binding to all interfaces
	localAddress.sin_port = htons( port );

	if(::bind(localSocket, (struct sockaddr *)&localAddress , sizeof(localAddress)) < 0) {
		log("BIND ERROR", intToString(errno));
		error("SERVER", "error al hacer bind al socket");
		exit(1);
	}

	listen(localSocket, 5);


	if(deep_learning){
		log("SERVER", "Esperando conexiones en puerto " + intToString(port) + " (DeepLearning)");
	} else{
		log("SERVER", "Esperando conexiones en puerto " + intToString(port) + " (LocalFeatures)");
	}
	

	while(!endServer){
		remoteSocket = accept(localSocket, (struct sockaddr *)&remoteAddress, (socklen_t*)&addressLength);

		if(remoteSocket < 0){
			error("SERVER", "Fallo en accept");
			exit(1);
		}

		log("SERVER", "Conexion aceptada");

		pthread_create(&thread_id, NULL, processImage, &remoteSocket);
	}

	return 0;
}

void *processImage(void *ptr){

	int socket = *(int *)ptr;
	bool end = false;

	/* Receive image */
	Mat imgColor;
	imgColor = Mat::zeros(480,640,CV_8UC3);
	Mat imgGray(imgColor.size(), CV_8UC1);
	Mat imgDest(imgColor.size(), CV_8UC3);
//	int imgSize = imgColor.total() * imgColor.elemSize();
	int bytes = 0;

	int receivedInt;

	if( ! imgColor.isContinuous()){
		imgColor = imgColor.clone();
	}

	// Mat allImages;
	// FileStorage fs("./TFG/feats_test_1000.xml", FileStorage::READ);
	// fs["FeatureMat"] >> allImages;
	int i = 0;
	//error("SizeMat", intToString(allImages.rows));

//	long long int totalDuration = 0.0;
//	int iterations = 0;

	Timer timerThread = Timer();
	timerThread.addTime("TOTAL");
	timerThread.addTime("RECEIVING");
	timerThread.addTime("DECODING");
	timerThread.addTime("PROCESSING - Total");
	timerThread.addTime("SERIALIZING");
	timerThread.addTime("SENDING");

	/* Enviar si estamos en Deep Learning o no */
	int isDeepLearning;
	if(deep_learning){
		isDeepLearning = htonl(1);
	} else{
		isDeepLearning = htonl(0);
	}

	bytes = send(socket, (const char*)&isDeepLearning, sizeof(isDeepLearning), 0);
	if(bytes < 0){
		error("SERVER", "Error al enviar modo");
		end = true;
	}

	while(!end){

		

		/* We receive first the image encoded size */
		log("SERVER", "Recibiendo size...");
		high_resolution_clock::time_point beginReceiving = high_resolution_clock::now();

		bytes = recv(socket, &receivedInt, sizeof(receivedInt), MSG_WAITALL);

		if(bytes < -1){
			error("SERVER", "Fallo al recibir size");
			end = true;
		} else if(bytes == 0){
			log("SERVER", "Conexion cerrada");
			end = true;
		} else{

			/* Size received -> We receive the image encoded */
			receivedInt = ntohl(receivedInt);

			log("SERVER", "Recibiendo imagen... (size " + intToString(receivedInt) + ")");

			vector<uchar> receivedEncoded(receivedInt);

			bytes = recv(socket, receivedEncoded.data(), receivedEncoded.size(), MSG_WAITALL);

			if(bytes < -1){
				error("SERVER", "Fallo al recibir imagen");
				end = true;
			} else if(bytes == 0){
				log("SERVER", "Conexion cerrada");
				end = true;
			} else{

				/* Image received -> We decode and process it */
				log("SERVER", "Imagen recibida " + intToString(i));

				high_resolution_clock::time_point endingReceiving = high_resolution_clock::now();
				auto durationReceiving = duration_cast<microseconds>( endingReceiving - beginReceiving ).count();
				timerThread.addDuration("RECEIVING", durationReceiving);


				high_resolution_clock::time_point beginDecoding = high_resolution_clock::now();
				imgColor = imdecode(receivedEncoded, CV_LOAD_IMAGE_COLOR);
				high_resolution_clock::time_point endingDecoding = high_resolution_clock::now();
				auto durationDecoding = duration_cast<microseconds>( endingDecoding - beginDecoding ).count();
				timerThread.addDuration("DECODING", durationDecoding);


				high_resolution_clock::time_point beginProcessing = high_resolution_clock::now();

				cvtColor(imgColor,imgGray,CV_BGR2GRAY);

				Result imgResult;
				if(deep_learning){
					imgResult = recognizerDl.recognizeObject(imgColor,imgDest, myMutex);
				} else{
					imgResult = recognizer.recognizeObject(imgGray,imgColor,imgDest);
				}
				high_resolution_clock::time_point endingProcessing = high_resolution_clock::now();
				auto durationProcessing = duration_cast<microseconds>( endingProcessing - beginProcessing ).count();
				timerThread.addDuration("PROCESSING - Total", durationProcessing);

				string output;

				high_resolution_clock::time_point beginSerializing = high_resolution_clock::now();
				imgResult.serializeResult(&output);
				high_resolution_clock::time_point endingSerializing  = high_resolution_clock::now();
				auto durationSerializing  = duration_cast<microseconds>( endingSerializing  - beginSerializing  ).count();
				timerThread.addDuration("SERIALIZING", durationSerializing);

				int encodedSize = htonl(output.size());

				high_resolution_clock::time_point beginSending = high_resolution_clock::now();

				log("SERVER", "Enviando size... (size " + intToString(output.size()) + ")");
				bytes = send(socket, (const char*)&encodedSize, sizeof(encodedSize), 0);

				if(bytes < 0){
					error("SERVER", "Fallo al enviar size");
					end = true;
				} else if(bytes == 0){
					log("SERVER", "Conexion cerrada");
					end = true;
				} else{

					/* Size sent -> We send the image serialized */
					log("SERVER", "Enviando imagen...");
					bytes = send(socket, &output[0], output.size(), 0);
					if(bytes < 0){
						error("SERVER", "Fallo al enviar imagen");
						end = true;
					} else if(bytes == 0){
						log("SERVER", "Conexion cerrada");
						end = true;
					} else{
						high_resolution_clock::time_point endingSending = high_resolution_clock::now();
						auto durationSending = duration_cast<microseconds>( endingSending - beginSending ).count();
						timerThread.addDuration("SENDING", durationSending);
						auto durationTotal = duration_cast<microseconds>( endingSending - beginReceiving ).count();
						timerThread.addDuration("TOTAL", durationTotal);

						log("SERVER", "Imagen enviada");
						i++;
						if(i == 150){
							error("Reiniciamos", "reiniciamos");
							i = 0;
						}
					}
				}

			}

		}
	}

	timer.printTimes();
	timerThread.printTimes();
	recognizer.printTimer();
	recognizerDl.printTimer();
//	log("RESULTADOS", "Duracion total: " + longToString(totalDuration) + " en " + intToString(iterations) + " iteraciones (Media: " + longToString(totalDuration/iterations) + ")");

	return 0;
}

void *processInt(void *ptr){

	int socket = *(int *)ptr;
	bool end = false;

	/* Receive image */
	int converted_int;
	int final_int;
	int sending_int;
	int bytes = 0;

	long long int totalDuration = 0.0;
	int iterations = 0;

	while(!end){

		log("SERVER", "Recibiendo entero...");
		bytes = recv(socket, &converted_int, 4, MSG_WAITALL);

		if(bytes < 0){
			error("SERVER", "Fallo al recibir");
			end = true;
		} else{
			log("SERVER", "entero recibido");

			final_int = ntohl(converted_int);
			log("SERVER", "RECIBIDO " + intToString(final_int) );
			final_int = final_int + 1;
			sending_int = htonl(final_int);

			/* Enviar imagen */
			log("SERVER", "Enviando entero...");
			bytes = send(socket, (const char*)&sending_int, sizeof(sending_int), 0);
			if(bytes < 0){
				error("SERVER", "Error al enviar");
				end = true;
			} else{
				log("SERVER", "Entero enviado " + intToString(final_int));
			}

		}
	}

	log("RESULTADOS", "Duracion total: " + longToString(totalDuration) + " en " + intToString(iterations) + " iteraciones (Media: " + longToString(totalDuration/iterations) + ")");

	return 0;
}
