#INCPATH = -I/usr/local/include/opencv
#LIBPATH = -L/usr/local/lib
#OPTIONS = -lcv -lcvaux -lcxcore -lhighgui -lstdc++ -Wl,--rpath -Wl,/usr/local/lib
#OPTIONS = -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_ts -lopencv_video -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_contrib -lstdc++ -Wl,--rpath -Wl,/usr/local/lib

OPTION = 
1: main.o flowabs.o cv2vector.o libCudaCalc.so
	g++ $(OPTION) main.o flowabs.o cv2vector.o  -L. -lCudaCalc -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_video -lopencv_features2d -lopencv_ml -lopencv_highgui -lopencv_objdetect -lopencv_contrib -lopencv_legacy  -o 1
main.o: main.cpp cv2vector.h flowabs.h cv2vector.cpp flowabs.cpp vector.h
	g++ $(OPTION) -c main.cpp  -I/usr/local/include/opencv
flowabs.o: flowabs.h flowabs.cpp vector.h
	g++ $(OPTION) -c flowabs.cpp
cv2vector.o: cv2vector.cpp cv2vector.h vector.h
	g++ $(OPTION) -c cv2vector.cpp  -I/usr/local/include/opencv

# 1:1.o libCudaCalc.so
# g++ 1.o -L. -lCudaCalc -o 1

# 1.o:1.cc
# g++ -c 1.cc

# libCudaCalc.so: CudaCalc.cu
# nvcc --compiler-options '-fPIC' --ptxas-options=-v -arch=compute_20 -code=sm_20 -o libCudaCalc.so --shared CudaCalc.cu

# .PHONY: cuda	
# cuda: libCudaCalc.so
libCudaCalc.so: flowabs.cu
	nvcc --compiler-options '-fPIC' --ptxas-options=-v --use_fast_math -arch=compute_20 -code=sm_20 -o libCudaCalc.so --shared flowabs.cu
