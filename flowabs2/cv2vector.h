//cv2vector.h

#ifndef CV2VECTOR_H
#define CV2VECTOR_H

#include "vector.h"
#include <cv.h>

void ucImg2frgb( IplImage* src, vec3* dest );
void frgb2ucImg( vec3* src, IplImage* dest );
void frgb2fcImg( vec3* src, IplImage* dest );
void f2fgImg( double* src, IplImage* dest );

#endif
