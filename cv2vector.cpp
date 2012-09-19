#include "cv2vector.h"


void ucImg2frgb( IplImage* src, vec3* dest )
{
	if ( src && dest)
	{
		for ( int y = 0; y < src->height; y++ )
		{
			uchar* ptr = (uchar*) ( src->imageData + y * src->widthStep);
			for (int x=0; x < src->width; x++ )
			{
				dest[ y * src->width + x].b = ptr[ 3 * x + 0] / 255.0;
				dest[ y * src->width + x].g = ptr[ 3 * x + 1] / 255.0;
				dest[ y * src->width + x].r = ptr[ 3 * x + 2] / 255.0;
			}
		}
	}
}

void frgb2ucImg( vec3* src, IplImage* dest )
{
	if ( src && dest)
	{
		for ( int y = 0; y < dest->height; y++ )
		{
			uchar* ptr = (uchar*) ( dest->imageData + y * dest->widthStep);
			for (int x=0; x < dest->width; x++ )
			{
				ptr[ 3 * x + 0] = clamp( src[ y * dest->width + x].b ) * 255.0;
				ptr[ 3 * x + 1] = clamp( src[ y * dest->width + x].g ) * 255.0;
				ptr[ 3 * x + 2] = clamp( src[ y * dest->width + x].r ) * 255.0;
			}
		}
	}
}

void frgb2fcImg( vec3* src, IplImage* dest )
{
	if ( src && dest)
	{
		for ( int y = 0; y < dest->height; y++ )
		{
			double* ptr = (double*) ( dest->imageData + y * dest->widthStep);
			for (int x=0; x < dest->width; x++ )
			{
				ptr[ 3 * x + 0] = src[ y * dest->width + x].b;
				ptr[ 3 * x + 1] = src[ y * dest->width + x].g;
				ptr[ 3 * x + 2] = src[ y * dest->width + x].r;
			}
		}
	}
}

void f2fgImg( double* src, IplImage* dest )
{
	if ( src && dest)
	{
		for ( int y = 0; y < dest->height; y++ )
		{
			double* ptr = (double*) ( dest->imageData + y * dest->widthStep);
			for (int x=0; x < dest->width; x++ )
			{
				ptr[ x + 0] = src[ y * dest->width + x];
			}
		}
	}
}


