//#include "windows.h"
//#include "stdio.h"
//
//#ifndef MAX
//#define MAX(A,B,C) A>B?A:B>C?A>B?A:B:C
//#endif
//
//#ifndef MIN
//#define MIN( A, B, C ) A<B?A:B<C?A<B?A:B:C
//#endif
//
//void RGBtoHSV( BYTE r, BYTE g, BYTE b, float &h, float &s, float &v ) 
//{ 
//	BYTE min, max; 
//	float delta; 
//	min = MIN( r, g, b ); 
//	max = MAX( r, g, b ); 
//	v = max; // v 
//	delta = max - min; 
//	if( max != 0 ) 
//	{ 
//		s = delta / max; // s 
//	} 
//	else 
//	{ 
//		// r = g = b = 0 
//		// s = 0, v is undefined 
//		s = 0; 
//		h = -1; 
//		return; 
//	} 
//	if( r == max ) 
//	{
//		h = ( g - b ) / delta; // between yellow & magenta 
//	} 
//	else if( g == max ) 
//	{ 
//		h = 2 + ( b - r ) / delta; // between cyan & yellow 
//	} else 
//	{ 
//		h = 4 + ( r - g ) / delta; // between magenta & cyan 
//	}
//	h *= 60; // degrees 
//	if( h < 0 ) 
//	{ 
//		h += 360; 
//	} 
//}
//
//
//int main( int argc, char** argv )
//{
//	BYTE r,g,b;
//	float h, s, v;
//	while(1)
//	{
//		printf("input rgb: ");
//		scanf("%d%d%d", &r,&g,&b);
//		RGBtoHSV( r, g, b, h, s ,v );
//		printf("output hsv: %f %f %f\n", h, s, v );
//	}
//	return 0;
//}