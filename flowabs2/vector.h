//vector.h

#ifndef VECTOR_H
#define VECTOR_H

#include <math.h>

struct vec2
{
	double x;
	double y;
};

struct vec3
{
	double r;
	double g;
	double b;
};


//Clamp every vec component to value ( min - max )
inline double clamp( double c, double min = 0.0, double max = 1.0 )
{
	return c < min ? min : ( c > max ? max : c );
}

inline vec2 clamp( vec2 c, double min = 0.0, double max = 1.0 )
{
	vec2 tmp;
	tmp.x =c.x < min ? min : ( c.x > max ? max : c.x );
	tmp.y =c.y < min ? min : ( c.y > max ? max : c.y );

	return tmp;
}

inline vec3 clamp( vec3 c, double min = 0.0 , double max = 1.0 )
{
	vec3 tmp;
	tmp.r = c.r < min ? min : ( c.r > max ? max : c.r );
	tmp.g = c.g < min ? min : ( c.g > max ? max : c.g );
	tmp.b = c.b < min ? min : ( c.b > max ? max : c.b );

	return tmp;
}

//return the length of vec
inline double length( vec2 c )
{
	return sqrt ( c.x * c.x + c.y * c.y );
}

inline double length( vec3 c )
{
	return sqrt ( c.r * c.r + c.g * c.g + c.b * c.b );
}



//return normalized vec
inline vec2 normalize( vec2 c )
{
	double len = length( c );
	vec2 n;
	n.x = c.x / len;
	n.y = c.y / len;
	return n;
}

inline vec3 normalize( vec3 c )
{
	double len = length( c );
	vec3 n;
	n.r = c.r / len;
	n.g = c.g / len;
	n.b = c.b / len;
	return n;
}

//return dot
inline double dot( vec2 a, vec2 b )
{
	return a.x * b.x + a.y * b.y;
}

inline double dot( vec3 a, vec3 b )
{
	return a.r * b.r + a.g * b.g + a.b * b.b;
}

//return abs

inline double dabs( double d){
     return d>0?d:-d;
}
inline vec2 abs( vec2 c )
{
	vec2 a = { dabs( c.x ), dabs( c.y ) };
	return a;
}

inline vec3 abs( vec3 c )
{
	vec3 a = { dabs( c.r ), dabs( c.g ), dabs( c.b ) };
	return a;
}

//return smooth step
inline double smoothstep( double e0, double e1, double x )
{
	double t = clamp( ( x - e0 ) / ( e1 - e0 ) ) ;
	return t * t * ( 3 - 2 * t );
}

#endif
