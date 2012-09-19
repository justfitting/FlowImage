#include<cstdio>
#include<cstdlib>
#include<cuda.h>
#include"vector.h"
#include <iostream>
#include<sys/time.h>
using namespace std;
/*
  
 */
//Bilinear interpolation
#define INTERPOLATE_IMAGE 0
#define INTERPOLATE_NORM  1
#define EPLISION 0.0000001f
/*__device__ int halfWidth_d;*/
/*__device__ float* kernel_d = NULL;*/


__device__
float cu_dabs( float d){
     return d>0?d:-d;
}

__device__
float2 abs2( float2 c )
{
     float2 a = { cu_dabs( c.x ), cu_dabs( c.y ) };
     return a;
}
__device__
float length3( float3 c){
     return sqrtf ( c.x * c.x + c.y * c.y + c.z * c.z );
}
__device__
float3 normalize( float3 c ){
     float len = length3(c);
     float3 n;
     if(fabs(len)>EPLISION){
          n.x = c.x / len;
          n.y = c.y / len;
          n.z = c.z / len;
     }else{
          
     }
     return n;
}

__device__
float dclamp(float c,  float min = 0.0 , float max = 1.0 ){
     return c < min ? min : ( c > max ? max : c );
}
__device__
float3 clamp(float3 c,  float min = 0.0 , float max = 1.0 ){
     float3 tmp;
     tmp.x = c.x < min ? min : ( c.x > max ? max : c.x );
     tmp.y = c.y < min ? min : ( c.y > max ? max : c.y );
     tmp.z = c.z < min ? min : ( c.z > max ? max : c.z );
     
     return tmp;
}

__device__
float3 bilinear_interpolate( float3* img, float2 p, int width, int height, int flag = INTERPOLATE_IMAGE )
{
     // const unsigned int x = blockIdx.x;
     // const unsigned int y = threadIdx.x;
     
     float2 q = { dclamp( p.x, 0.0, width - 1.1 ), dclamp( p.y, 0.0, height - 1.1 ) };//prevent being out of map

     int qx = int ( q.x );
     int qy = int ( q.y );

     float tx = qx + 1 -q.x;
     float ty = qy + 1 -q.y;

     float3 result;
     switch (flag)
     {
     case INTERPOLATE_IMAGE:
          result.x = ( img[ qy * width + qx ].x * tx + img[ qy * width + qx + 1 ].x * ( 1 - tx ) ) * ty +
               ( img[ ( qy + 1 ) * width + qx ].x * tx + img[ ( qy + 1 ) * width + qx + 1 ].x * ( 1 - tx ) ) * ( 1 - ty ); 
          result.y = ( img[ qy * width + qx ].y * tx + img[ qy * width + qx + 1 ].y * ( 1 - tx ) ) * ty +
               ( img[ ( qy + 1 ) * width + qx ].y * tx + img[ ( qy + 1 ) * width + qx + 1 ].y * ( 1 - tx ) ) * ( 1 - ty ); 
          result.z = ( img[ qy * width + qx ].z * tx + img[ qy * width + qx + 1 ].z * ( 1 - tx ) ) * ty +
               ( img[ ( qy + 1 ) * width + qx ].z * tx + img[ ( qy + 1 ) * width + qx + 1 ].z * ( 1 - tx ) ) * ( 1 - ty );
          break;
     case INTERPOLATE_NORM:
          result.x = ( img[ qy * width + qx ].x * tx + img[ qy * width + qx + 1 ].x * ( 1 - tx ) ) * ty +
               ( img[ ( qy + 1 ) * width + qx ].x * tx + img[ ( qy + 1 ) * width + qx + 1 ].x * ( 1 - tx ) ) * ( 1 - ty ); 
          result.y = ( img[ qy * width + qx ].y * tx + img[ qy * width + qx + 1 ].y * ( 1 - tx ) ) * ty +
               ( img[ ( qy + 1 ) * width + qx ].y * tx + img[ ( qy + 1 ) * width + qx + 1 ].y * ( 1 - tx ) ) * ( 1 - ty ); 
          result.z = ( img[ qy * width + qx ].z * tx + img[ qy * width + qx + 1 ].z * ( 1 - tx ) ) * ty +
               ( img[ ( qy + 1 ) * width + qx ].z * tx + img[ ( qy + 1 ) * width + qx + 1 ].z * ( 1 - tx ) ) * ( 1 - ty );
	
     }
     return result;

}


// __device__
// float length( float3 c )
// {
// 	return sqrtf ( c.x * c.x + c.y * c.y + c.z * c.z );
// }
__global__
void cu_rgb2lab(float3* src, float3* dest, int width, int height){
     const int x = blockIdx.x*blockDim.x+threadIdx.x;
     const int y = blockIdx.y*blockDim.y+threadIdx.y;
     if(x>=width || y>=height) return;
     float3 c, tmp, xyz, n, v, lab;
     c = src[ y * width + x];
     tmp.x = ( c.x > 0.04045 ) ? powf( ( c.x + 0.055 ) / 1.055, 2.4 ) : c.x / 12.92;
     tmp.y = ( c.y > 0.04045 ) ? powf( ( c.y + 0.055 ) / 1.055, 2.4 ) : c.y / 12.92,
          tmp.z = ( c.z > 0.04045 ) ? powf( ( c.z + 0.055 ) / 1.055, 2.4 ) : c.z / 12.92;

     xyz.x = 100.0 * ( tmp.x * 0.4124 + tmp.y * 0.3576 + tmp.z * 0.1805 ) ;
     xyz.y = 100.0 * ( tmp.x * 0.2126 + tmp.y * 0.7152 + tmp.z * 0.0722 ) ;
     xyz.z = 100.0 * ( tmp.x * 0.0193 + tmp.y * 0.1192 + tmp.z * 0.9505 ) ;

     n.x = xyz.x / 95.047;
     n.y = xyz.y / 100;
     n.z = xyz.z / 108.883;

     v.x = ( n.x > 0.008856 ) ? powf( n.x, 1.0 / 3.0 ) : ( 7.787 * n.x ) + ( 16.0 / 116.0 );
     v.y = ( n.y > 0.008856 ) ? powf( n.y, 1.0 / 3.0 ) : ( 7.787 * n.y ) + ( 16.0 / 116.0 );
     v.z = ( n.z > 0.008856 ) ? powf( n.z, 1.0 / 3.0 ) : ( 7.787 * n.z ) + ( 16.0 / 116.0 );

     lab.x = ( 116.0 * v.y ) - 16.0;
     lab.y =  500.0 * ( v.x - v.y );
     lab.z = 200.0 * ( v.y - v.z );

     dest[ y * width + x ].x = lab.x / 100.0;
     dest[ y * width + x ].y = 0.5 + 0.5 * ( lab.y / 127.0 );
     dest[ y * width + x ].z = 0.5 + 0.5 * ( lab.z / 127.0 );
}

__global__
void cu_lab2rgb(float3* src, float3* dest, int width, int height){
     // const  int x = blockIdx.x;
     // const  int y = threadIdx.x;
     const int x = blockIdx.x*blockDim.x+threadIdx.x;
     const int y = blockIdx.y*blockDim.y+threadIdx.y;
     if(x>=width || y>=height) return;
     float3 c, tmp, xyz, r, v, lab;
     c = clamp ( src[ y * width + x] );
     lab.x = 100.0 * c.x;
     lab.y = 2.0 * 127.0 * ( c.y - 0.5 );
     lab.z = 2.0 * 127.0 * ( c.z - 0.5 );

     tmp.y = ( lab.x + 16.0 ) / 116.0;
     tmp.x = lab.y / 500.0 + tmp.y;
     tmp.z = tmp.y - lab.z / 200.0;

     xyz.x = 95.047 * (( tmp.x > 0.206897 ) ? tmp.x * tmp.x * tmp.x : ( tmp.x -16.0 / 116.0 ) / 7.787);
     xyz.y = 100.000 * (( tmp.y > 0.206897 ) ? tmp.y * tmp.y * tmp.y : ( tmp.y -16.0 / 116.0 ) / 7.787);
     xyz.z = 108.883 * (( tmp.z > 0.206897 ) ? tmp.z * tmp.z * tmp.z : ( tmp.z -16.0 / 116.0 ) / 7.787);

     v.x = ( xyz.x * 3.2406 + xyz.y * -1.5372 + xyz.z * -0.4986 ) / 100.0;
     v.y = ( xyz.x * -0.9689 + xyz.y * 1.8758 + xyz.z * 0.0415 ) / 100.0;
     v.z = ( xyz.x * 0.0557 + xyz.y * -0.2040 + xyz.z * 1.0570 ) / 100.0;

     r.x = ( v.x > 0.0031308 ) ? (( 1.055 * powf( v.x, (1.0 / 2.4 ))) - 0.055 ) : 12.92 * v.x;
     r.y = ( v.y > 0.0031308 ) ? (( 1.055 * powf( v.y, (1.0 / 2.4 ))) - 0.055 ) : 12.92 * v.y;
     r.z = ( v.z > 0.0031308 ) ? (( 1.055 * powf( v.z, (1.0 / 2.4 ))) - 0.055 ) : 12.92 * v.z;

     dest[ y * width + x ] = clamp( r );
}

__global__
void cu_structure_tensor(float3* src, float3* st, int width, int height){
     const int x = blockIdx.x*blockDim.x+threadIdx.x;
     const int y = blockIdx.y*blockDim.y+threadIdx.y;
     if(x>=width || y>=height || x<0 || y<0) return;
     float3 u,v;
     int ym1 = ( y - 1 ) < 0 ? 0 : ( y - 1 );
     int xm1 = ( x - 1 ) < 0 ? 0 : ( x - 1 );
     int yp1 = ( y + 1 ) > ( height - 1 ) ? ( height - 1 ) : ( y + 1 );
     int xp1 = ( x + 1 ) > ( width  - 1 ) ? ( width  - 1 ) : ( x + 1 );

     // x gradient
     u.x = ( 
          -1.0 * src[ ym1 * width + xm1 ].x + 
          -2.0 * src[ y   * width + xm1 ].x +
          -1.0 * src[ yp1 * width + xm1 ].x +
          +1.0 * src[ ym1 * width + xp1 ].x + 
          +2.0 * src[ y   * width + xp1 ].x +
          +1.0 * src[ yp1 * width + xp1 ].x ) / 4.0;

     u.y = ( 
          -1.0 * src[ ym1 * width + xm1 ].y + 
          -2.0 * src[ y   * width + xm1 ].y +
          -1.0 * src[ yp1 * width + xm1 ].y +
          +1.0 * src[ ym1 * width + xp1 ].y + 
          +2.0 * src[ y   * width + xp1 ].y +
          +1.0 * src[ yp1 * width + xp1 ].y ) / 4.0;

     u.z = ( 
          -1.0 * src[ ym1 * width + xm1 ].z + 
          -2.0 * src[ y   * width + xm1 ].z +
          -1.0 * src[ yp1 * width + xm1 ].z +
          +1.0 * src[ ym1 * width + xp1 ].z + 
          +2.0 * src[ y   * width + xp1 ].z +
          +1.0 * src[ yp1 * width + xp1 ].z ) / 4.0;

     //y gradient
     v.x = ( 
          -1.0 * src[ ym1 * width + xm1 ].x + 
          -2.0 * src[ ym1 * width + x ].x +
          -1.0 * src[ ym1 * width + xp1 ].x +
          +1.0 * src[ yp1 * width + xm1 ].x + 
          +2.0 * src[ yp1 * width + x ].x +
          +1.0 * src[ yp1 * width + xp1 ].x ) / 4.0;

     v.y = ( 
          -1.0 * src[ ym1 * width + xm1 ].y + 
          -2.0 * src[ ym1 * width + x ].y +
          -1.0 * src[ ym1 * width + xp1 ].y +
          +1.0 * src[ yp1 * width + xm1 ].y + 
          +2.0 * src[ yp1 * width + x ].y +
          +1.0 * src[ yp1 * width + xp1 ].y ) / 4.0;

     v.z = ( 
          -1.0 * src[ ym1 * width + xm1 ].z + 
          -2.0 * src[ ym1 * width + x ].z +
          -1.0 * src[ ym1 * width + xp1 ].z +
          +1.0 * src[ yp1 * width + xm1 ].z + 
          +2.0 * src[ yp1 * width + x ].z +
          +1.0 * src[ yp1 * width + xp1 ].z ) / 4.0;

     //structure tensor
     st[ y * width + x ].x = u.x * u.x + u.y * u.y + u.z * u.z;
     st[ y * width + x ].y = v.x * v.x + v.y * v.y + v.z * v.z;
     st[ y * width + x ].z = u.x * v.x + u.y * v.y + u.z * v.z;
}

__global__
void cu_gauss_filter(float3* src, float sigma, float3* dest, int width, int height, int halfWidth_d, float *kernel_d){
     const int x = blockIdx.x*blockDim.x+threadIdx.x;
     const int y = blockIdx.y*blockDim.y+threadIdx.y;
	 if(x>=width || y>=height || x<0 || y<0) return;

     float3 result = {0.,0.,0.};
     float norm = 0.;
     // parallel
     for ( int i = -halfWidth_d; i<= halfWidth_d; ++i )
     {
          for ( int j = -halfWidth_d; j<= halfWidth_d; ++j )
          {
			   //if (blockIdx.x == 0 && blockIdx.y == 0)
               /*printf("id = %d, (%f, %f, %f), norm = %f\n", y * width + x, dest[ y * width + x ].x, dest[ y * width + x ].y, dest[ y * width + x ].z, norm);*/
               /*printf("bidx.x = %d, bidx.y = %d\n", blockIdx.x, blockIdx.y);*/
               if ( ( y + i ) >= 0 && ( x + j ) >= 0 && ( y + i ) < height && ( x + j ) < width )// judge whether out of map
               {
                    result.x += src[ ( y + i ) * width + x + j ].x * kernel_d[ ( i + halfWidth_d ) * ( halfWidth_d * 2 + 1 ) + j + halfWidth_d ];
                    result.y += src[ ( y + i ) * width + x + j ].y * kernel_d[ ( i + halfWidth_d ) * ( halfWidth_d * 2 + 1 ) + j + halfWidth_d ];
                    result.z += src[ ( y + i ) * width + x + j ].z * kernel_d[ ( i + halfWidth_d ) * ( halfWidth_d * 2 + 1 ) + j + halfWidth_d ];
                    norm += kernel_d[ ( i + halfWidth_d ) * ( halfWidth_d * 2 + 1 ) + j + halfWidth_d ];
               }
          }
     }
     if(fabs(norm)>EPLISION){
          dest[ y * width + x ].x = result.x / norm;
          dest[ y * width + x ].y = result.y / norm;
          dest[ y * width + x ].z = result.z / norm;
     }else{
          dest[ y * width + x ].x = 0.;
          dest[ y * width + x ].y = 0.;
          dest[ y * width + x ].z = 0.;
     }
	 /*if (blockIdx.x == 0 && blockIdx.y == 0)
	 /*{*/
     /*printf("id = %d, (%f, %f, %f), norm = %f\n", y * width + x, dest[ y * width + x ].x, dest[ y * width + x ].y, dest[ y * width + x ].z, norm);*/
	 /*}*/
     //
	 /*dest[ y * width + x ].x = 0.123;*/
	 /*dest[ y * width + x ].y = 0.123;*/
	 /*dest[ y * width + x ].z = 0.123;*/
}

__global__
void cu_tangent_flow_map(float3* sst, float sigma, float3* tfm, int width, int height ){
     const int x = blockIdx.x*blockDim.x+threadIdx.x;
     const int y = blockIdx.y*blockDim.y+threadIdx.y;
     if(x>=width || y>=height) return;
     float3 g = sst[ y * width + x ];
     float lambda1 = 0.5 * ( g.y + g.x + sqrt( g.y * g.y - 2.0 * g.x * g.y + g.x * g.x + 4.0 * g.z * g.z ) );
     float3 v = { g.x - lambda1, g.z, 0.0 };

     if ( length3( v ) > 0.0 )
     {
          tfm[ y * width + x ] = normalize( v );
          tfm[ y * width + x ].z = sqrt( lambda1 );//may be used for weight???????????????????????????????????????????
     }
     else//ensure no zero floattor
     {
          tfm[ y * width + x ].x = 0.0;
          tfm[ y * width + x ].y = 1.0;
          tfm[ y * width + x ].z = 0.0;
     }
}

__global__
void cu_orientation_aligned_bilateral_filter( float3*src, float3* tfm, float3*dest, float3* tmp, int n, float sigma_d, float sigma_r, int width, int height){
     const int x = blockIdx.x*blockDim.x+threadIdx.x;
     const int y = blockIdx.y*blockDim.y+threadIdx.y;
     if(x>=width || y>=height) return;
     float twoSigmaD2 = 2.0 * sigma_d * sigma_d;
     float twoSigmaR2 = 2.0 * sigma_r * sigma_r;
     for(int i = 0; i < n; ++i)
     {
          for(int pass = 0; pass < 2; ++pass)
          {
               float2 t = { tfm[ y * width + x ].x, tfm[ y * width + x ].y };
               float2 tt = { t.y, -t.x };
               float2 dir = ( pass == 0 ) ? tt : t;
               float2 dabs = abs2( dir );
               float ds = 1.0 / ( ( dabs.x > dabs.y ) ? dabs.x : dabs.y );

               float3* midsrc = ( i == 0 ? ( pass == 0 ? src : tmp ) : ( pass == 0 ? dest : tmp )  );
               float3 center = midsrc[ y * width + x ];
               float3 sum = center;
               float norm = 1.0;
               float halfWidth = 2.0 * sigma_d;

               for ( float d = ds; d <= halfWidth; d += ds )
               {
                    float2 p0 = { x + d * dir.x, y + d * dir.y };
                    float3 c0 = bilinear_interpolate( midsrc, p0, width, height );
                    float2 p1 = { x - d * dir.x, y - d * dir.y };
                    float3 c1 = bilinear_interpolate( midsrc, p1, width, height );


                    float3 d0 = { c0.x -center.x, c0.y - center.y, c0.z - center.z };
                    float3 d1 = { c1.x -center.x, c1.y - center.y, c1.z - center.z };
                    float e0 = length3( d0 );
                    float e1 = length3( d1 );

                    float kerneld = expf( -d * d / twoSigmaD2 );
                    float kernele0 = expf( - e0 * e0 / twoSigmaR2 );
                    float kernele1 = expf( - e1 * e1 / twoSigmaR2 );	
                    norm += kerneld * kernele0;
                    norm += kerneld * kernele1;
	 				//printf("%lf, %lf, %lf\n", kerneld, kernele0, kernele1);

                    sum.x += kerneld * kernele0 * c0.x;
                    sum.x += kerneld * kernele1 * c1.x;
                    sum.y += kerneld * kernele0 * c0.y;
                    sum.y += kerneld * kernele1 * c1.y;
                    sum.z += kerneld * kernele0 * c0.z;
                    sum.z += kerneld * kernele1 * c1.z;
               }
               if(fabs(norm)>EPLISION){
                    sum.x /= norm;
                    sum.y /= norm;
                    sum.z /= norm;
               }else{
                    sum.x /= 1;
                    sum.y /= 1;
                    sum.z /= 1;
               }

               ( pass == 0 ? tmp : dest )[ y * width + x ] = sum ;
          }
     }
}

void cudaSafeFree(void** p){
     if(*p){
          cudaFree(p);
     }
}
void cleanup(float3* fsrc, vec3* dest, int size){
     if(fsrc&&dest){
          printf("clean up\n");
          for(int i = 0; i < size; ++i)
          {
               dest[i].r = fsrc[i].x;
               dest[i].g = fsrc[i].y;
               dest[i].b = fsrc[i].z;
          }

     }else{
          printf("ZERO\n");
     }
}

void print(float3* fsrc, int width, int height){
     if(fsrc){
          for(int i = 0; i < height; ++i)
          {
               for(int j = 0; j < width; ++j)
               {
                    fprintf(stdout, "(%.6f %.6f %.6f) ", fsrc[i*width+j].x, fsrc[i*width+j].y, fsrc[i*width+j].z);
               }
               fprintf(stdout, "\n");
          }
     }
}

void printd(float* d,int size){
     if(d){
          printf("%d\n",size);
          for(int i = 0; i < size; ++i)
          {
               fprintf(stdout, "%.6f ", d[i]);
          }
     }else{
          printf("ZERO\n");
     }
}


void creat_gauss_kernel(float** ker, float sigma, int* half){
     float twoSigma2 =  2.0 * sigma * sigma;
     int halfWidth = int ( ceil( 2.0 * sigma ) );
	 cout << "halfwidth = " << halfWidth << endl;
     *half = halfWidth;
     float* kernel = new float[(2*halfWidth+1)*(2*halfWidth+1)];
     *ker = kernel;
     float norm = 0.0;
     for ( int i = -halfWidth; i <= halfWidth; i++ )
     {
          for ( int j = -halfWidth; j <= halfWidth; j++)
          {
               norm += kernel[ ( i + halfWidth ) * ( halfWidth * 2 + 1 ) + j + halfWidth ] = exp( - ( i * i + j * j ) / twoSigma2 );
          }
     }
     for ( int i = -halfWidth; i <= halfWidth; i++ )
     {
          for ( int j = -halfWidth; j <= halfWidth; j++)
          {
               kernel[ ( i + halfWidth ) * ( halfWidth * 2 + 1 ) + j + halfWidth ] /= norm ;
          }
     }
     //printd(kernel, (2*halfWidth+1)*(2*halfWidth+1));
}
#define NO 900
void cu_flowabs(vec3* src, vec3* dest, float sigma, vec3* tfm, int width, int height){
     if(!(src && dest && tfm)){
          return;
     }
     if(width*height>1024*1024){
          printf("Too Large Image\n");
          return;
     }
	 struct timeval ts;
	 struct timezone tz;
	 gettimeofday (&ts , &tz);
	 long sec = ts.tv_sec;
	 long usec = ts.tv_usec;
	 
     dim3 dimBlock(16,16);
     dim3 dimGrid((width+dimBlock.x-1)/dimBlock.x, (height+dimBlock.y-1)/dimBlock.y);
	 cout << "dim = (" << dimGrid.x << ", " << dimGrid.y << ")" << endl;
     
     float3* st;
     cudaMalloc((void**)&st, sizeof(float3)*width*height); // structure tensor
     cudaError_t err = cudaGetLastError();
     if(err)
          cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;

     float3* tmp;
     cudaMalloc((void**)&tmp,  sizeof(float3)*width*height);
     err = cudaGetLastError();
     if(err)
          cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;

     float3* sst; // smoothed structure tensor
     cudaMalloc((void**)&sst,  sizeof(float3)*width*height);
     err = cudaGetLastError();
     if(err)
          cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;

     float3* ftfm; // 
     cudaMalloc((void**)&ftfm,  sizeof(float3)*width*height);
     err = cudaGetLastError();
     if(err)
          cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;

     float3* lab;
     cudaMalloc((void**)&lab,  sizeof(float3)*width*height);
     err = cudaGetLastError();
     if(err)
          cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;

     float3* midlab;
     cudaMalloc((void**)&midlab,  sizeof(float3)*width*height);
     err = cudaGetLastError();
     if(err)
          cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;

     float3* rgb;
     cudaMalloc((void**)&rgb,  sizeof(float3)*width*height);
     err = cudaGetLastError();
     if(err)
          cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;
     
	 float3* fsrc = 0;
     fsrc = new float3[width*height];
	 //cout << "fsrc = " << (long)fsrc << endl;
     for(int i = 0; i < width*height; ++i)
     {
          fsrc[i].x = src[i].r;
          fsrc[i].y = src[i].g;
          fsrc[i].z = src[i].b;
     }
     //printf("%.6f %.6f %.6f\n", fsrc[NO].x,fsrc[NO].y, fsrc[NO].z);
     cudaMemcpy(tmp, fsrc, sizeof(float3)*width*height, cudaMemcpyHostToDevice);
     err = cudaGetLastError();
     if(err)
          cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;

     // structure_tensor
     cu_structure_tensor<<<dimGrid, dimBlock>>>(tmp, st, width, height);
     err = cudaGetLastError();
     if(err)
          printf("line : %d, %s\n", __LINE__, cudaGetErrorString(err));


     // 
     cudaMemcpy(fsrc, st, sizeof(float3)*width*height, cudaMemcpyDeviceToHost);
     //printf("%.6f %.6f %.6f\n", fsrc[NO].x,fsrc[NO].y, fsrc[NO].z);
     // print(fsrc, width, height);
     err = cudaGetLastError();
     if(err)
          printf("%d %s\n", err, cudaGetErrorString(err));

     // Creat gauss kernal
     float* kernel = NULL;
     int halfWidth;
     creat_gauss_kernel(&kernel, sigma, &halfWidth);
     //
     //printd(kernel, (2*halfWidth+1)*(2*halfWidth+1));
     //halfWidth_d = halfWidth;
	 float *kernel_d;
     cudaMalloc((void**)&kernel_d,  sizeof(float)*(2*halfWidth+1)*(2*halfWidth+1));
     err = cudaGetLastError();
     if(err)
          cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;

     cudaMemcpy(kernel_d, kernel, sizeof(float)*(2*halfWidth+1)*(2*halfWidth+1), cudaMemcpyHostToDevice);
     err = cudaGetLastError();
     if(err)
          cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;
     
     printf("line : %d, dimGrid %d %d \n", __LINE__ , dimGrid.x, dimGrid.y);
     cu_gauss_filter<<<dimGrid, dimBlock>>>(st, sigma, sst, width, height, halfWidth, kernel_d);
     err = cudaGetLastError();
     if(err)
          cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;

     cudaMemcpy(fsrc, sst, sizeof(float3)*width*height, cudaMemcpyDeviceToHost);
     err = cudaGetLastError();
     if(err)
          cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;
     printf("line : %d, error = %d %.6f %.6f %.6f\n", __LINE__, err, fsrc[NO].x,fsrc[NO].y, fsrc[NO].z);
     
     cu_tangent_flow_map<<<dimGrid, dimBlock>>>(sst, sigma, ftfm, width, height);
     err = cudaGetLastError();
	 cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;

     cudaMemcpy(fsrc, ftfm, sizeof(float3)*width*height, cudaMemcpyDeviceToHost);
     err = cudaGetLastError();
	 cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;

     printf("%.6f %.6f %.6f\n", fsrc[NO].x,fsrc[NO].y, fsrc[NO].z);
     // rgb
     cu_rgb2lab<<<dimGrid, dimBlock>>>(tmp,lab,width, height);
     err = cudaGetLastError();
	 cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;

     cudaMemcpy(fsrc, lab, sizeof(float3)*width*height, cudaMemcpyDeviceToHost);
     err = cudaGetLastError();
	 cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;
     printf("%.6f %.6f %.6f\n", fsrc[NO].x,fsrc[NO].y, fsrc[NO].z);

     cu_orientation_aligned_bilateral_filter<<<dimGrid, dimBlock>>>(lab, ftfm, midlab, tmp, 4, 3.0f, 0.0425f, width, height);
     err = cudaGetLastError();
	 cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;

     cudaMemcpy(fsrc, midlab, sizeof(float3)*width*height, cudaMemcpyDeviceToHost);
     err = cudaGetLastError();
	 cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;
     printf(" %.6f %.6f %.6f\n", fsrc[NO].x,fsrc[NO].y, fsrc[NO].z);

     cu_lab2rgb<<<dimGrid, dimBlock>>>(midlab,rgb,width, height);
     err = cudaGetLastError();
     if(err)
          cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;

     cudaMemcpy(fsrc, rgb, sizeof(float3)*width*height, cudaMemcpyDeviceToHost);
     //err = cudaGetLastError();
	 //cout << "LINE = " << __LINE__ << ", err No = " << err << ", " << cudaGetErrorString(err) << endl;
     printf(" %.6f %.6f %.6f\n",  fsrc[NO].x,fsrc[NO].y, fsrc[NO].z);
    
     // clean up
     cudaMemcpy(fsrc, rgb, sizeof(float3)*width*height, cudaMemcpyDeviceToHost);
     cleanup(fsrc, dest,width*height);
     printf("%.6f %.6f %.6f\n", fsrc[512].x,fsrc[512].y, fsrc[512].z);
     // release
FREE:     
     cudaSafeFree((void**)&st);
     cudaSafeFree((void**)&lab);
     cudaSafeFree((void**)&rgb);
     cudaSafeFree((void**)&midlab);
     cudaSafeFree((void**)&tmp);
     cudaSafeFree((void**)&sst);
     cudaSafeFree((void**)&ftfm);
     cudaSafeFree((void**)&kernel_d);
     delete[] fsrc;
     delete[] kernel;

	 gettimeofday (&ts , &tz); 
	 printf("sec; %ld\n", ts.tv_sec - sec); 
	 printf("usec; %ld\n",ts.tv_usec - usec); 
}
