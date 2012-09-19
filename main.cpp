#include<sys/time.h>
#include "stdio.h"
#include "cv2vector.h"
#include "flowabs.h"
//#include "marching_squares.h"
#include "highgui.h"

extern void cu_flowabs(vec3* src, vec3* dest, float sigma, vec3* tfm, int width, int height);

void print(vec3* fsrc, int width, int height){
     if(fsrc){
          for(int i = 0; i < height; ++i)
          {
               for(int j = 0; j < width; ++j)
               {
                    fprintf(stdout, "(%.2f %.2f %.2f) ", fsrc[i*width+j].r, fsrc[i*width+j].g, fsrc[i*width+j].b);
               }
               fprintf(stdout, "\n");
          }
     }
}

int main( int argc, char** args)
{
    char filename[1000];
    if(argc <= 1)
         strcpy(filename,"liushishi.bmp");
    else
         strcpy(filename,args[1]);
    fprintf(stdout, "Processing %s...\n", filename);
	char outname[1000];
	IplImage* in = cvLoadImage( filename, CV_LOAD_IMAGE_COLOR );
	if ( in == NULL )
	{
		printf("file cannot be opened!!!");
		return 0;
	}
	cvNamedWindow( "original" );
	cvShowImage( "original", in );
    cvWaitKey( 0 );
	IplImage* fcimg = cvCreateImage( cvGetSize( in ), IPL_DEPTH_64F, 3 );
	IplImage* fgimg = cvCreateImage( cvGetSize( in ), IPL_DEPTH_64F, 1 );
	IplImage* ucimg = cvCreateImage( cvGetSize( in ), IPL_DEPTH_8U, 3 );
	int w = in->width;
	int h = in->height;

	vec3* frgb = new vec3[ w * h ];
	ucImg2frgb( in, frgb );

    //fprintf(stdout, "noise\n");

	double* noise = new double[ w * h ];
	make_noises( noise, w, h );
	vec3* vnoise = new vec3[ w * h ];
	for ( int y = 0; y < h; y++)
	{
		for ( int x = 0; x < w; x++ )
		{
			vnoise[ y * w + x ].r = noise[ y * w + x ];
			vnoise[ y * w + x ].g = noise[ y * w + x ];
			vnoise[ y * w + x ].b = noise[ y * w + x ];
		}
	}

	// frgb2fcImg( vnoise, fcimg );
	// cvNamedWindow( "noise" );
	// cvShowImage( "noise", fcimg );

	vec3* tfm = new vec3[ w * h];
    vec3* flow = new vec3[ w * h ];
	vec3* smflow = new vec3[ w * h ];

    fprintf(stdout, "flow\n");
    
    // GPU Show
    cu_flowabs(frgb, flow, 2.0f, tfm, w, h);
    //print(flow, w, h);          // 
    frgb2ucImg(flow, ucimg );
	cvNamedWindow( "GPU_FLOWAB" );
	cvShowImage( "GPU_FLOWAB", ucimg );
    cvWaitKey( 0 );
    strcpy(outname, filename);
    strcat( outname, "_gpu.jpg");
    printf("%s is Saved\n", outname);
    cvSaveImage(outname, ucimg);

    struct timeval ts;
    struct timezone tz;
    gettimeofday (&ts , &tz);
    unsigned long sec = ts.tv_sec;
    unsigned long usec = ts.tv_usec;

    // CPU Show
    
	tangent_flow_map( frgb, 2.0f, tfm, w, h );
	//lic_filter( tfm, vnoise, 5.0, flow, w, h );
	////gauss_filter( flow, 0.5, smflow, w, h );
	// frgb2fcImg( tfm, fcimg );
	// cvNamedWindow( "flow" );
	// cvShowImage( "flow" , fcimg );
	// cvWaitKey( 0 );

	vec3* lab = new vec3[ w * h ];
	vec3* midlab = new vec3[ w * h ];
	vec3* rgb = new vec3[ w * h ];

	rgb2lab( frgb, lab, w, h );
	//frgb2fcImg( lab, fcimg );
	//cvNamedWindow( "rgb2lab" );
	//cvShowImage( "rgb2lab", fcimg );
	//lic_filter( tfm, lab, 5.0, midlab, w, h );
    fprintf(stdout, "lic\n");

	orientation_aligned_bilateral_filter( lab, tfm, midlab, 4, 3.0, 0.0425, w, h );
	lab2rgb( midlab, rgb, w, h );
	frgb2fcImg( rgb, fcimg );
	cvNamedWindow( "orientation_aligned_bilateral_filter" );
	cvShowImage( "orientation_aligned_bilateral_filter", fcimg );

    gettimeofday (&ts , &tz); 
    printf("sec; %ld\n", ts.tv_sec - sec); 
    printf("usec; %ld\n",ts.tv_usec - usec); 

    
    // cvWaitKey(0);
	color_quantization( midlab, 8, 3.4, 0, lab, w, h );
    // frgb2ucImg( lab, ucimg );
	// cvNamedWindow( "lab_color_quantization" );
	// cvShowImage( "lab_color_quantization", ucimg );
    // cvWaitKey(0);
	
	lab2rgb( midlab, rgb, w, h );
	frgb2ucImg( rgb, ucimg );
	cvNamedWindow( "color_quantization" );
	cvShowImage( "color_quantization", ucimg );

    strcpy(outname, "color_quatization_");
	strcat( outname, filename );
	strcat( outname, ".bmp");
	cvSaveImage( outname, ucimg);
	// gauss_filter( rgb, 2.0, frgb, w, h );
	// frgb2fcImg( frgb, fcimg );
	// cvNamedWindow( "gauss" );
	// cvShowImage( "gauss" , fcimg );

	cvWaitKey( 0 );

	//frgb2uImg( rgb, in );
	//cvSaveImage("liushishi.png", in );


	return 0;
}



