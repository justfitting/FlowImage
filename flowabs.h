//flowabs.h

#ifndef FLOWABS_H
#define FLOWABS_H

#include "vector.h"

void rgb2lab( vec3* src, vec3* dest, int width, int height );
void lab2rgb( vec3* src, vec3* dest, int width, int height );
void gauss_filter( vec3* src, double sigma, vec3* dest, int width, int height );
void tangent_flow_map( vec3* src, double sigma, vec3* tfm, int width, int height );
void orientation_aligned_bilateral_filter( vec3*src, vec3* tfm, vec3*dest, int n, double sigma_d, double sigma_r, int width, int height );
void overlay( vec3* edges, vec3* image, vec3* dest, int width, int height );
void fdog_filter( vec3* src, vec3* tfm, int n, double sigma_e, double sigma_r, double tau, double sigma_m, double phi, vec3* dest, int width, int height );
void dog_filter( vec3* src,  int n, double sigma_e, double sigma_r, double tau, double phi, vec3*dest, int width, int height);
void color_quantization( vec3* img, int nbins, double phi_q, int filter, vec3* dest, int wdith, int height );
void make_noises( double* noise ,int width, int height );
void lic_filter( vec3* tfm, vec3* img, double sigma, vec3* dest, int width, int height );
void mix_filter( vec3* edges, vec3* img, float edge_color[3], vec3* dest, int width, int height );
void smooth_filter( vec3* tfm, vec3* img, int type, double sigma, vec3* dest, int width, int height );

#endif
