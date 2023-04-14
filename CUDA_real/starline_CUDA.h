#pragma once
#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

#define COLOR_TYPE uchar4
#define MAKE_COLOR(b,g,r,a) make_uchar4(b,g,r,a)
#define PARAM_TYPE float
#define COORD_TYPE float3
#define MAKE_COORD(x,y,z) make_float3(x,y,z)
#define PIXEL_TYPE short2
#define MAKE_PIXEL(x,y) make_short2(x,y)

#define MAT_3_3(mat_param,i,j) (*(mat_param+(3*(i))+(j)))
#define COL_3_1(col_param,i) (*(col_param+(i)))

#define pixel_Nx 640
#define pixel_Ny 480
#define point_splatting_Nx 320
#define point_splatting_Ny 240

#define src_pixel_color_Nx 1280
#define src_pixel_color_Ny 720
#define src_pixel_depth_Nx 1280
#define src_pixel_depth_Ny 720
#define crop_src_pixel_color_Nx 320//
#define crop_src_pixel_color_Ny 120//
#define crop_src_pixel_depth_Nx 320//
#define crop_src_pixel_depth_Ny 120//


#define up_left_corner_x -2.0f
#define up_left_corner_y -1.5f
#define up_left_corner_z 3.788f
#define horizontal_x 4.0f
#define horizontal_y 0.0f
#define horizontal_z 0.0f
#define vertical_x 0.0f
#define vertical_y 3.0f
#define vertical_z 0.0f

#define prec 0.01f

////depth view
//__constant__
//PARAM_TYPE depth_view_internal_param[9]
//= { 504.271545,   0,   320,
//0,   504.474945,   240,
//0,   0,   1 };
//__constant__
//PARAM_TYPE depth_R_view_external_param[9]
//= { 0.99950026,   0.029935994, -0.010152577,
//-0.028774172,   0.99458327,   0.09988073,
//0.013087612, -0.099538683,   0.99494762 };
//__constant__
//PARAM_TYPE depth_T_view_external_param[3]
//= { -0.06637426e+02, -0.25744997e+02,   1.0324199e+02 };
//__constant__
//PARAM_TYPE depth_R_view_external_param_inverse[9]
//= { 0.99950025, -0.02877417,  0.01308761,
//0.02993599,  0.99458328, -0.09953868,
//-0.01015258,  0.09988073,  0.99494762 };
//__constant__
//PARAM_TYPE depth_T_view_external_param_inverse[3]
//= { 0.06637426e+02, 0.25744997e+02,   -1.0324199e+02 };
////depth left
//__constant__
//PARAM_TYPE depth_left_internal_param[9]
//= { 504.805695,   0,   320,
//0,   504.943237,   240,
//0,   0,   1 };
//__constant__
//PARAM_TYPE depth_R_left_external_param[9]
//= { 0.88141694,   0.012041774, -0.47218552,
//-0.11227235,   0.97636546, -0.18467651,
//0.4588018,   0.21579038,   0.86193702 };
//__constant__
//PARAM_TYPE depth_T_left_external_param[3]
//= { -0.080758244e+02, -0.16990661e+02,   0.982846e+02 };
//__constant__
//PARAM_TYPE depth_R_left_external_param_inverse[9]
//= { 0.88141695, -0.11227234,  0.4588018,
//0.01204178,  0.97636546,  0.21579038,
//-0.47218552, -0.18467651,  0.86193701 };
//__constant__
//PARAM_TYPE depth_T_left_external_param_inverse[3]
//= { 0.080758244e+02, 0.16990661e+02,   -0.982846e+02 };
////depth mid
//__constant__
//PARAM_TYPE depth_mid_internal_param[9]
//= { 504.487091,   0,   320,
//0,   504.636749,   240,
//0,   0,   1 };
//__constant__
//PARAM_TYPE depth_R_mid_external_param[9]
//= { 0.99239665,   0.054741063,   0.11023748,
//-0.069761989,   0.98805903,   0.13737763,
//-0.10140094, -0.14402348,   0.98436532 };
//__constant__
//PARAM_TYPE depth_T_mid_external_param[3]
//= { -0.08105746e+02, -0.23327866e+02,   1.0072768e+02 };
//__constant__
//PARAM_TYPE depth_R_mid_external_param_inverse[9]
//= { 0.99239665, -0.06976199, -0.10140094,
//0.05474106,  0.98805904, -0.14402349,
//0.11023748,  0.13737762,  0.98436532 };
//__constant__
//PARAM_TYPE depth_T_mid_external_param_inverse[3]
//= { 0.08105746e+02, 0.23327866e+02,   -1.0072768e+02 };
////depth right
//__constant__
//PARAM_TYPE depth_right_internal_param[9]
//= { 505.464600,   0,   320,
//0,   505.600983,   240,
//0,   0,   1 };
//__constant__
//PARAM_TYPE depth_R_right_external_param[9]
//= { 0.93960733,   0.019679058,   0.34168816,
//0.012269075,   0.99576722, -0.091088535,
//-0.34203441,   0.089779653,   0.93538873 };
//__constant__
//PARAM_TYPE depth_T_right_external_param[3]
//= { -0.1565215e+02, -0.18469312e+02,   1.0924272e+02 };
//__constant__
//PARAM_TYPE depth_R_right_external_param_inverse[9]
//= { 0.93960733,  0.01226908, -0.3420344,
//0.01967906,  0.99576721,  0.08977965,
//0.34168817, -0.09108853,  0.93538873 };
//__constant__
//PARAM_TYPE depth_T_right_external_param_inverse[3]
//= { 0.1565215e+02, 0.18469312e+02,   -1.0924272e+02 };
//
////color view
//__constant__
//PARAM_TYPE color_view_internal_param[9]
//= { 607.447937,   0,   320,
//0,   607.522095,   240,
//0,   0,   1 };
//__constant__
//PARAM_TYPE color_R_view_external_param[9]
//= { 0.99591103,   0.035190824,   0.08320356,
//-0.036246951,   0.99927991,   0.011216525,
//-0.082748928, -0.014186537,   0.99646945 };
//__constant__
//PARAM_TYPE color_T_view_external_param[3]
//= { -0.10772186e+02, -0.15827784e+02,   0.97874419e+02 };
//__constant__
//PARAM_TYPE color_R_view_external_param_inverse[9]
//= { 0.99591102, -0.03624695, -0.08274893,
//0.03519082,  0.99927992, -0.01418654,
//0.08320356,  0.01121653,  0.99646944 };
//__constant__
//PARAM_TYPE color_T_view_external_param_inverse[3]
//= { 0.10772186e+02, 0.15827784e+02,   -0.97874419e+02 };
////color left
//__constant__
//PARAM_TYPE color_left_internal_param[9]
//= { 605.924133,   0,   320,
//0,   605.991394,   240,
//0,   0,   1 };
//__constant__
//PARAM_TYPE color_R_left_external_param[9]
//= { 0.89840538,   0.021280963, -0.43865122,
//-0.061524214,   0.99507407, -0.077732715,
//0.43483623,   0.096823161,   0.89528919 };
//__constant__
//PARAM_TYPE color_T_left_external_param[3]
//= { -0.12094466e+02, -0.076613206e+02,   0.97991322e+02 };
//__constant__
//PARAM_TYPE color_R_left_external_param_inverse[9]
//= { 0.89840538, -0.06152421,  0.43483622,
//0.02128096,  0.99507406,  0.09682316,
//-0.43865122, -0.07773271,  0.89528919 };
//__constant__
//PARAM_TYPE color_T_left_external_param_inverse[3]
//= { 0.12094466e+02, 0.076613206e+02,   -0.97991322e+02 };
////color mid
//__constant__
//PARAM_TYPE color_mid_internal_param[9]
//= { 603.888306,   0,   320,
//0,   603.703613,   240,
//0,   0,   1 };
//__constant__
//PARAM_TYPE color_R_mid_external_param[9]
//= { 0.96836924,   0.069108717,   0.2397603,
//-0.087117218,   0.99405352,   0.065331375,
//-0.2338196, -0.084152145,   0.96863141 };
//__constant__
//PARAM_TYPE color_T_mid_external_param[3]
//= { -0.11676304e+02, -0.14276823e+02,   0.95677673e+02 };
//__constant__
//PARAM_TYPE color_R_mid_external_param_inverse[9]
//= { 0.96836924, -0.08711722, -0.2338196,
//0.06910872,  0.99405352, -0.08415214,
//0.2397603,   0.06533138,  0.96863141 };
//__constant__
//PARAM_TYPE color_T_mid_external_param_inverse[3]
//= { 0.11676304e+02, 0.14276823e+02,   -0.95677673e+02 };
////color right
//__constant__
//PARAM_TYPE color_right_internal_param[9]
//= { 605.486389,   0,   320,
//0,   605.411194,   240,
//0,   0,   1 };
//__constant__
//PARAM_TYPE color_R_right_external_param[9]
//= { 0.92267762,   0.037298515,   0.38376403,
//-0.013167598,   0.99777771, -0.065316643,
//-0.38534741,   0.055212954,   0.92111829 };
//__constant__
//PARAM_TYPE color_T_right_external_param[3]
//= { -0.184855e+02, -0.0778905e+02,   1.0641658e+02 };
//__constant__
//PARAM_TYPE color_R_right_external_param_inverse[9]
//= { 0.92267762, - 0.0131676, - 0.38534741,
// 0.03729851,  0.9977777,   0.05521295,
//0.38376403, - 0.06531664,  0.92111829 };
//__constant__
//PARAM_TYPE color_T_right_external_param_inverse[3]
//= { 0.184855e+02, 0.0778905e+02,   -1.0641658e+02 };



__constant__
PARAM_TYPE view_internal_param[9]
= { 607.447937,   0,   320,
0,   607.522095,   240,
0,   0,   1 };
__constant__
PARAM_TYPE R_view_external_param[9]
= { 0.97282915,   0.072511701, -0.21987611,
-0.035734491,   0.98533569,   0.16684311,
0.22874985, -0.15445268,   0.96115445 };
__constant__
PARAM_TYPE R_view_external_param_inverse[9]
= { 0.97282915, -0.03573449,  0.22874985,
    0.0725117,   0.98533569, -0.15445268,
    -0.21987611,  0.16684311,  0.96115445 };
__constant__
PARAM_TYPE T_view_external_param[3]
= { -0.063299724e+02, -0.049407028e+02, 0.76696249e+02 };
__constant__
PARAM_TYPE T_view_external_param_inverse[3]
= { 0.063299724e+02, 0.049407028e+02, -0.76696249e+02 };




__constant__
PARAM_TYPE left_internal_param[9]
= { 606.134,   0,   320,
0,   605.951,   240,
0,   0,   1 };
__constant__
PARAM_TYPE R_left_external_param[9]
= { 0.87402824,   0.04817779, -0.48348065,
-0.032189193,   0.99862732,   0.041319944,
0.48480769, -0.020551946,   0.87437928 };
__constant__
PARAM_TYPE R_left_external_param_inverse[9]
= { 0.87402824, -0.03218919,  0.48480769,
   0.04817779,  0.99862732, -0.02055195,
    -0.48348065,  0.04131994,  0.87437928 };
__constant__
PARAM_TYPE T_left_external_param[3]
= { 0.025058259e+02, -0.035155708e+02,   0.79021269e+02 };
__constant__
PARAM_TYPE T_left_external_param_inverse[3]
= { -0.025058259e+02, 0.035155708e+02,   -0.79021269e+02 };



__constant__
PARAM_TYPE mid_internal_param[9]
= { 605.924133,   0,   320,
0,   605.991394,   240,
0,   0,   1 };
__constant__
PARAM_TYPE R_mid_external_param[9]
= { 0.92238833,   0.016037037,   0.38593081,
-0.06859051,   0.99005924,  0.12279266,
-0.38012513, -0.1397337, 0.91431908 };
__constant__
PARAM_TYPE R_mid_external_param_inverse[9]
= { 0.92238833, -0.06859051, -0.38012513,
    0.01603704,  0.99005924, -0.1397337,
    0.38593081,  0.12279266,  0.91431908 };
__constant__
PARAM_TYPE T_mid_external_param[3]
= { -0.044661758e+02, -0.080205671e+02,   0.85012851e+02 };
__constant__
PARAM_TYPE T_mid_external_param_inverse[3]
= { 0.044661758e+02, 0.080205671e+02,   -0.85012851e+02 };


__constant__
PARAM_TYPE right_internal_param[9]
= { 603.888306,   0,   320,
0,   603.703613,   240,
0,   0,   1 };
__constant__
PARAM_TYPE R_right_external_param[9]
= { 0.79995099,   0.021275003,   0.59968808,
-0.021173286,   0.99974972, -0.0072239019,
-0.59969168, -0.0069185994,   0.80020124 };
__constant__
PARAM_TYPE R_right_external_param_inverse[9]
= { 0.79995099, -0.02117329, -0.59969168,
    0.021275,    0.99974972, -0.0069186,
    0.59968808, -0.0072239,   0.80020124 };
__constant__
PARAM_TYPE T_right_external_param[3]
= { -0.066983473e+02, -0.079096781e+02,   0.95016361e+02 };
__constant__
PARAM_TYPE T_right_external_param_inverse[3]
= { 0.066983473e+02, 0.079096781e+02,   -0.95016361e+02 };

__device__
class ray {
public:
	COORD_TYPE origin, dir, ray_pos;
	__device__ ray(const float u, const float v);
	__device__ void point_at_param(const float t);
};

__device__
float my_clamp(float data, float _t, float t);

__device__
COORD_TYPE SrcCampos_DstCampos(
    const COORD_TYPE* SrcCampos,
    const PARAM_TYPE* R_src_external_param,
    const PARAM_TYPE* T_src_external_param,
    const PARAM_TYPE* R_dst_external_param_inverse,
    const PARAM_TYPE* T_dst_external_param_inverse);

__device__
PIXEL_TYPE Campos_Pix(
    const COORD_TYPE* SrcCampos,
    const PARAM_TYPE* internal_param);

__device__
COORD_TYPE Pix_Campos(
    const float depth,
    const int pixel_x,
    const int pixel_y,
    const PARAM_TYPE* internal_param);

class CudaResource {
public:
    const int src_color_N = src_pixel_color_Nx * src_pixel_color_Ny;
    const int src_depth_N = src_pixel_depth_Nx * src_pixel_depth_Ny;
    const int N = pixel_Nx * pixel_Ny;
    const int point_splatting_N = point_splatting_Nx * point_splatting_Ny;
    const int Border_W = src_pixel_depth_Nx + 2;
    const int Border_H = src_pixel_depth_Ny + 2;
    const int Border_N = Border_W * Border_H;

    cv::Mat h_color_img_left;
    cv::Mat h_color_img_mid;
    cv::Mat h_color_img_right;

    cv::Mat h_depth_img_left;
    cv::Mat h_depth_img_mid;
    cv::Mat h_depth_img_right;
    cv::Mat h_depth_img_left_border;
    cv::Mat h_depth_img_mid_border;
    cv::Mat h_depth_img_right_border;

    COLOR_TYPE* p_h_color_img_left;
    COLOR_TYPE* p_h_color_img_mid;
    COLOR_TYPE* p_h_color_img_right;

    cv::uint16_t* p_h_depth_img_left;
    cv::uint16_t* p_h_depth_img_mid;
    cv::uint16_t* p_h_depth_img_right;
    cv::uint16_t* p_h_depth_img_left_border;
    cv::uint16_t* p_h_depth_img_mid_border;
    cv::uint16_t* p_h_depth_img_right_border;

    cudaChannelFormatDesc color_tex_channel;
    
    cudaArray* cuA_color_img_left;
    struct cudaResourceDesc color_tex_left_resDesc;
    struct cudaTextureDesc color_tex_left_texDesc;
    

    cudaArray* cuA_color_img_mid;
    struct cudaResourceDesc color_tex_mid_resDesc;
    struct cudaTextureDesc color_tex_mid_texDesc;
    

    cudaArray* cuA_color_img_right;                                              
    struct cudaResourceDesc color_tex_right_resDesc;
    struct cudaTextureDesc color_tex_right_texDesc;
    

    cudaChannelFormatDesc depth_tex_channel;

    cudaArray* cuA_depth_img_left;
    struct cudaResourceDesc depth_tex_left_resDesc;
    struct cudaTextureDesc depth_tex_left_texDesc;
    

    cudaArray* cuA_depth_img_mid;
    struct cudaResourceDesc depth_tex_mid_resDesc;
    struct cudaTextureDesc depth_tex_mid_texDesc;
    

    cudaArray* cuA_depth_img_right;
    struct cudaResourceDesc depth_tex_right_resDesc;
    struct cudaTextureDesc depth_tex_right_texDesc;
    

    cudaArray* cuA_depth_img_left_border;
    struct cudaResourceDesc depth_tex_left_border_resDesc;
    struct cudaTextureDesc depth_tex_left_border_texDesc;
    

    cudaArray* cuA_depth_img_mid_border;
    struct cudaResourceDesc depth_tex_mid_border_resDesc;
    struct cudaTextureDesc depth_tex_mid_border_texDesc;
    

    cudaArray* cuA_depth_img_right_border;
    struct cudaResourceDesc depth_tex_right_border_resDesc;
    struct cudaTextureDesc depth_tex_right_border_texDesc;
    


    cudaChannelFormatDesc depth_tex_rasterize_channel;
    
    struct cudaResourceDesc depth_tex_rasterize_min_resDesc;
    struct cudaTextureDesc depth_tex_rasterize_min_texDesc;
    

    struct cudaResourceDesc depth_tex_rasterize_max_resDesc;
    struct cudaTextureDesc depth_tex_rasterize_max_texDesc;

    

    
public:
    CudaResource();
    ~CudaResource();

    void LoadImg();
    void UpdateTexture();
    void DestroyResource();

    float* d_depth_rasterize_min;
    float* d_depth_rasterize_max;


    cudaTextureObject_t color_tex_left_obj;
    cudaTextureObject_t color_tex_mid_obj;
    cudaTextureObject_t color_tex_right_obj;

    cudaTextureObject_t depth_tex_left_obj;
    cudaTextureObject_t depth_tex_mid_obj;
    cudaTextureObject_t depth_tex_right_obj;

    cudaTextureObject_t depth_tex_left_border_obj;
    cudaTextureObject_t depth_tex_mid_border_obj;
    cudaTextureObject_t depth_tex_right_border_obj;

    cudaTextureObject_t depth_tex_rasterize_min_obj;
    cudaTextureObject_t depth_tex_rasterize_max_obj;

};

__global__
void depth_img_point_splatting(
    float* depth_rasterize_min,
    float* depth_rasterize_max,
    cudaTextureObject_t depth_tex_left,
    cudaTextureObject_t depth_tex_mid,
    cudaTextureObject_t depth_tex_right);

__global__
void ray_cast(
    COORD_TYPE* crood,
    short4* color,
    cudaTextureObject_t color_tex_left,
    cudaTextureObject_t color_tex_mid,
    cudaTextureObject_t color_tex_right,
    cudaTextureObject_t depth_tex_left,
    cudaTextureObject_t depth_tex_mid,
    cudaTextureObject_t depth_tex_right,
    cudaTextureObject_t depth_tex_left_border,
    cudaTextureObject_t depth_tex_mid_border,
    cudaTextureObject_t depth_tex_right_border,
    cudaTextureObject_t depth_tex_rasterize_min,
    cudaTextureObject_t depth_tex_rasterize_max);





