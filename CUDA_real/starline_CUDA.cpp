#include "starline_CUDA.h"

__device__ 
ray::ray(const float u, const float v) :origin(MAKE_COORD(0.0f, 0.0f, 0.0f)), ray_pos(MAKE_COORD(0.0f, 0.0f, 0.0f)) {
    dir.x = up_left_corner_x + u * horizontal_x + v * vertical_x;
    dir.y = up_left_corner_y + u * horizontal_y + v * vertical_y;
    dir.z = up_left_corner_z + u * horizontal_z + v * vertical_z;
    //标准化
    //float temp = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
    //temp = sqrt(temp);
    //dir.x /= temp;
    //dir.y /= temp;
    //dir.z /= temp;
}

__device__ 
void ray::point_at_param(const float t) {
    ray_pos.x = origin.x + t * dir.x;
    ray_pos.y = origin.y + t * dir.y;
    ray_pos.z = origin.z + t * dir.z;
}

__device__
float my_clamp(float data, float _t, float t) {
    return fmaxf(_t, fminf(t, data));
}

__device__
COORD_TYPE SrcCampos_DstCampos(
    const COORD_TYPE* SrcCampos,
    const PARAM_TYPE* R_src_external_param,
    const PARAM_TYPE* T_src_external_param,
    const PARAM_TYPE* R_dst_external_param_inverse,
    const PARAM_TYPE* T_dst_external_param_inverse) {
    //坐标单位：cm

    ///*      转换到世界坐标     */
    COORD_TYPE world_ray_pos;
    COORD_TYPE temp_world_ray_pos;
    temp_world_ray_pos.x = (SrcCampos->x) - COL_3_1(T_src_external_param, 0);
    temp_world_ray_pos.y = (SrcCampos->y) - COL_3_1(T_src_external_param, 1);
    temp_world_ray_pos.z = (SrcCampos->z) - COL_3_1(T_src_external_param, 2);
    world_ray_pos.x
        = (temp_world_ray_pos.x) * MAT_3_3(R_src_external_param, 0, 0)
        + (temp_world_ray_pos.y) * MAT_3_3(R_src_external_param, 1, 0)
        + (temp_world_ray_pos.z) * MAT_3_3(R_src_external_param, 2, 0);
        

    world_ray_pos.y
        = (temp_world_ray_pos.x) * MAT_3_3(R_src_external_param, 0, 1)
        + (temp_world_ray_pos.y) * MAT_3_3(R_src_external_param, 1, 1)
        + (temp_world_ray_pos.z) * MAT_3_3(R_src_external_param, 2, 1);
        

    world_ray_pos.z
        = (temp_world_ray_pos.x) * MAT_3_3(R_src_external_param, 0, 2)
        + (temp_world_ray_pos.y) * MAT_3_3(R_src_external_param, 1, 2)
        + (temp_world_ray_pos.z) * MAT_3_3(R_src_external_param, 2, 2);
        

    /*      转换到目标相机坐标       */
    COORD_TYPE dst_camera_pos;

    dst_camera_pos.x
        = world_ray_pos.x * MAT_3_3(R_dst_external_param_inverse, 0, 0)
        + world_ray_pos.y * MAT_3_3(R_dst_external_param_inverse, 1, 0)
        + world_ray_pos.z * MAT_3_3(R_dst_external_param_inverse, 2, 0)
        - COL_3_1(T_dst_external_param_inverse, 0);
    dst_camera_pos.y
        = world_ray_pos.x * MAT_3_3(R_dst_external_param_inverse, 0, 1)
        + world_ray_pos.y * MAT_3_3(R_dst_external_param_inverse, 1, 1)
        + world_ray_pos.z * MAT_3_3(R_dst_external_param_inverse, 2, 1)
        - COL_3_1(T_dst_external_param_inverse, 1);
    dst_camera_pos.z
        = world_ray_pos.x * MAT_3_3(R_dst_external_param_inverse, 0, 2)
        + world_ray_pos.y * MAT_3_3(R_dst_external_param_inverse, 1, 2)
        + world_ray_pos.z * MAT_3_3(R_dst_external_param_inverse, 2, 2)
        - COL_3_1(T_dst_external_param_inverse, 2);



    return dst_camera_pos;

}

__device__
PIXEL_TYPE Campos_Pix(
    const COORD_TYPE* SrcCampos,
    const PARAM_TYPE* internal_param) {
    /*      转换到目标像素坐标       */
    //计算缩放
    COORD_TYPE temp_dst_camera_pos;
    temp_dst_camera_pos.x = SrcCampos->x / abs(SrcCampos->z);
    temp_dst_camera_pos.y = SrcCampos->y / abs(SrcCampos->z);
    temp_dst_camera_pos.z = 1.0f;
    COORD_TYPE dst_pixel_pos;
    dst_pixel_pos.x
        = temp_dst_camera_pos.x * MAT_3_3(internal_param, 0, 0)
        + temp_dst_camera_pos.y * MAT_3_3(internal_param, 0, 1)
        + temp_dst_camera_pos.z * MAT_3_3(internal_param, 0, 2);
    dst_pixel_pos.y
        = temp_dst_camera_pos.x * MAT_3_3(internal_param, 1, 0)
        + temp_dst_camera_pos.y * MAT_3_3(internal_param, 1, 1)
        + temp_dst_camera_pos.z * MAT_3_3(internal_param, 1, 2);
    dst_pixel_pos.z
        = temp_dst_camera_pos.z;
    PIXEL_TYPE dst_pixel;

    
    dst_pixel.x = short(dst_pixel_pos.x + 0.5f);
    dst_pixel.y = short(dst_pixel_pos.y + 0.5f);


    return dst_pixel;
}

__device__
COORD_TYPE Pix_Campos(
    const float depth,
    const int pixel_x,
    const int pixel_y,
    const PARAM_TYPE* internal_param) {

    return MAKE_COORD(
        ((pixel_x - MAT_3_3(internal_param, 0, 2)) / MAT_3_3(internal_param, 0, 0)) * abs(depth),
        ((pixel_y - MAT_3_3(internal_param, 1, 2)) / MAT_3_3(internal_param, 1, 1)) * abs(depth),
        depth);
}


CudaResource::CudaResource() {
    /* 彩色 */
    color_tex_channel = cudaCreateChannelDesc<COLOR_TYPE>();

    cudaMallocArray(&cuA_color_img_left, &color_tex_channel, src_pixel_color_Nx, src_pixel_color_Ny);
    cudaMallocArray(&cuA_color_img_mid, &color_tex_channel, src_pixel_color_Nx, src_pixel_color_Ny);
    cudaMallocArray(&cuA_color_img_right, &color_tex_channel, src_pixel_color_Nx, src_pixel_color_Ny);
    
    memset(&color_tex_left_resDesc, 0, sizeof(color_tex_left_resDesc));
    color_tex_left_resDesc.resType = cudaResourceTypeArray;
    color_tex_left_resDesc.res.array.array = cuA_color_img_left;
    memset(&color_tex_left_texDesc, 0, sizeof(color_tex_left_texDesc));
    color_tex_left_texDesc.addressMode[0] = cudaAddressModeWrap;
    color_tex_left_texDesc.addressMode[1] = cudaAddressModeWrap;//寻址方式 如果是三维数组则设置texRef.addressMode[2]
    color_tex_left_texDesc.normalizedCoords = false;
    color_tex_left_texDesc.filterMode = cudaFilterModePoint;
    color_tex_left_texDesc.readMode = cudaReadModeElementType;

    memset(&color_tex_mid_resDesc, 0, sizeof(color_tex_mid_resDesc));
    color_tex_mid_resDesc.resType = cudaResourceTypeArray;
    color_tex_mid_resDesc.res.array.array = cuA_color_img_mid;
    memset(&color_tex_mid_texDesc, 0, sizeof(color_tex_mid_texDesc));
    color_tex_mid_texDesc.addressMode[0] = cudaAddressModeWrap;
    color_tex_mid_texDesc.addressMode[1] = cudaAddressModeWrap;//寻址方式 如果是三维数组则设置texRef.addressMode[2]
    color_tex_mid_texDesc.normalizedCoords = false;
    color_tex_mid_texDesc.filterMode = cudaFilterModePoint;
    color_tex_mid_texDesc.readMode = cudaReadModeElementType;

    memset(&color_tex_right_resDesc, 0, sizeof(color_tex_right_resDesc));
    color_tex_right_resDesc.resType = cudaResourceTypeArray;
    color_tex_right_resDesc.res.array.array = cuA_color_img_right;
    memset(&color_tex_right_texDesc, 0, sizeof(color_tex_right_texDesc));
    color_tex_right_texDesc.addressMode[0] = cudaAddressModeWrap;
    color_tex_right_texDesc.addressMode[1] = cudaAddressModeWrap;//寻址方式 如果是三维数组则设置texRef.addressMode[2]
    color_tex_right_texDesc.normalizedCoords = false;
    color_tex_right_texDesc.filterMode = cudaFilterModePoint;
    color_tex_right_texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&color_tex_left_obj, &color_tex_left_resDesc, &color_tex_left_texDesc, NULL);
    cudaCreateTextureObject(&color_tex_mid_obj, &color_tex_mid_resDesc, &color_tex_mid_texDesc, NULL);
    cudaCreateTextureObject(&color_tex_right_obj, &color_tex_right_resDesc, &color_tex_right_texDesc, NULL);


    /* 深度 */
    depth_tex_channel = cudaCreateChannelDesc<cv::uint16_t>();
    
    cudaMallocArray(&cuA_depth_img_left, &depth_tex_channel, src_pixel_depth_Nx, src_pixel_depth_Ny);
    cudaMallocArray(&cuA_depth_img_mid, &depth_tex_channel, src_pixel_depth_Nx, src_pixel_depth_Ny);
    cudaMallocArray(&cuA_depth_img_right, &depth_tex_channel, src_pixel_depth_Nx, src_pixel_depth_Ny);
    cudaMallocArray(&cuA_depth_img_left_border, &depth_tex_channel, Border_W, Border_H);
    cudaMallocArray(&cuA_depth_img_mid_border, &depth_tex_channel, Border_W, Border_H);
    cudaMallocArray(&cuA_depth_img_right_border, &depth_tex_channel, Border_W, Border_H);

    memset(&depth_tex_left_resDesc, 0, sizeof(depth_tex_left_resDesc));
    depth_tex_left_resDesc.resType = cudaResourceTypeArray;
    depth_tex_left_resDesc.res.array.array = cuA_depth_img_left;
    memset(&depth_tex_left_texDesc, 0, sizeof(depth_tex_left_texDesc));
    depth_tex_left_texDesc.addressMode[0] = cudaAddressModeWrap;
    depth_tex_left_texDesc.addressMode[1] = cudaAddressModeWrap;//寻址方式 如果是三维数组则设置texRef.addressMode[2]
    depth_tex_left_texDesc.normalizedCoords = false;
    depth_tex_left_texDesc.filterMode = cudaFilterModePoint;
    depth_tex_left_texDesc.readMode = cudaReadModeElementType;

    memset(&depth_tex_mid_resDesc, 0, sizeof(depth_tex_mid_resDesc));
    depth_tex_mid_resDesc.resType = cudaResourceTypeArray;
    depth_tex_mid_resDesc.res.array.array = cuA_depth_img_mid;
    memset(&depth_tex_mid_texDesc, 0, sizeof(depth_tex_mid_texDesc));
    depth_tex_mid_texDesc.addressMode[0] = cudaAddressModeWrap;
    depth_tex_mid_texDesc.addressMode[1] = cudaAddressModeWrap;//寻址方式 如果是三维数组则设置texRef.addressMode[2]
    depth_tex_mid_texDesc.normalizedCoords = false;
    depth_tex_mid_texDesc.filterMode = cudaFilterModePoint;
    depth_tex_mid_texDesc.readMode = cudaReadModeElementType;

    memset(&depth_tex_right_resDesc, 0, sizeof(depth_tex_right_resDesc));
    depth_tex_right_resDesc.resType = cudaResourceTypeArray;
    depth_tex_right_resDesc.res.array.array = cuA_depth_img_right;
    memset(&depth_tex_right_texDesc, 0, sizeof(depth_tex_right_texDesc));
    depth_tex_right_texDesc.addressMode[0] = cudaAddressModeWrap;
    depth_tex_right_texDesc.addressMode[1] = cudaAddressModeWrap;//寻址方式 如果是三维数组则设置texRef.addressMode[2]
    depth_tex_right_texDesc.normalizedCoords = false;
    depth_tex_right_texDesc.filterMode = cudaFilterModePoint;
    depth_tex_right_texDesc.readMode = cudaReadModeElementType;

    memset(&depth_tex_left_border_resDesc, 0, sizeof(depth_tex_left_border_resDesc));
    depth_tex_left_border_resDesc.resType = cudaResourceTypeArray;
    depth_tex_left_border_resDesc.res.array.array = cuA_depth_img_left_border;
    memset(&depth_tex_left_border_texDesc, 0, sizeof(depth_tex_left_border_texDesc));
    depth_tex_left_border_texDesc.addressMode[0] = cudaAddressModeWrap;
    depth_tex_left_border_texDesc.addressMode[1] = cudaAddressModeWrap;//寻址方式 如果是三维数组则设置texRef.addressMode[2]
    depth_tex_left_border_texDesc.normalizedCoords = false;
    depth_tex_left_border_texDesc.filterMode = cudaFilterModePoint;
    depth_tex_left_border_texDesc.readMode = cudaReadModeElementType;

    memset(&depth_tex_mid_border_resDesc, 0, sizeof(depth_tex_mid_border_resDesc));
    depth_tex_mid_border_resDesc.resType = cudaResourceTypeArray;
    depth_tex_mid_border_resDesc.res.array.array = cuA_depth_img_mid_border;
    memset(&depth_tex_mid_border_texDesc, 0, sizeof(depth_tex_mid_border_texDesc));
    depth_tex_mid_border_texDesc.addressMode[0] = cudaAddressModeWrap;
    depth_tex_mid_border_texDesc.addressMode[1] = cudaAddressModeWrap;//寻址方式 如果是三维数组则设置texRef.addressMode[2]
    depth_tex_mid_border_texDesc.normalizedCoords = false;
    depth_tex_mid_border_texDesc.filterMode = cudaFilterModePoint;
    depth_tex_mid_border_texDesc.readMode = cudaReadModeElementType;

    memset(&depth_tex_right_border_resDesc, 0, sizeof(depth_tex_right_border_resDesc));
    depth_tex_right_border_resDesc.resType = cudaResourceTypeArray;
    depth_tex_right_border_resDesc.res.array.array = cuA_depth_img_right_border;
    memset(&depth_tex_right_border_texDesc, 0, sizeof(depth_tex_right_border_texDesc));
    depth_tex_right_border_texDesc.addressMode[0] = cudaAddressModeWrap;
    depth_tex_right_border_texDesc.addressMode[1] = cudaAddressModeWrap;//寻址方式 如果是三维数组则设置texRef.addressMode[2]
    depth_tex_right_border_texDesc.normalizedCoords = false;
    depth_tex_right_border_texDesc.filterMode = cudaFilterModePoint;
    depth_tex_right_border_texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&depth_tex_left_obj, &depth_tex_left_resDesc, &depth_tex_left_texDesc, NULL);
    cudaCreateTextureObject(&depth_tex_mid_obj, &depth_tex_mid_resDesc, &depth_tex_mid_texDesc, NULL);
    cudaCreateTextureObject(&depth_tex_right_obj, &depth_tex_right_resDesc, &depth_tex_right_texDesc, NULL);
    cudaCreateTextureObject(&depth_tex_left_border_obj, &depth_tex_left_border_resDesc, &depth_tex_left_border_texDesc, NULL);
    cudaCreateTextureObject(&depth_tex_mid_border_obj, &depth_tex_mid_border_resDesc, &depth_tex_mid_border_texDesc, NULL);
    cudaCreateTextureObject(&depth_tex_right_border_obj, &depth_tex_right_border_resDesc, &depth_tex_right_border_texDesc, NULL);

    /* 点溅法 */
    /*      记录点溅法在主视角光栅化后的深度        */
    float* h_depth_rasterize_min = new float[point_splatting_N];
    float* h_depth_rasterize_max = new float[point_splatting_N];
    for (int i = 0; i < point_splatting_N; ++i) {
        h_depth_rasterize_min[i] = 300.0f;
        h_depth_rasterize_max[i] = -300.0f;
 
    }
    cudaMalloc((void**)&d_depth_rasterize_min, point_splatting_N * sizeof(float));
    cudaMalloc((void**)&d_depth_rasterize_max, point_splatting_N * sizeof(float));
    cudaMemcpy(d_depth_rasterize_min, h_depth_rasterize_min, point_splatting_N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_depth_rasterize_max, h_depth_rasterize_max, point_splatting_N * sizeof(float), cudaMemcpyHostToDevice);
    

    depth_tex_rasterize_channel = cudaCreateChannelDesc<float>();

    memset(&depth_tex_rasterize_min_resDesc, 0, sizeof(depth_tex_rasterize_min_resDesc));
    depth_tex_rasterize_min_resDesc.resType = cudaResourceTypeLinear;
    depth_tex_rasterize_min_resDesc.res.linear.devPtr = d_depth_rasterize_min;
    depth_tex_rasterize_min_resDesc.res.linear.desc = depth_tex_rasterize_channel;
    depth_tex_rasterize_min_resDesc.res.linear.sizeInBytes = point_splatting_N * sizeof(float);
    memset(&depth_tex_rasterize_min_texDesc, 0, sizeof(depth_tex_rasterize_min_texDesc));
    depth_tex_rasterize_min_texDesc.addressMode[0] = cudaAddressModeWrap;
    depth_tex_rasterize_min_texDesc.normalizedCoords = false;
    depth_tex_rasterize_min_texDesc.filterMode = cudaFilterModePoint;
    depth_tex_rasterize_min_texDesc.readMode = cudaReadModeElementType;

    memset(&depth_tex_rasterize_max_resDesc, 0, sizeof(depth_tex_rasterize_max_resDesc));
    depth_tex_rasterize_max_resDesc.resType = cudaResourceTypeLinear;
    depth_tex_rasterize_max_resDesc.res.linear.devPtr = d_depth_rasterize_max;
    depth_tex_rasterize_max_resDesc.res.linear.desc = depth_tex_rasterize_channel;
    depth_tex_rasterize_max_resDesc.res.linear.sizeInBytes = point_splatting_N * sizeof(float);
    memset(&depth_tex_rasterize_max_texDesc, 0, sizeof(depth_tex_rasterize_max_texDesc));
    depth_tex_rasterize_max_texDesc.addressMode[0] = cudaAddressModeWrap;
    depth_tex_rasterize_max_texDesc.normalizedCoords = false;
    depth_tex_rasterize_max_texDesc.filterMode = cudaFilterModePoint;
    depth_tex_rasterize_max_texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(
        &depth_tex_rasterize_min_obj,
        &depth_tex_rasterize_min_resDesc,
        &depth_tex_rasterize_min_texDesc,
        NULL);

    cudaCreateTextureObject(
        &depth_tex_rasterize_max_obj,
        &depth_tex_rasterize_max_resDesc,
        &depth_tex_rasterize_max_texDesc,
        NULL);

  ;

}

CudaResource::~CudaResource() {
    DestroyResource();
}

void CudaResource::LoadImg() {
    /*      读取彩色图       */
    h_color_img_left = cv::imread("E:\\CGT\\starline\\real_data_2\\left_color.png", cv::IMREAD_UNCHANGED);
    h_color_img_mid = cv::imread("E:\\CGT\\starline\\real_data_2\\mid_color.png", cv::IMREAD_UNCHANGED);
    h_color_img_right = cv::imread("E:\\CGT\\starline\\real_data_2\\right_color.png", cv::IMREAD_UNCHANGED);
    p_h_color_img_left = (COLOR_TYPE*)h_color_img_left.data;
    p_h_color_img_mid = (COLOR_TYPE*)h_color_img_mid.data;
    p_h_color_img_right = (COLOR_TYPE*)h_color_img_right.data;

    /*      读取深度图       */
    h_depth_img_left = cv::imread("E:\\CGT\\starline\\real_data_2\\left_depth.png", cv::IMREAD_ANYDEPTH);
    h_depth_img_mid = cv::imread("E:\\CGT\\starline\\real_data_2\\mid_depth.png", cv::IMREAD_ANYDEPTH);
    h_depth_img_right = cv::imread("E:\\CGT\\starline\\real_data_2\\right_depth.png", cv::IMREAD_ANYDEPTH);
    cv::copyMakeBorder(h_depth_img_left, h_depth_img_left_border, 1, 1, 1, 1, cv::BORDER_CONSTANT);
    cv::copyMakeBorder(h_depth_img_mid, h_depth_img_mid_border, 1, 1, 1, 1, cv::BORDER_CONSTANT);
    cv::copyMakeBorder(h_depth_img_right, h_depth_img_right_border, 1, 1, 1, 1, cv::BORDER_CONSTANT);
    p_h_depth_img_left = (cv::uint16_t*)h_depth_img_left.data;
    p_h_depth_img_mid = (cv::uint16_t*)h_depth_img_mid.data;
    p_h_depth_img_right = (cv::uint16_t*)h_depth_img_right.data;
    p_h_depth_img_left_border = (cv::uint16_t*)h_depth_img_left_border.data;
    p_h_depth_img_mid_border = (cv::uint16_t*)h_depth_img_mid_border.data;
    p_h_depth_img_right_border = (cv::uint16_t*)h_depth_img_right_border.data;
}

void CudaResource::UpdateTexture() {
    
    cudaMemcpyToArray(cuA_color_img_left, 0, 0, p_h_color_img_left, src_color_N * sizeof(COLOR_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpyToArray(cuA_color_img_mid, 0, 0, p_h_color_img_mid, src_color_N * sizeof(COLOR_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpyToArray(cuA_color_img_right, 0, 0, p_h_color_img_right, src_color_N * sizeof(COLOR_TYPE), cudaMemcpyHostToDevice);

    cudaMemcpyToArray(cuA_depth_img_left, 0, 0, p_h_depth_img_left, src_depth_N * sizeof(cv::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpyToArray(cuA_depth_img_mid, 0, 0, p_h_depth_img_mid, src_depth_N * sizeof(cv::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpyToArray(cuA_depth_img_right, 0, 0, p_h_depth_img_right, src_depth_N * sizeof(cv::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpyToArray(cuA_depth_img_left_border, 0, 0, p_h_depth_img_left_border, Border_N * sizeof(cv::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpyToArray(cuA_depth_img_mid_border, 0, 0, p_h_depth_img_mid_border, Border_N * sizeof(cv::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpyToArray(cuA_depth_img_right_border, 0, 0, p_h_depth_img_right_border, Border_N * sizeof(cv::uint16_t), cudaMemcpyHostToDevice);
    
}

void CudaResource::DestroyResource() {
    cudaDestroyTextureObject(color_tex_left_obj);
    cudaDestroyTextureObject(color_tex_mid_obj);
    cudaDestroyTextureObject(color_tex_right_obj);

    cudaDestroyTextureObject(depth_tex_left_obj);
    cudaDestroyTextureObject(depth_tex_mid_obj);
    cudaDestroyTextureObject(depth_tex_right_obj);
    cudaDestroyTextureObject(depth_tex_left_border_obj);
    cudaDestroyTextureObject(depth_tex_mid_border_obj);
    cudaDestroyTextureObject(depth_tex_right_border_obj);

    cudaDestroyTextureObject(depth_tex_rasterize_min_obj);
    cudaDestroyTextureObject(depth_tex_rasterize_max_obj);


    cudaFreeArray(cuA_color_img_left);
    cudaFreeArray(cuA_color_img_mid);
    cudaFreeArray(cuA_color_img_right);

    cudaFreeArray(cuA_depth_img_left);
    cudaFreeArray(cuA_depth_img_mid);
    cudaFreeArray(cuA_depth_img_right);
    cudaFreeArray(cuA_depth_img_left_border);
    cudaFreeArray(cuA_depth_img_mid_border);
    cudaFreeArray(cuA_depth_img_right_border);

    cudaFree(d_depth_rasterize_min);
    cudaFree(d_depth_rasterize_max);
}

__global__
void depth_img_point_splatting(
    float* depth_rasterize_min,
    float* depth_rasterize_max,
    cudaTextureObject_t depth_tex_left,
    cudaTextureObject_t depth_tex_mid,
    cudaTextureObject_t depth_tex_right) {

    /*      确定像素坐标      */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int id = gridDim.x * blockDim.x * j + i;

    /*     确定裁剪前像素坐标    */
    int pix_i = i + crop_src_pixel_depth_Nx;
    int pix_j = j + crop_src_pixel_depth_Ny;

    /*      确定对应像素深度——单位:cm        */
    float depth_left = (tex2D<cv::uint16_t>(depth_tex_left, pix_i, pix_j) / 10.0f);
    float depth_mid = (tex2D<cv::uint16_t>(depth_tex_mid, pix_i, pix_j) / 10.0f);
    float depth_right = (tex2D<cv::uint16_t>(depth_tex_right, pix_i, pix_j) / 10.0f);

    /*      深度图坐标转换到主视角相机坐标     */
    COORD_TYPE left_campos = Pix_Campos(depth_left, i, j, left_internal_param);
    COORD_TYPE left_view_campos = SrcCampos_DstCampos(
        &left_campos,
        R_left_external_param,
        T_left_external_param,
        R_view_external_param_inverse,
        T_view_external_param_inverse);
    float left_view_campos_depth = left_view_campos.z;
    PIXEL_TYPE left_view_pixel = Campos_Pix(&left_view_campos, view_internal_param);
    left_view_pixel.x /= 2;
    left_view_pixel.y /= 2;

    COORD_TYPE mid_campos = Pix_Campos(depth_mid, i, j, mid_internal_param);
    COORD_TYPE mid_view_campos = SrcCampos_DstCampos(
        &mid_campos,
        R_mid_external_param,
        T_mid_external_param,
        R_view_external_param_inverse,
        T_view_external_param_inverse);
    float mid_view_campos_depth = mid_view_campos.z;
    PIXEL_TYPE mid_view_pixel = Campos_Pix(&mid_view_campos, view_internal_param);
    mid_view_pixel.x /= 2;
    mid_view_pixel.y /= 2;

    COORD_TYPE right_campos = Pix_Campos(depth_right, i, j, right_internal_param);
    COORD_TYPE right_view_campos = SrcCampos_DstCampos(
        &right_campos,
        R_right_external_param,
        T_right_external_param,
        R_view_external_param_inverse,
        T_view_external_param_inverse);
    float right_view_campos_depth = right_view_campos.z;
    PIXEL_TYPE right_view_pixel = Campos_Pix(&right_view_campos, view_internal_param);
    right_view_pixel.x /= 2;
    right_view_pixel.y /= 2;

    
    /*      世界坐标到主视角下光栅化(点溅法)        */
    //所有深度比较基于三维重建坐标系，depth为正数，first为min，second为max
    //确定三张深度图在主视角相机下的像素坐标对应到一维数组的下标
    //这里的处理方法条件是必须与raycast核函数的线程网络一样的情况下执行
    if (left_view_pixel.x >= 0 && left_view_pixel.x < point_splatting_Nx && left_view_pixel.y >= 0 && left_view_pixel.y < point_splatting_Ny) {
        int id_left = point_splatting_Nx * left_view_pixel.y + left_view_pixel.x;
        float* local_depth_rasterize_min_left = depth_rasterize_min + id_left;
        float* local_depth_rasterize_max_left = depth_rasterize_max + id_left;

        //short4* local_color_left = color + id_left;

        //取最小深度值对应颜色      
        //(*local_color_left) = left_view_campos_depth > (*local_depth_rasterize_min_left) ? tex2D(color_tex_left, left_view_pixel.x, left_view_pixel.y) : MAKE_COLOR(0, 0, 0, 0);
        /*if (left_view_campos_depth > (*local_depth_rasterize_min_left)) {
            local_color_left->x = tex2D(color_tex_left, left_view_pixel.x, left_view_pixel.y).x;
            local_color_left->y = tex2D(color_tex_left, left_view_pixel.x, left_view_pixel.y).y;
            local_color_left->z = tex2D(color_tex_left, left_view_pixel.x, left_view_pixel.y).z;
            local_color_left->w = tex2D(color_tex_left, left_view_pixel.x, left_view_pixel.y).w;
        }*/
        //最小深度比较
        (*local_depth_rasterize_min_left) = left_view_campos_depth < (*local_depth_rasterize_min_left) ? left_view_campos_depth : (*local_depth_rasterize_min_left);
        //最大深度比较
        (*local_depth_rasterize_max_left) = left_view_campos_depth > (*local_depth_rasterize_max_left) ? left_view_campos_depth : (*local_depth_rasterize_max_left);

    }
    if (mid_view_pixel.x >= 0 && mid_view_pixel.x < point_splatting_Nx && mid_view_pixel.y >= 0 && mid_view_pixel.y < point_splatting_Ny) {
        int id_mid = point_splatting_Nx * mid_view_pixel.y + mid_view_pixel.x;
        float* local_depth_rasterize_min_mid = depth_rasterize_min + id_mid;
        float* local_depth_rasterize_max_mid = depth_rasterize_max + id_mid;

        //short4* local_color_mid = color + id_mid;

        //取最小深度值对应颜色      
        //(*local_color_mid) = mid_view_campos_depth > (*local_depth_rasterize_min_mid) ? tex2D(color_tex_mid, mid_view_pixel.x, mid_view_pixel.y) : MAKE_COLOR(0, 0, 0, 0);
        /*if (mid_view_campos_depth > (*local_depth_rasterize_min_mid)) {
            local_color_mid->x = tex2D(color_tex_mid, mid_view_pixel.x, mid_view_pixel.y).x;
            local_color_mid->y = tex2D(color_tex_mid, mid_view_pixel.x, mid_view_pixel.y).y;
            local_color_mid->z = tex2D(color_tex_mid, mid_view_pixel.x, mid_view_pixel.y).z;
            local_color_mid->w = tex2D(color_tex_mid, mid_view_pixel.x, mid_view_pixel.y).w;
        }*/
        //最小深度比较
        (*local_depth_rasterize_min_mid) = mid_view_campos_depth < (*local_depth_rasterize_min_mid) ? mid_view_campos_depth : (*local_depth_rasterize_min_mid);
        //最大深度比较
        (*local_depth_rasterize_max_mid) = mid_view_campos_depth > (*local_depth_rasterize_max_mid) ? mid_view_campos_depth : (*local_depth_rasterize_max_mid);

    }
    if (right_view_pixel.x >= 0 && right_view_pixel.x < point_splatting_Nx && right_view_pixel.y >= 0 && right_view_pixel.y < point_splatting_Ny) {
        int id_right = point_splatting_Nx * right_view_pixel.y + right_view_pixel.x;
        float* local_depth_rasterize_min_right = depth_rasterize_min + id_right;
        float* local_depth_rasterize_max_right = depth_rasterize_max + id_right;

        //short4* local_color_right = color + id_right;

        //取最小深度值对应颜色      
        //(*local_color_right) = right_view_campos_depth > (*local_depth_rasterize_min_right) ? tex2D(color_tex_right, right_view_pixel.x, right_view_pixel.y) : MAKE_COLOR(0, 0, 0, 0);
        /*if (right_view_campos_depth > (*local_depth_rasterize_min_right)) {
            local_color_right->x = tex2D(color_tex_right, right_view_pixel.x, right_view_pixel.y).x;
            local_color_right->y = tex2D(color_tex_right, right_view_pixel.x, right_view_pixel.y).y;
            local_color_right->z = tex2D(color_tex_right, right_view_pixel.x, right_view_pixel.y).z;
            local_color_right->w = tex2D(color_tex_right, right_view_pixel.x, right_view_pixel.y).w;
        }*/
        //最小深度比较
        (*local_depth_rasterize_min_right) = right_view_campos_depth < (*local_depth_rasterize_min_right) ? right_view_campos_depth : (*local_depth_rasterize_min_right);
        //最大深度比较
        (*local_depth_rasterize_max_right) = right_view_campos_depth > (*local_depth_rasterize_max_right) ? right_view_campos_depth : (*local_depth_rasterize_max_right);

    }
}

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
    cudaTextureObject_t depth_tex_rasterize_max) {

    /*      确定像素坐标      */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int id = gridDim.x * blockDim.x * j + i;
    int point_splatting_id = point_splatting_Nx * (j / 2) + (i / 2);
    /*     确定裁剪前像素坐标    */
    int pix_i = i + crop_src_pixel_depth_Nx;
    int pix_j = j + crop_src_pixel_depth_Ny;
    //int pix_id = src_pixel_Nx * pix_j + pix_i;
    /*      生成当前像素的光线表达式        */
    float u = float(i) / float(pixel_Nx);
    float v = float(j) / float(pixel_Ny);
    ray r(u, v);
    /*      确定光线步长范围        */
    float depth_min = tex1Dfetch<float>(depth_tex_rasterize_min, point_splatting_id)-0.5f;//-65.9798;
    float depth_max = tex1Dfetch<float>(depth_tex_rasterize_max, point_splatting_id)+0.5f;//-108.059;

    float step = depth_min / r.dir.z;
    float step_max = depth_max / r.dir.z;
    
    float s = 5.1e-2;

    COORD_TYPE camera_pos_left, camera_pos_mid, camera_pos_right;
    PIXEL_TYPE pixel_left, pixel_mid, pixel_right;


    float sj_left, sj_mid, sj_right;
    float wj_left, wj_mid, wj_right;

    if (depth_min < 140.0f) {

        /*      沿光线过程       */
        while (s > 5.0e-2 && step < step_max) {
            /*      重置最终tsdf值       */
            s = 0.0f;
            /*		生成光射线当前位置       */
            r.point_at_param(step);
            /*		光线从主视角坐标系转换到深度相机坐标系		*/
            camera_pos_left = SrcCampos_DstCampos(
                &r.ray_pos,
                R_view_external_param,
                T_view_external_param,
                R_left_external_param_inverse,
                T_left_external_param_inverse);
            pixel_left = Campos_Pix(&camera_pos_left, left_internal_param);
            camera_pos_mid = SrcCampos_DstCampos(
                &r.ray_pos,
                R_view_external_param,
                T_view_external_param,
                R_mid_external_param_inverse,
                T_mid_external_param_inverse);
            pixel_mid = Campos_Pix(&camera_pos_mid, mid_internal_param);
            camera_pos_right = SrcCampos_DstCampos(
                &r.ray_pos,
                R_view_external_param,
                T_view_external_param,
                R_right_external_param_inverse,
                T_right_external_param_inverse);
            pixel_right = Campos_Pix(&camera_pos_right, right_internal_param);
            /*      动态融合TSDF        */

            //左边深度图
            //判断是否在深度图像素范围 in_range =0 不在  =1 在
            //单位：cm
            float in_left_range = float(
                (pixel_left.x >= 0) &
                (pixel_left.x < pixel_Nx) &
                (pixel_left.y >= 0) &
                (pixel_left.y < pixel_Ny));
            //获取深度值
            float depth_left = (tex2D<cv::uint16_t>(depth_tex_left, int(pixel_left.x) + crop_src_pixel_depth_Nx, int(pixel_left.y) + crop_src_pixel_depth_Ny) / 10.0f);
            //计算sdf值
            sj_left = depth_left - camera_pos_left.z;
            //高方差区域降噪
            wj_left = 0.0f;
            float w_left = 0.0f;
            /*for (int it = 0; it < 9; ++it) {
                float current_depth = (tex2D<cv::uint16_t>(depth_tex_left_border, int(pixel_left.x + crop_src_pixel_depth_Nx + (it % 3)), int(pixel_left.y + crop_src_pixel_depth_Ny + (it / 3))) / 10.0f);
                float pow_temp = powf((depth_left - current_depth), 2);
                w_left += (fminf(pow_temp, 4.0f) / 9.0f);
            }*/
            float omiga_left = 0.082f;//prec / sqrtf(w_left);
            wj_left = fminf(fminf(omiga_left, 1.0f), !(sj_left < -2.0f));
            wj_left = 0.0f * (1.0f - in_left_range) + wj_left * in_left_range;
            s += wj_left * fmaxf(-2.0f, fminf(2.0f, sj_left));

            //中间深度图
            //判断是否在深度图像素范围 in_range =0 不在  =1 在
            //单位：cm
            float in_mid_range = float(
                (pixel_mid.x >= 0) &
                (pixel_mid.x < pixel_Nx) &
                (pixel_mid.y >= 0) &
                (pixel_mid.y < pixel_Ny));
            //获取深度值
            float depth_mid = (tex2D<cv::uint16_t>(depth_tex_mid, int(pixel_mid.x) + crop_src_pixel_depth_Nx, int(pixel_mid.y) + crop_src_pixel_depth_Ny) / 10.0f);
            //计算sdf值
            sj_mid = depth_mid - camera_pos_mid.z;
            //高方差区域降噪
            wj_mid = 0.0f;
            float w_mid = 0.0f;
            /*for (int it = 0; it < 9; ++it) {
                float current_depth = (tex2D<cv::uint16_t>(depth_tex_mid_border, int(pixel_mid.x + crop_src_pixel_depth_Nx + (it % 3)), int(pixel_mid.y + crop_src_pixel_depth_Ny + (it / 3))) / 10.0f);
                float pow_temp = powf((depth_mid - current_depth), 2);
                w_mid += (fminf(pow_temp, 4.0f) / 9.0f);
            }*/
            float omiga_mid = 0.082f;//prec / sqrtf(w_mid);
            wj_mid = fminf(fminf(omiga_mid, 1.0f), !(sj_mid < -2.0f));
            wj_mid = 0.0f * (1.0f - in_mid_range) + wj_mid * in_mid_range;
            s += wj_mid * fmaxf(-2.0f, fminf(2.0f, sj_mid));

            //右间深度图
            //判断是否在深度图像素范围 in_range =0 不在  =1 在
            //单位：cm
            float in_right_range = float(
                (pixel_right.x >= 0) &
                (pixel_right.x < pixel_Nx) &
                (pixel_right.y >= 0) &
                (pixel_right.y < pixel_Ny));
            //获取深度值
            float depth_right = (tex2D<cv::uint16_t>(depth_tex_right, int(pixel_right.x) + crop_src_pixel_depth_Nx, int(pixel_right.y) + crop_src_pixel_depth_Ny) / 10.0f);
            //计算sdf值
            sj_right = depth_right - camera_pos_right.z;
            //高方差区域降噪
            wj_right = 0.0f;
            float w_right = 0.0f;
            /*for (int it = 0; it < 9; ++it) {
                float current_depth = (tex2D<cv::uint16_t>(depth_tex_right_border, int(pixel_right.x + crop_src_pixel_depth_Nx + (it % 3)), int(pixel_right.y + crop_src_pixel_depth_Ny + (it / 3))) / 10.0f);
                float pow_temp = powf((depth_right - current_depth), 2);
                w_right += (fminf(pow_temp, 4.0f) / 9.0f);
            }*/
            float omiga_right = 0.082f;//prec / sqrtf(w_right);
            wj_right = fminf(fminf(omiga_right, 1.0f), !(sj_right < -2.0f));
            wj_right = 0.0f * (1.0f - in_right_range) + wj_right * in_right_range;
            s += wj_right * fmaxf(-2.0f, fminf(2.0f, sj_right));

            step += 0.8 * s;
        }
    }


    /*		生成光射线当前位置       */
    r.point_at_param(step);
    /*		光线从主视角坐标系转换到彩色相机坐标系		*/
    camera_pos_left = SrcCampos_DstCampos(
        &r.ray_pos,
        R_view_external_param,
        T_view_external_param,
        R_left_external_param_inverse,
        T_left_external_param_inverse);
    pixel_left = Campos_Pix(&camera_pos_left, left_internal_param);

    camera_pos_mid = SrcCampos_DstCampos(
        &r.ray_pos,
        R_view_external_param,
        T_view_external_param,
        R_mid_external_param_inverse,
        T_mid_external_param_inverse);
    pixel_mid = Campos_Pix(&camera_pos_mid, mid_internal_param);

    camera_pos_right = SrcCampos_DstCampos(
        &r.ray_pos,
        R_view_external_param,
        T_view_external_param,
        R_right_external_param_inverse,
        T_right_external_param_inverse);
    pixel_right = Campos_Pix(&camera_pos_right, right_internal_param);


    short3 left_color, mid_color, right_color;
    float pcf_mid = 0.0f;
    float pcf_left = 0.0f;
    float pcf_right = 0.0f;


    

    

    

    

    

    if (pixel_right.x >= 0 && pixel_right.x < pixel_Nx && pixel_right.y >= 0 && pixel_right.y < pixel_Ny) {
        float color_shadowmap_depth_right = (tex2D<cv::uint16_t>(depth_tex_right, int(pixel_right.x) + crop_src_pixel_depth_Nx, int(pixel_right.y) + crop_src_pixel_depth_Ny) / 10.0f);
        if (camera_pos_right.z >= (color_shadowmap_depth_right - 5.0f) && camera_pos_right.z <= (color_shadowmap_depth_right + 5.0f)) {
            (color + id)->x = tex2D<uchar4>(color_tex_right, int(pixel_right.x) + crop_src_pixel_color_Nx, int(pixel_right.y) + crop_src_pixel_color_Ny).x;
            (color + id)->y = tex2D<uchar4>(color_tex_right, int(pixel_right.x) + crop_src_pixel_color_Nx, int(pixel_right.y) + crop_src_pixel_color_Ny).y;
            (color + id)->z = tex2D<uchar4>(color_tex_right, int(pixel_right.x) + crop_src_pixel_color_Nx, int(pixel_right.y) + crop_src_pixel_color_Ny).z;
        }
        ////PCF 7*7计算平均可见性
        //for (int it = 0; it < 49; ++it) {
        //    float current_color_shadowmap_depth_right
        //        = -(tex2D(depth_tex_right_border, int(pixel_right.x + (it % 7)) + h_h_src_pixel_Nx, int(pixel_right.y + (it / 7)) + h_h_src_pixel_Ny) / 10.0f);
        //    pcf_right += (camera_pos_right.z >= (color_shadowmap_depth_right - 3.0f) ? 1 : 0) / 49.0f;
        //}
        //if (pcf_right > pcf_left) {
        //    (color + id)->x = tex2D(color_tex_right, pixel_right.x, pixel_right.y).x;
        //    (color + id)->y = tex2D(color_tex_right, pixel_right.x, pixel_right.y).y;
        //    (color + id)->z = tex2D(color_tex_right, pixel_right.x, pixel_right.y).z;
        //}   
    }

    if (pixel_left.x >= 0 && pixel_left.x < pixel_Nx && pixel_left.y >= 0 && pixel_left.y < pixel_Ny) {
        float color_shadowmap_depth_left = (tex2D<cv::uint16_t>(depth_tex_left, int(pixel_left.x) + crop_src_pixel_depth_Nx, int(pixel_left.y) + crop_src_pixel_depth_Ny) / 10.0f);
        if (camera_pos_left.z >= (color_shadowmap_depth_left - 5.0f) && camera_pos_left.z <= (color_shadowmap_depth_left + 5.0f)) {
            (color + id)->x = tex2D<uchar4>(color_tex_left, int(pixel_left.x) + crop_src_pixel_color_Nx, int(pixel_left.y) + crop_src_pixel_color_Ny).x;
            (color + id)->y = tex2D<uchar4>(color_tex_left, int(pixel_left.x) + crop_src_pixel_color_Nx, int(pixel_left.y) + crop_src_pixel_color_Ny).y;
            (color + id)->z = tex2D<uchar4>(color_tex_left, int(pixel_left.x) + crop_src_pixel_color_Nx, int(pixel_left.y) + crop_src_pixel_color_Ny).z;
        }
        ////PCF 7*7计算平均可见性
        //for (int it = 0; it < 49; ++it) {
        //    float current_color_shadowmap_depth_left
        //        = -(tex2D(depth_tex_left_border, int(pixel_left.x + (it % 7)) + h_h_src_pixel_Nx, int(pixel_left.y + (it / 7)) + h_h_src_pixel_Ny) / 10.0f);
        //    pcf_left += (camera_pos_left.z >= (color_shadowmap_depth_left - 3.0f) ? 1 : 0) / 49.0f;
        //}
        //if (pcf_left > pcf_mid) {
        //    (color + id)->x = pcf_left * tex2D(color_tex_left, pixel_left.x, pixel_left.y).x;
        //    (color + id)->y = pcf_left * tex2D(color_tex_left, pixel_left.x, pixel_left.y).y;
        //    (color + id)->z = pcf_left * tex2D(color_tex_left, pixel_left.x, pixel_left.y).z;
        //}

    }

    if (pixel_mid.x >= 0 && pixel_mid.x < pixel_Nx && pixel_mid.y >= 0 && pixel_mid.y < pixel_Ny) {
        float color_shadowmap_depth_mid = (tex2D<cv::uint16_t>(depth_tex_mid, int(pixel_mid.x) + crop_src_pixel_depth_Nx, int(pixel_mid.y) + crop_src_pixel_depth_Ny) / 10.0f);
        if (camera_pos_mid.z >= (color_shadowmap_depth_mid - 5.0f) && camera_pos_mid.z <= (color_shadowmap_depth_mid + 5.0f)) {
            (color + id)->x = tex2D<uchar4>(color_tex_mid, int(pixel_mid.x) + crop_src_pixel_color_Nx, int(pixel_mid.y) + crop_src_pixel_color_Ny).x;
            (color + id)->y = tex2D<uchar4>(color_tex_mid, int(pixel_mid.x) + crop_src_pixel_color_Nx, int(pixel_mid.y) + crop_src_pixel_color_Ny).y;
            (color + id)->z = tex2D<uchar4>(color_tex_mid, int(pixel_mid.x) + crop_src_pixel_color_Nx, int(pixel_mid.y) + crop_src_pixel_color_Ny).z;
        }
        ////PCF 7*7计算平均可见性
        //for (int it = 0; it < 49; ++it) {
        //    float current_color_shadowmap_depth_mid 
        //        = -(tex2D(depth_tex_mid_border, int(pixel_mid.x + (it % 7)) + h_h_src_pixel_Nx, int(pixel_mid.y + (it / 7)) + h_h_src_pixel_Ny) / 10.0f);
        //    pcf_mid += (camera_pos_mid.z >= (color_shadowmap_depth_mid - 3.0f) ? 1 : 0) / 49.0f;
        //}
        //(color + id)->x = pcf_mid * tex2D(color_tex_mid, pixel_mid.x, pixel_mid.y).x;
        //(color + id)->y = pcf_mid * tex2D(color_tex_mid, pixel_mid.x, pixel_mid.y).y;
        //(color + id)->z = pcf_mid * tex2D(color_tex_mid, pixel_mid.x, pixel_mid.y).z;

    }

    (crood + id)->x = r.ray_pos.x;
    (crood + id)->y = r.ray_pos.y;
    (crood + id)->z = r.ray_pos.z;
}