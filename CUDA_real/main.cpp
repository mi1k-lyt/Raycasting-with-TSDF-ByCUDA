#include <iostream>
#include "starline_CUDA.h"

int main() {
	/*int num;
	cudaDeviceProp myDev;
	cudaGetDeviceCount(&num);
	cudaGetDeviceProperties(&myDev, 0);

	for (int i = 0; i < num; ++i) {
		std::cout << "dev id: " << i << std::endl;
		std::cout << "dev name: " << myDev.name << std::endl;
		std::cout << "全局内存总量： " << myDev.totalGlobalMem << std::endl;
		std::cout << "常量内存总量： " << myDev.totalConstMem << std::endl;
		std::cout << "异步引擎数量： " << myDev. << std::endl;
	}*/

    cudaSetDevice(0);
    const int N = pixel_Nx * pixel_Ny;
    dim3 block(32, 15);
    dim3 grid(20, 32); //(40, 48)

	CudaResource mCuResource;
	mCuResource.LoadImg();
	mCuResource.UpdateTexture();


    /*      记录坐标数组      */
    COORD_TYPE* h_crood = (COORD_TYPE*)malloc(N * sizeof(COORD_TYPE));
    short4* h_color = (short4*)malloc(N * sizeof(short4));
    for (int i = 0; i < N; ++i) {
        (h_color + i)->x = 255;
        (h_color + i)->y = 0;
        (h_color + i)->z = 0;
        (h_color + i)->w = 0;
    }
    COORD_TYPE* d_crood;
    short4* d_color;
    cudaMalloc((void**)&d_crood, sizeof(COORD_TYPE) * N);
    cudaMalloc((void**)&d_color, sizeof(short4) * N);
    cudaMemcpy(d_crood, h_crood, sizeof(COORD_TYPE) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_color, h_color, sizeof(short4) * N, cudaMemcpyHostToDevice);

    /*      CUDA记时开始     */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    depth_img_point_splatting <<<grid, block >>> (
        mCuResource.d_depth_rasterize_min, 
        mCuResource.d_depth_rasterize_max,
        mCuResource.depth_tex_left_obj,
        mCuResource.depth_tex_mid_obj,
        mCuResource.depth_tex_right_obj);
    cudaDeviceSynchronize();


   /* float* h_depth_rasterize_min = new float[N];
    float* h_depth_rasterize_max = new float[N];
    std::string pc_name1 = "E:\\CGT\\starline\\real_data_1\\depth_img_point_splatting.txt";
    std::ofstream fout_pc_name1(pc_name1);
    fout_pc_name1 << "point_splatting" << std::endl;

    cudaMemcpy(h_depth_rasterize_min, mCuResource.d_depth_rasterize_min, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_depth_rasterize_max, mCuResource.d_depth_rasterize_max, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int it = 0; it < N; ++it) {
        fout_pc_name1 << "new: " << h_depth_rasterize_min[it] << "   " << h_depth_rasterize_max[it] << std::endl;
    }
    fout_pc_name1.close();*/

    

    ray_cast <<<grid, block >>> (
        d_crood,
        d_color,
        mCuResource.color_tex_left_obj,
        mCuResource.color_tex_mid_obj,
        mCuResource.color_tex_right_obj,
        mCuResource.depth_tex_left_obj,
        mCuResource.depth_tex_mid_obj,
        mCuResource.depth_tex_right_obj,
        mCuResource.depth_tex_left_border_obj,
        mCuResource.depth_tex_mid_border_obj,
        mCuResource.depth_tex_right_border_obj,
        mCuResource.depth_tex_rasterize_min_obj,
        mCuResource.depth_tex_rasterize_max_obj);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "cuda error is " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(h_color, d_color, sizeof(short4) * N, cudaMemcpyDeviceToHost);//隐式同步
    cudaMemcpy(h_crood, d_crood, sizeof(COORD_TYPE) * N, cudaMemcpyDeviceToHost);//隐式同步


    /*      CUDA记时结束        */
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("test_time:   %3.1f  ms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /*记录*/
    std::string pc_name = "E:\\CGT\\starline\\real_data_2\\gg.ply";
    std::ofstream fout_pc_name(pc_name);
    fout_pc_name << "ply" << std::endl;
    fout_pc_name << "format ascii 1.0" << std::endl;
    fout_pc_name << "element vertex " << N << std::endl;
    fout_pc_name << "property float x" << std::endl;
    fout_pc_name << "property float y" << std::endl;
    fout_pc_name << "property float z" << std::endl;
    fout_pc_name << "property uchar red" << std::endl;
    fout_pc_name << "property uchar green" << std::endl;
    fout_pc_name << "property uchar blue" << std::endl;
    fout_pc_name << "element face 0" << std::endl;
    fout_pc_name << "property list uchar int vertex_index" << std::endl;
    fout_pc_name << "end_header" << std::endl;

    for (int it = 0; it < N; ++it) {
        fout_pc_name << (h_crood + it)->x << " " << -(h_crood + it)->y << " " << -(h_crood + it)->z << " ";
        fout_pc_name << (h_color + it)->z << " " << (h_color + it)->y << " " << (h_color + it)->x << std::endl;

    }
    fout_pc_name.close();

    free(h_crood);
    free(h_color);
    cudaFree(d_crood);
    cudaFree(d_color);

	return 0;
}