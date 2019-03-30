#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "rpsroi_pooling_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


__global__ void RPSROIPoolForward(const int nthreads, const float* bottom_data,
    const float spatial_scale, const int height, const int width,
    const int channels, const int pooled_height, const int pooled_width,
    const int group_size, const int output_dim,
    const float* bottom_rois, float* top_data, int* mapping_channel, float* areas)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
      	int ph = (index / pooled_width) % pooled_height;
      	int ctop = (index / pooled_width / pooled_height) % output_dim;
      	int n = index / pooled_width / pooled_height / output_dim;


        bottom_rois += n * 9;
        int roi_batch_ind = bottom_rois[0];

        float roi_x1 = static_cast<float>(round(bottom_rois[1])) * spatial_scale;
        float roi_y1 = static_cast<float>(round(bottom_rois[2])) * spatial_scale;
        float roi_x2 = static_cast<float>(round(bottom_rois[3])) * spatial_scale;
        float roi_y2 = static_cast<float>(round(bottom_rois[4])) * spatial_scale;
        float roi_x3 = static_cast<float>(round(bottom_rois[5])) * spatial_scale;
        float roi_y3 = static_cast<float>(round(bottom_rois[6])) * spatial_scale;
        float roi_x4 = static_cast<float>(round(bottom_rois[7])) * spatial_scale;
        float roi_y4 = static_cast<float>(round(bottom_rois[8])) * spatial_scale;

        ////////////////////////////////DEBUG////////////////////////////////////
        //cout << "rois: " << roi_x1 << " " << roi_y1 << " " << roi_x2 << " " << roi_y2 << " " << roi_x3 << " " << roi_y3 << " " << roi_x4 << " " << roi_y4 << endl;
        //printf("rois: %f, %f, %f, %f, %f, %f, %f, %f\n", roi_x1, roi_y1, roi_x2, roi_y2, roi_x3, roi_y3, roi_x4, roi_y4);

        float anchor_x1 = static_cast<float>(pw) * (roi_x2 - roi_x1) / pooled_width + roi_x1;
        float anchor_y1 = static_cast<float>(pw) * (roi_y2 - roi_y1) / pooled_width + roi_y1;
        float anchor_x2 = static_cast<float>(pw+1) * (roi_x2 - roi_x1) / pooled_width + roi_x1;
        float anchor_y2 = static_cast<float>(pw+1) * (roi_y2 - roi_y1) / pooled_width + roi_y1;
        float anchor_x3 = static_cast<float>(pw+1) * (roi_x3 - roi_x4) / pooled_width + roi_x4;
        float anchor_y3 = static_cast<float>(pw+1) * (roi_y3 - roi_y4) / pooled_width + roi_y4;
        float anchor_x4 = static_cast<float>(pw) * (roi_x3 - roi_x4) / pooled_width + roi_x4;
        float anchor_y4 = static_cast<float>(pw) * (roi_y3 - roi_y4) / pooled_width + roi_y4;

        ////////////////////////////////DEBUG////////////////////////////////////
        //cout << "anchor: " << anchor_x1 << " " << anchor_y1 << " " << anchor_x2 << " " << anchor_y2 << " " << anchor_x3 << " " << anchor_y3 << " " << anchor_x4 << " " << anchor_y4 <<endl;
        //printf("anchor: %f, %f, %f, %f, %f, %f, %f, %f\n", anchor_x1, anchor_y1, anchor_x2, anchor_y2, anchor_x3, anchor_y3, anchor_x4, anchor_y4);

        float grid_x1 = static_cast<float>(ph) * (anchor_x4 - anchor_x1) / pooled_height + anchor_x1;
        float grid_y1 = static_cast<float>(ph) * (anchor_y4 - anchor_y1) / pooled_height + anchor_y1;
        float grid_x4 = static_cast<float>(ph + 1) * (anchor_x4 - anchor_x1) / pooled_height + anchor_x1;
        float grid_y4 = static_cast<float>(ph + 1) * (anchor_y4 - anchor_y1) / pooled_height + anchor_y1;
        float grid_x2 = static_cast<float>(ph) * (anchor_x3 - anchor_x2) / pooled_height + anchor_x2;
        float grid_y2 = static_cast<float>(ph) * (anchor_y3 - anchor_y2) / pooled_height + anchor_y2;
        float grid_x3 = static_cast<float>(ph + 1) * (anchor_x3 - anchor_x2) / pooled_height + anchor_x2;
        float grid_y3 = static_cast<float>(ph + 1) * (anchor_y3 - anchor_y2) / pooled_height + anchor_y2;

        ////////////////////////////////DEBUG////////////////////////////////////
        //cout << "grid: " << grid_x1 << " " << grid_y1 << " " << grid_x2 << " " << grid_y2 << " " << grid_x3 << " " << grid_y3 << " " << grid_x4 << " " << grid_y4 << endl;
        // printf("grid: %f, %f, %f, %f, %f, %f, %f, %f\n", grid_x1, grid_y1, grid_x2, grid_y2, grid_x3, grid_y3, grid_x4, grid_y4);

        //printf("min:%f, %f, %f\n", grid_y1, grid_y2,  min(grid_y1, grid_y2));
        //printf("min_grid:%f, %f, %f\n", grid_y1, grid_y2, floor(min(grid_y1, grid_y2)));
        
        int hstart = static_cast<int>(floor(min(min(min(grid_y1, grid_y2) , grid_y3), grid_y4)));
        int hend = static_cast<int>(ceil(max(max(max(grid_y1, grid_y2) , grid_y3), grid_y4)));
        int wstart = static_cast<int>(floor(min(min(min(grid_x1, grid_x2) , grid_x3), grid_x4)));
        int wend = static_cast<int>(ceil(max(max(max(grid_x1, grid_x2) , grid_x3), grid_x4)));

        ///////////////////////////////DEBUG/////////////////////////////////////
        //cout << "start&&end: " << hstart << " " << hend << " " << wstart << " " << wend << endl;
        //printf("start&&end: %d, %d, %d, %d\n", hstart, hend, wstart, wend);
        
        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
      	hend = min(max(hend, 0), height);
      	wstart = min(max(wstart, 0), width);
      	wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        /////////////////////////////////////////////////////////////////////
        //cout << "start&&end norm: " << hstart << " " << hend << " " << wstart << " " << wend;
        //printf("start&&end norm: %d, %d, %d, %d\n", hstart, hend, wstart, wend);

        int gw = pw;
      	int gh = ph;
      	int c = (ctop*group_size + gh)*group_size + gw;
        // printf("c:%d %d %d %d\n", c, channels, height, width);

        bottom_data += (roi_batch_ind * channels + c) * height * width;

        //printf("get value: %d, %d, %d, %f\n", c, 270, 765, bottom_data[270*width + 765]);
        float out_sum = 0;
        float bin_area = 0;
      	for (int h = hstart; h < hend; ++h) {
      	  for (int w = wstart; w < wend; ++w) {
      	    int bottom_index = h*width + w;
            float p1 = (grid_x2 - grid_x1) * (h - grid_y1) - (w - grid_x1) * (grid_y2 - grid_y1);
            float p2 = (grid_x3 - grid_x2) * (h - grid_y2) - (w - grid_x2) * (grid_y3 - grid_y2);
            float p3 = (grid_x4 - grid_x3) * (h - grid_y3) - (w - grid_x3) * (grid_y4 - grid_y3);
            float p4 = (grid_x1 - grid_x4) * (h - grid_y4) - (w - grid_x4) * (grid_y1 - grid_y4);
            if(p1 >= 0 && p2 >= 0 && p3 >= 0 && p4 >= 0){
              out_sum += bottom_data[bottom_index];
              bin_area += 1;
            }
      	  }
      	}

        /////////////////////////////DEBUG//////////////////////////
        //cout << "bin_area: " << bin_area <<" out_sum: " << out_sum << endl;
        //printf("bin_area: %f, out_sum: %f\n", bin_area, out_sum);
      	top_data[index] = (is_empty || (bin_area ==0)) ? 0. : out_sum/bin_area;
      	mapping_channel[index] = c;
        areas[index] = bin_area;

    }
}


int RPSROIPoolForwardLauncher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    const int group_size, const int output_dim,
    float* top_data, int* mapping_channel, float* areas,  cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    const int output_size = output_dim * pooled_height * pooled_width * num_rois;
    cudaError_t err;


    RPSROIPoolForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, bottom_data, spatial_scale, height, width, channels, pooled_height,
      pooled_width, group_size, output_dim, bottom_rois, top_data, mapping_channel, areas);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


__global__ void RPSROIPoolBackward(const int nthreads, const float* top_diff,
    const int* mapping_channel, const float* areas, const int num_rois, const float spatial_scale,
    const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width, const int output_dim, float* bottom_diff,
    const float* bottom_rois) {
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {

      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 9;
      int roi_batch_ind = bottom_rois[0];

      float roi_x1 = static_cast<float>(round(bottom_rois[1])) * spatial_scale;
      float roi_y1 = static_cast<float>(round(bottom_rois[2])) * spatial_scale;
      float roi_x2 = static_cast<float>(round(bottom_rois[3])) * spatial_scale;
      float roi_y2 = static_cast<float>(round(bottom_rois[4])) * spatial_scale;
      float roi_x3 = static_cast<float>(round(bottom_rois[5])) * spatial_scale;
      float roi_y3 = static_cast<float>(round(bottom_rois[6])) * spatial_scale;
      float roi_x4 = static_cast<float>(round(bottom_rois[7])) * spatial_scale;
      float roi_y4 = static_cast<float>(round(bottom_rois[8])) * spatial_scale;

      ////////////////////////////////DEBUG////////////////////////////////////
      //cout << "rois: " << roi_x1 << " " << roi_y1 << " " << roi_x2 << " " << roi_y2 << " " << roi_x3 << " " << roi_y3 << " " << roi_x4 << " " << roi_y4 << endl;
      //printf("rois: %f, %f, %f, %f, %f, %f, %f, %f\n", roi_x1, roi_y1, roi_x2, roi_y2, roi_x3, roi_y3, roi_x4, roi_y4);

      float anchor_x1 = static_cast<float>(pw) * (roi_x2 - roi_x1) / pooled_width + roi_x1;
      float anchor_y1 = static_cast<float>(pw) * (roi_y2 - roi_y1) / pooled_width + roi_y1;
      float anchor_x2 = static_cast<float>(pw+1) * (roi_x2 - roi_x1) / pooled_width + roi_x1;
      float anchor_y2 = static_cast<float>(pw+1) * (roi_y2 - roi_y1) / pooled_width + roi_y1;
      float anchor_x3 = static_cast<float>(pw+1) * (roi_x3 - roi_x4) / pooled_width + roi_x4;
      float anchor_y3 = static_cast<float>(pw+1) * (roi_y3 - roi_y4) / pooled_width + roi_y4;
      float anchor_x4 = static_cast<float>(pw) * (roi_x3 - roi_x4) / pooled_width + roi_x4;
      float anchor_y4 = static_cast<float>(pw) * (roi_y3 - roi_y4) / pooled_width + roi_y4;

      ////////////////////////////////DEBUG////////////////////////////////////
      //cout << "anchor: " << anchor_x1 << " " << anchor_y1 << " " << anchor_x2 << " " << anchor_y2 << " " << anchor_x3 << " " << anchor_y3 << " " << anchor_x4 << " " << anchor_y4 <<endl;
      //printf("anchor: %f, %f, %f, %f, %f, %f, %f, %f\n", anchor_x1, anchor_y1, anchor_x2, anchor_y2, anchor_x3, anchor_y3, anchor_x4, anchor_y4);

      float grid_x1 = static_cast<float>(ph) * (anchor_x4 - anchor_x1) / pooled_height + anchor_x1;
      float grid_y1 = static_cast<float>(ph) * (anchor_y4 - anchor_y1) / pooled_height + anchor_y1;
      float grid_x4 = static_cast<float>(ph + 1) * (anchor_x4 - anchor_x1) / pooled_height + anchor_x1;
      float grid_y4 = static_cast<float>(ph + 1) * (anchor_y4 - anchor_y1) / pooled_height + anchor_y1;
      float grid_x2 = static_cast<float>(ph) * (anchor_x3 - anchor_x2) / pooled_height + anchor_x2;
      float grid_y2 = static_cast<float>(ph) * (anchor_y3 - anchor_y2) / pooled_height + anchor_y2;
      float grid_x3 = static_cast<float>(ph + 1) * (anchor_x3 - anchor_x2) / pooled_height + anchor_x2;
      float grid_y3 = static_cast<float>(ph + 1) * (anchor_y3 - anchor_y2) / pooled_height + anchor_y2;

      ////////////////////////////////DEBUG////////////////////////////////////
      //cout << "grid: " << grid_x1 << " " << grid_y1 << " " << grid_x2 << " " << grid_y2 << " " << grid_x3 << " " << grid_y3 << " " << grid_x4 << " " << grid_y4 << endl;
      //printf("grid: %f, %f, %f, %f, %f, %f, %f, %f\n", grid_x1, grid_y1, grid_x2, grid_y2, grid_x3, grid_y3, grid_x4, grid_y4);

      //printf("min:%f, %f, %f\n", grid_y1, grid_y2,  min(grid_y1, grid_y2));
      //printf("min_grid:%f, %f, %f\n", grid_y1, grid_y2, floor(min(grid_y1, grid_y2)));
        
      int hstart = static_cast<int>(floor(min(min(min(grid_y1, grid_y2) , grid_y3), grid_y4)));
      int hend = static_cast<int>(ceil(max(max(max(grid_y1, grid_y2) , grid_y3), grid_y4)));
      int wstart = static_cast<int>(floor(min(min(min(grid_x1, grid_x2) , grid_x3), grid_x4)));
      int wend = static_cast<int>(ceil(max(max(max(grid_x1, grid_x2) , grid_x3), grid_x4)));

      ///////////////////////////////DEBUG/////////////////////////////////////
      //cout << "start&&end: " << hstart << " " << hend << " " << wstart << " " << wend << endl;
      //printf("start&&end: %d, %d, %d, %d\n", hstart, hend, wstart, wend);
        
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0), width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Compute c at bottom
      int c = mapping_channel[index];
      float* offset_bottom_diff = bottom_diff +
        (roi_batch_ind * channels + c) * height * width;
      float bin_area = areas[index];
      float diff_val = (is_empty || (bin_area == 0)) ? 0. : top_diff[index] / bin_area;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int bottom_index = h*width + w;
          float p1 = (grid_x2 - grid_x1) * (h - grid_y1) - (w - grid_x1) * (grid_y2 - grid_y1);
          float p2 = (grid_x3 - grid_x2) * (h - grid_y2) - (w - grid_x2) * (grid_y3 - grid_y2);
          float p3 = (grid_x4 - grid_x3) * (h - grid_y3) - (w - grid_x3) * (grid_y4 - grid_y3);
          float p4 = (grid_x1 - grid_x4) * (h - grid_y4) - (w - grid_x4) * (grid_y1 - grid_y4);
          if(p1 >= 0 && p2 >= 0 && p3 >= 0 && p4 >= 0){
            atomicAdd(offset_bottom_diff + bottom_index, diff_val);
          }
        }
      }
  }
}

int RPSROIPoolBackwardLauncher(const float* top_diff, const int* mapping_channel, const float* areas, const int batch_size, const int num_rois, const float spatial_scale, const int channels,
    const int height, const int width, const int pooled_width,
    const int pooled_height, const int output_dim,
    float* bottom_diff, const float* bottom_rois, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    //const int output_size = output_dim * height * width * channels;
    const int output_size = output_dim * pooled_height * pooled_width * num_rois;
    cudaError_t err;

    RPSROIPoolBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, top_diff, mapping_channel, areas, num_rois, spatial_scale, height, width, channels, pooled_height,
      pooled_width, output_dim, bottom_diff, bottom_rois);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


#ifdef __cplusplus
}
#endif
