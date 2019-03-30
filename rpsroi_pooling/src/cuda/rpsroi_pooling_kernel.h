#ifndef RPS_ROI_POOLING_KERNEL
#define RPS_ROI_POOLING_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int RPSROIPoolForwardLauncher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height, const int pooled_width,
    const float* bottom_rois, const int group_size, const int output_dim, float* top_data, int* mapping_channel, float* areas,  cudaStream_t stream);


int RPSROIPoolBackwardLauncher(const float* top_diff, const int* mapping_channel, const float* areas, const int batch_size, const int num_rois, const float spatial_scale, const int channels, const int height, const int width, const int pooled_width, const int pooled_height, const int output_dim, float* bottom_diff, const float* bottom_rois, cudaStream_t stream);

#ifdef __cplusplus
}

#endif

#endif
