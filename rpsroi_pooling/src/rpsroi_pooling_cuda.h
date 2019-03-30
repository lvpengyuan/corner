int rpsroi_pooling_forward_cuda( int pooled_height, int pooled_width, float spatial_scale,int group_size, int output_dim,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output, THCudaIntTensor * mappingchannel, THCudaTensor * areas);

int rpsroi_pooling_backward_cuda(int pooled_height, int pooled_width, float spatial_scale, int output_dim,
                        THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad, THCudaIntTensor * mappingchannel, THCudaTensor * areas);
