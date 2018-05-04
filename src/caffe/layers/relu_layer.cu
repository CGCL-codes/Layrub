#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

template <typename Dtype>
__global__ void TransformData(const int n, const Dtype* in, char* relu_vec){
	CUDA_KERNEL_LOOP(index,n){relu_vec[index]=in[index]>0?1:0;}
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, negative_slope);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;

    if(this->layer_param_.phase() == TRAIN){
//	DLOG(INFO)<<"forward: layer "<<this->layer_param_.name();
      if(!char_bottom_data){
    	  char_bottom_data.reset(new SyncedMemory(count*sizeof(char)));
//    	  DLOG(INFO)<<" char_bottom_data alloc.";
      }
  	  TransformData<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(
  			  count, bottom_data, (char*)(char_bottom_data->mutable_gpu_data()));
  	  CUDA_POST_KERNEL_CHECK;
    }
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
  }
}

template <typename Dtype>
__global__ void ReLUBackward_char_data(const int n, const Dtype* in_diff,
    const char* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * (in_data[index] > 0);
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    // NOLINT_NEXT_LINE(whitespace/operators)
    if(negative_slope){
    	CHECK(0);//we assume the relu layer's negative_slope=0, so this should be failed.
    	ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    			count, top_diff, bottom_data, bottom_diff, negative_slope);
    	CUDA_POST_KERNEL_CHECK;
    } else{
	  ReLUBackward_char_data<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		count, top_diff, (char*)(char_bottom_data->gpu_data()), bottom_diff);
	  CUDA_POST_KERNEL_CHECK;
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);


}  // namespace caffe
