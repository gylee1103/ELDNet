#include "caffe/layers/lowlevel_distance_layer.hpp"

namespace caffe {

template<typename Dtype>
void LowlevelDistanceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  LowlevelDistanceParameter ld_param =
    this->layer_param_.lowlevel_distance_param(); 
  hf_option_ = ld_param.feature_option();
  grid_size_ = ld_param.grid_size();
  R_ = ld_param.train_region_per_image();
  N_ = bottom[0]->shape(0);
  H_ = bottom[0]->shape(2);
  W_ = bottom[0]->shape(3);
  sp_size_ = ld_param.spixel_size();

  //TODO: change to enum
  switch(hf_option_) {
    case 0:
      dim_output_ = 164; 
      break;
    case 1:
      dim_output_ = 2;
      break;
    case 2:
      dim_output_ = 20;
      break;
    case 3:
      dim_output_ = 29;
      break;
    default:
      dim_output_ = 164; 
      break;
  }

  vector<int> fixed_shape(4);
  if (R_ == 0) {
    //CHECK_EQ(N_, 1);
    // use bottom[2] to get maximum number of regions
    fixed_shape[0] = N_ * bottom[2]->shape(2) * bottom[2]->shape(3);
    R_ = bottom[2]->shape(2) * bottom[2]->shape(3);
  } else {
    fixed_shape[0] = N_ * R_;
  }
  fixed_shape[1] = dim_output_; // Dimension of initial distance features
  fixed_shape[2] = grid_size_;
  fixed_shape[3] = grid_size_;
  top[0]->Reshape(fixed_shape); // (N*R) x C x G x G

  if (top.size() > 1) {
    vector<int> label_shape(1, fixed_shape[0]);
    top[1]->Reshape(label_shape); // (N*R)
  }

  fixed_shape[0] = N_;
  fixed_shape[1] = 1;
  grid_features_.Reshape(fixed_shape); // N x 1 x G x G

  vector<int> tmp_shape(2);
  tmp_shape[0] = N_;
  tmp_shape[1] = R_;
  query_features_.Reshape(tmp_shape); // N x R
  query_region_indexes_.Reshape(tmp_shape);
}

#ifdef CPU_ONLY
STUB_GPU(LowlevelDistanceLayer);
#endif

INSTANTIATE_CLASS(LowlevelDistanceLayer);
REGISTER_LAYER_CLASS(LowlevelDistance);
}
