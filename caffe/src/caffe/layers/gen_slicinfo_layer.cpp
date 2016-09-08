#include "caffe/layers/gen_slicinfo_layer.hpp"

namespace caffe {

/*
*  Output 1: indicating assigned superpixel_index
*  Output 2: indicating dead superpixel indexes
*/
template<typename Dtype>
void GenSlicinfoLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {

  GenSlicinfoParameter slic_param = this->layer_param_.gen_slicinfo_param();
  min_pixel_num_ = slic_param.min_pixel_num();
  coh_weight_ = slic_param.coh_weight();

  vector<int> fixed_shape(4);

  // initialize private variables and output blobs
  fixed_shape[0] = bottom[0]->shape(0);
  fixed_shape[1] = 1;
  fixed_shape[2] = bottom[0]->shape(2);
  fixed_shape[3] = bottom[0]->shape(3);
  cvt_img_.Reshape(fixed_shape);
  
  tmp_idx_img_.Reshape(fixed_shape);  
  top[0]->Reshape(fixed_shape); // This will be idx_img in original gSLICr implementation

  int img_size = slic_param.image_size();
  CHECK_EQ(fixed_shape[2], img_size);

  spixel_size_ = slic_param.slic_size();
  int map_size = img_size / spixel_size_;
  map_size_.x = map_size;
  map_size_.y = map_size;
  img_size_.x = img_size;
  img_size_.y = img_size;
  fixed_shape[2] = map_size;
  fixed_shape[3] = map_size;
  if (top.size() > 1) {
   	top[1]->Reshape(fixed_shape); // dead superpixel indexes
  }

  spixel_map_.Reshape(fixed_shape);
  float total_pixel_to_search = (float)(spixel_size_*spixel_size_*9); // per superpixel
  num_block_per_superpixel_ = (int)ceil(total_pixel_to_search /
          (float)(BLOCK_SEARCH_RANGE_ * BLOCK_SEARCH_RANGE_)); // no_grid_per_center in original
  fixed_shape[1] = num_block_per_superpixel_;
  accum_map_.Reshape(fixed_shape);

  if (slic_param.cielab()) {
    COLOR_CHAN_ = gSLICr::CIELAB;
  } else {
    COLOR_CHAN_ = gSLICr::XYZ;
  }
  max_xy_dist_ = 1.0f / (1.4242f * spixel_size_);
  switch(COLOR_CHAN_) {
  case XYZ:
    max_color_dist_ = 5.0f / 1.7321f;
    break;
  case CIELAB:
    max_color_dist_ = 15.0f / (1.7321f * 128);
    break;
  }
  max_color_dist_ *= max_color_dist_;
  max_xy_dist_ *= max_xy_dist_;
}

#ifdef CPU_ONLY
STUB_GPU(GenSlicinfoLayer);
#endif

INSTANTIATE_CLASS(GenSlicinfoLayer);
REGISTER_LAYER_CLASS(GenSlicinfo);

}  // namespace caffe
