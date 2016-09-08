#ifndef CAFFE_LOWLEVEL_DISTANCE_LAYER_HPP_
#define CAFFE_LOWLEVEL_DISTANCE_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "ORUtils/gSLICr_defines.h"
using namespace gSLICr;

namespace caffe {

template <typename Dtype>
class LowlevelDistanceLayer : public Layer<Dtype> {
 public:
  explicit LowlevelDistanceLayer(const LayerParameter& param) :
    Layer<Dtype>(param), dim_output_(0) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "LowlevelDistance"; }
  // input : data, slic_index, dead pixel map(superpixel num), (label_map)
  virtual inline int MinBottomBlobs() const { return 3; }
  // output : (N*R)*C*H*W[initial distance maps], (N*R) [labels]
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      NOT_IMPLEMENTED;
  }
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      NOT_IMPLEMENTED;
  }

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      // pass
  }
  void PickQueryRegions(const Blob<Dtype>* dead_spmap);

  int dim_output_; // Dimension of the initial distance features
  int hf_option_;
  int sp_size_;


  Blob<Handcrafted> grid_features_; // N * 1 * G * G
  Blob<Handcrafted> query_features_; // N * R
  Blob<float> query_region_indexes_; // N * R 
  int grid_size_;
  int R_; // the number of query regions per an image.
  int N_;
  int H_;
  int W_;

};

} // namespace caffe

#endif // CAFFE_LOWLEVEL_DISTANCE_LAYER_HPP_
