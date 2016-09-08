#ifndef CAFFE_GEN_SLICINFO_LAYER_HPP_
#define CAFFE_GEN_SLICINFO_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "ORUtils/gSLICr_defines.h"
using namespace gSLICr;

#define BLOCK_SEARCH_RANGE_ 16

namespace caffe {

/**
 * @brief Generate slic information
 */
template <typename Dtype>
class GenSlicinfoLayer : public Layer<Dtype> {
 public:
  explicit GenSlicinfoLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GenSlicinfo"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
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

 private:
  float max_xy_dist_, max_color_dist_;
  int COLOR_CHAN_;
  float coh_weight_;
  Vector2i map_size_;
  Vector2i img_size_;
  int spixel_size_;
  Blob<Vector4f> cvt_img_;
  Blob<Dtype> tmp_idx_img_;
  Blob<spixel_info> spixel_map_;
  Blob<spixel_info> accum_map_; // C : thread num
  int num_block_per_superpixel_;
  int min_pixel_num_;

  void Cvt_Img_Space(const Blob<Dtype>* inimg, Blob<Vector4f>* outimg, int color_space);
  void Init_Cluster_Centers();
  void Find_Center_Association(Blob<Dtype>* idx_blob);
  void Update_Cluster_Center(Blob<Dtype>* idx_blob);
  void Enforce_Connectivity(Blob<Dtype>* idx_blob, Blob<Dtype>* dead_blob);
};

}  // namespace caffe

#endif  // CAFFE_GEN_SLICINFO_LAYER_HPP
