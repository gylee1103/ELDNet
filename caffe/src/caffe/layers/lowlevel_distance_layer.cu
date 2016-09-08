#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/chrono.hpp>
#include "caffe/layers/lowlevel_distance_layer.hpp"

using std::cout;
using std::endl;
using cv::Mat;

namespace caffe {
// Calculate Grid features
template <typename Dtype>
__global__ void CalculateGridFeatures(const int nthreads, const int grid_size,
    const int num, const int channels, const int height, const int width,
    const Dtype* const data, Handcrafted* grid_features) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / (grid_size * grid_size);
    const int grid_index = index % (grid_size * grid_size);
    const int grid_x = grid_index % grid_size;
    const int grid_y = grid_index / grid_size;
    const int output_idx = (n * grid_size + grid_y) * grid_size + grid_x;
    const int x_start = grid_x * (width / grid_size);
    const int x_end = (grid_x + 1) * (width / grid_size) - 1;
    const int y_start = grid_y * (height / grid_size);
    const int y_end = (grid_y + 1) * (height / grid_size) - 1;
    float color_mean[10] = {0};
    float color_histo[10][8] = {0};
    // collect features

    for (int c = 0; c < channels; c++) { // (0, 1, 2) : 0-mean normalized BGR data
      const Dtype* data_slice = data + (n * channels + c) * height * width;
      for (int y = y_start; y < y_end; y++) {
        for (int x = x_start; x < x_end; x++) {
          color_mean[c] += data_slice[y * width + x];
          // Bin size : 256 / 8 = 32
          int bin_idx = data_slice[y * width + x] / 32;
          color_histo[c][bin_idx] += 1;
        }
      }
      color_mean[c] /= ((width / grid_size) * (height / grid_size));
      for (int j = 0; j < 8; j++) {
        color_histo[c][j] /= ((width / grid_size) * (height / grid_size));
      }
    }

    Handcrafted* current_hc = &(grid_features[output_idx]);
    for (int i = 0; i < channels; i++) {
      current_hc->color_mean[i] = color_mean[i];
      for (int j = 0; j < 8; j++) {
        current_hc->color_histogram[i][j] = color_histo[i][j];
      }
    }
    current_hc->center_location[0] = (y_start + y_end) / 2.0;
    current_hc->center_location[1] = (x_start + x_end) / 2.0;
  }
}

// Calculate Region features and label
template <typename Dtype>
__global__ void CalculateRegionFeatures(const int nthreads,
    const int num, const int channels, const int height, const int width, 
    const int R, const int slic_xdim, const int slic_ydim, 
    const int spixel_size, const Dtype* const data, const Dtype* const label_map,
    const Dtype* const slic_index_data, const float* const query_indexes,
    Handcrafted* query_features, Dtype* label_output) {

  CUDA_KERNEL_LOOP(c_index, nthreads) {
    // index = (n * R) + r;
    const int n = c_index / channels / R;
    const int index = c_index / channels;
    const int current_channel = c_index % channels;
    //const int r = index % R;
    const int query_sp_idx = query_indexes[index];
    const int slic_yind = query_sp_idx / slic_xdim;
    const int slic_xind = query_sp_idx % slic_xdim;
    const int x_start = max((slic_xind - 2) * spixel_size, 0);
    const int x_end = min((slic_xind + 2) * spixel_size, width);
    const int y_start = max((slic_yind - 2) * spixel_size, 0);
    const int y_end = min((slic_yind + 2) * spixel_size, height);
    float color_mean = 0;
    float color_histo[8] = {0};
    float center_x = 0;
    float center_y = 0;
    if (label_map != NULL && current_channel == 0) {
      int count = 0;
      label_output[index] = 0;
      for (int y = y_start; y < y_end; y++) {
        for (int x = x_start; x < x_end; x++) {
          if (slic_index_data[(n * height + y) * width + x] == query_sp_idx) {
            // label
            label_output[index] += label_map[(n * height + y) * width + x];
            count += 1;
          }
        }
      }
      if (count != 0) {
        label_output[index] /= count;
      } else {
        label_output[index] = 255;
      }
    }
      
    const Dtype* data_slice = data + (n * channels + current_channel) * height * width;
    int count = 0;
    for (int y = y_start; y < y_end; y++) {
      for (int x = x_start; x < x_end; x++) {
        if (slic_index_data[(n * height + y) * width + x] == query_sp_idx) {
          count += 1;
          color_mean += data_slice[y * width + x];
          int bin_idx = data_slice[y * width + x] / 32;
          color_histo[bin_idx] += 1;
          if (current_channel == 0) {
            center_x += x;
            center_y += y;
          }
        }
      }
    }
    if (count == 0) {
      count = 1;
    }
    color_mean /= count;
    if (current_channel == 0) {
      center_x /= count;
      center_y /= count;
    }
    for (int j = 0; j < 8; j++) {
      color_histo[j] /= count;
    }

    Handcrafted* current_hc = &(query_features[index]);
    for (int i = 0; i < channels; i++) {
      current_hc->color_mean[current_channel] = color_mean;
      for (int j = 0; j < 8; j++) {
        current_hc->color_histogram[current_channel][j] = color_histo[j];
      }
    }
    if (current_channel == 0) {
      current_hc->center_location[0] = center_y;
      current_hc->center_location[1] = center_x;
    }
  }
}

__device__ inline int GetOffset(const int n, const int c, const int h,
    const int w, const int C, const int H, const int W) {
  return (((n * C + c) * H + h) * W + w);
}

// Calculate distance between query and grid features
template <typename Dtype>
__global__ void CalculateDistanceBetweenQueryAndGrid(const int nthreads,
    const int N, const int R, const int channels, const int height,
    const int width, const int grid_size,
    const int dim_output, const Handcrafted* grid_features,
    const Handcrafted* query_features, Dtype* const top_data, int option) {

  CUDA_KERNEL_LOOP(c_index, nthreads) {
    const int n = c_index / channels / R;
    const int index = c_index / channels;
    const int current_channel = c_index % channels;

    //const int r = index % R;

    const Handcrafted* sliced_grid_features =
      &(grid_features[n * grid_size * grid_size]);
    const Handcrafted* current_query_features = &(query_features[index]);

    for (int gx = 0; gx < grid_size; gx++) {
      for (int gy = 0; gy < grid_size; gy++) {
        int grid_index = gy * grid_size + gx;
        int chidx = current_channel * (dim_output - 2) / channels; // top channel index

        // BGR color difference
        float grid_val = sliced_grid_features[grid_index].color_mean[current_channel];
        float query_val = current_query_features->color_mean[current_channel];

        if (option != 1) {
          // Query region val
          *(top_data + GetOffset(index, chidx, gy, gx, 
                dim_output, grid_size, grid_size)) = query_val/256 - 0.5;
          chidx += 1;

          // Current grid val (to normalize, subtract 0.5)
          *(top_data + GetOffset(index, chidx, gy, gx, 
                dim_output, grid_size, grid_size)) = grid_val/256 - 0.5;
          chidx += 1;
        }
        if (option == 0) {
          // histogram 
          for (int b = 0; b < 8; b++) {
            float tmp1 = sliced_grid_features[grid_index].color_histogram[current_channel][b];
            float tmp2 = current_query_features->color_histogram[current_channel][b];
            *(top_data + GetOffset(index, chidx, gy, gx, 
                dim_output, grid_size, grid_size)) = tmp1;
            chidx += 1;
            *(top_data + GetOffset(index, chidx, gy, gx, 
                dim_output, grid_size, grid_size)) = tmp2;
            chidx += 1;
          }
        } else if (option == 3) {
          float sum = 0;
          for (int b = 0; b < 8; b++) {
            float tmp1 = sliced_grid_features[grid_index].color_histogram[current_channel][b];
            float tmp2 = current_query_features->color_histogram[current_channel][b];
            sum += 2 * (tmp1 - tmp2) * (tmp1 - tmp2) / (tmp1 + tmp2 + 0.00000001);
          }
          *(top_data + GetOffset(index, chidx, gy, gx, 
                dim_output, grid_size, grid_size)) = sum / 4.0;
          chidx += 1;
        }
        
        

        if (current_channel == channels - 1) {
          // Location difference
          *(top_data + GetOffset(index, dim_output - 2, gy, gx, 
                dim_output, grid_size, grid_size)) =
            (sliced_grid_features[grid_index].center_location[0] -
            current_query_features->center_location[0])/(height/2);
          *(top_data + GetOffset(index, dim_output - 1, gy, gx, 
                dim_output, grid_size, grid_size)) =
            (sliced_grid_features[grid_index].center_location[1] -
            current_query_features->center_location[1])/(width/2);
        }
      }
    }
    
  }
}

// Pick R regions to query
template<typename Dtype>
void LowlevelDistanceLayer<Dtype>::PickQueryRegions(const Blob<Dtype>* dead_spmap) {
  int N = dead_spmap->shape(0);
  int map_h = dead_spmap->shape(2);
  int map_w = dead_spmap->shape(3);
  if (R_ == map_w * map_h * N) {
    // test phase. query all regions
    float* qindx_ptr = query_region_indexes_.mutable_cpu_data();
    for (int i = 0; i < R_; i++) {
      *qindx_ptr = i;
      qindx_ptr += query_region_indexes_.offset(0, 1);
    }
  } else {
    float* const qindx_ptr = query_region_indexes_.mutable_cpu_data();
    for (int n = 0; n < N; n++) {
      vector<float> candidates;
      // Pick R regions
      const Dtype* dead_spmap_ptr = dead_spmap->cpu_data() + dead_spmap->offset(n);
      for (int w = 0; w < map_w; w++) {
        for (int h = 0; h < map_h; h++) {
          bool is_dead = (dead_spmap_ptr[dead_spmap->offset(0, 0, h, w)] == 1);
          if (!is_dead) {
            candidates.push_back(h*map_w + w);
          }
        }
      }
      // randomly pick R superpixels
      std::random_shuffle(candidates.begin(), candidates.end());
      for (int r = 0; r < R_; r++) {
        qindx_ptr[query_region_indexes_.offset(n, r)] = candidates[r];
      }
    }
  }
}

template<typename Dtype>
void generate_multicolor_data(const Blob<Dtype>& data,
    Blob<Dtype>* const mcs_data) {
  const Dtype* data_cpu = data.cpu_data();
  Dtype* mcs_data_cpu_ptr = mcs_data->mutable_cpu_data();
  int num = data.shape(0);
  int channels = data.shape(1);
  int height = data.shape(2);
  int width = data.shape(3);
  int mcs_channels = mcs_data->shape(1);
  const unsigned char MEAN[3] = {104, 117, 123};
  cv::Mat bgr_mat(height, width, CV_8UC3);
  cv::Mat hsv_mat;
  cv::Mat lab_mat;
  for (int n = 0; n < num; n++) {
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          bgr_mat.at<cv::Vec3b>(h, w)[c] = static_cast<unsigned char>(
            data_cpu[((n * channels + c) * height + h) * width + w] + MEAN[c]);
        }
      }
    }
    cv::cvtColor(bgr_mat, hsv_mat, CV_BGR2HSV);
    cv::cvtColor(bgr_mat, lab_mat, CV_BGR2Lab);
    // convert to blob
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          mcs_data_cpu_ptr[((n * mcs_channels + c) * height + h) * width + w] = 
            static_cast<float>(bgr_mat.at<cv::Vec3b>(h, w)[c]);
          mcs_data_cpu_ptr[((n * mcs_channels + c + 3) * height + h) * width + w] =
            static_cast<float>(hsv_mat.at<cv::Vec3b>(h, w)[c]);
          mcs_data_cpu_ptr[((n * mcs_channels + c + 6) * height + h) * width + w] =
            static_cast<float>(lab_mat.at<cv::Vec3b>(h, w)[c]);
        }
      }
    }
  }
}


// ---------------------------------------------------
// 
//  Caffe forward implementation
//
// ---------------------------------------------------
template <typename Dtype>
void LowlevelDistanceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  // bottom[0] : data, bottom[1] : slic_index, bottom[2] : dead map
  // bottom[3] : label_map

  // Prepare various channel data and gabor filtered map
  // data blob to Mat
  CHECK_EQ(bottom[0]->shape(1), 3);
  const int mcs_channels = 9;
  Blob<Dtype> mcs_data(N_, mcs_channels, H_, W_); // multi colorspaces data
  generate_multicolor_data(*bottom[0], &mcs_data);

  // Calculate grid features
  int block_cnt = N_ * grid_size_ * grid_size_;
  Handcrafted* grid_fptr = grid_features_.mutable_gpu_data();
  const Dtype* mcs_gpu_ptr = mcs_data.mutable_gpu_data();

  CalculateGridFeatures<<<CAFFE_GET_BLOCKS(block_cnt),
    CAFFE_CUDA_NUM_THREADS>>>(block_cnt, grid_size_, N_, mcs_channels,
        H_, W_, mcs_gpu_ptr, grid_fptr);

  
  // Select R regions per an image for training
   
  PickQueryRegions(bottom[2]);
  
  
  // Calculate query features and its label
  // N * R
  block_cnt = N_ * R_ * mcs_channels;
  int slic_xdim = bottom[2]->shape(3);
  int slic_ydim = bottom[2]->shape(2);
  const Dtype* labelmap_ptr = NULL;
  if (bottom.size() > 3) {
    labelmap_ptr = bottom[3]->gpu_data();
  }
  const Dtype* slic_idx_ptr = bottom[1]->gpu_data();
  const float* query_idx_ptr = query_region_indexes_.gpu_data();
  Handcrafted* query_fptr = query_features_.mutable_gpu_data();
  Dtype* label_output = NULL;
  if (top.size() > 1) {
    label_output = top[1]->mutable_gpu_data();
  }
  
  CalculateRegionFeatures<<<CAFFE_GET_BLOCKS(block_cnt),
    CAFFE_CUDA_NUM_THREADS>>>(block_cnt, N_, mcs_channels, H_, W_, R_,
        slic_xdim, slic_ydim, sp_size_, mcs_gpu_ptr, labelmap_ptr,
        slic_idx_ptr, query_idx_ptr, query_fptr, label_output);
 
  
  // Calculate low level distance
  const Handcrafted* updated_grid_fptr = grid_features_.gpu_data();
  const Handcrafted* updated_query_fptr = query_features_.gpu_data();

  CalculateDistanceBetweenQueryAndGrid<<<CAFFE_GET_BLOCKS(block_cnt),
    CAFFE_CUDA_NUM_THREADS>>>(block_cnt, N_, R_, mcs_channels, H_, W_, grid_size_, 
        dim_output_, updated_grid_fptr, updated_query_fptr,
        top[0]->mutable_gpu_data(), hf_option_);
/*
  const Dtype* check = top[0]->cpu_data();
  int n = 0;
  for (int c = 0; c < 29; c++) {
    for (int i = 0; i < 18; i++) {
      for (int j = 0; j < 18; j ++) {
        cout << *(check + (((n * R_ + 1)* dim_output_ + c)*18 + i) *18 +  j) << ", ";
      }
      cout << endl;
    }
    cout << "------------------------------------------" << endl;
  }
  cout << "============================================" << endl;
  
  */
}

INSTANTIATE_LAYER_GPU_FUNCS(LowlevelDistanceLayer);

} // namespace caffe
