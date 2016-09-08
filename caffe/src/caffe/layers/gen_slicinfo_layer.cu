#include <stdio.h>
#include <iostream>
#include <boost/chrono.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "caffe/layers/gen_slicinfo_layer.hpp"
using namespace gSLICr;
using namespace cv;
using std::cout;
using std::endl;

namespace caffe {

// ----------------------------------------------------
//
//	Image Space Conversion
//
// ----------------------------------------------------
template<typename Dtype>
__global__ void Cvt_Img_Space_device(const int nthreads, const Dtype* inimg,
        Vector4f* outimg, Vector2i img_size, int color_space) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // input index : (n, h, w) => blob index : (n, c, h, w)
        const int max_height = img_size.y;
        const int max_width = img_size.x;
        const int w = index % max_width;
        const int h = (index / max_width) % max_height;
        const int n = (index / max_width) / max_height;
        const int b_index = ((n * 3 + 0) * max_height + h) * max_width + w;
        const int g_index = ((n * 3 + 1) * max_height + h) * max_width + w;
        const int r_index = ((n * 3 + 2) * max_height + h) * max_width + w;
        float _b = (float)(inimg[b_index] + 104)* 0.0039216f;
        float _g = (float)(inimg[g_index] + 117)* 0.0039216f;
        float _r = (float)(inimg[r_index] + 123)* 0.0039216f;
        float x = _r*0.412453f + _g*0.357580f + _b*0.180423f;
        float y = _r*0.212671f + _g*0.715160f + _b*0.072169f;
        float z = _r*0.019334f + _g*0.119193f + _b*0.950227f;
        switch (color_space)
        {
            case gSLICr::XYZ:
                outimg[index].x = x;
                outimg[index].y = y;
                outimg[index].z = z;
                break;
            case gSLICr::CIELAB:
                float epsilon = 0.008856f;	//actual CIE standard
                float kappa = 903.3f;		//actual CIE standard
                float Xr = 0.950456f;	//reference white
                float Yr = 1.0f;		//reference white
                float Zr = 1.088754f;	//reference white
                float xr = x / Xr;
                float yr = y / Yr;
                float zr = z / Zr;
                float fx, fy, fz;
                if (xr > epsilon)	fx = pow(xr, 1.0f / 3.0f);
                else				fx = (kappa*xr + 16.0f) / 116.0f;
                if (yr > epsilon)	fy = pow(yr, 1.0f / 3.0f);
                else				fy = (kappa*yr + 16.0f) / 116.0f;
                if (zr > epsilon)	fz = pow(zr, 1.0f / 3.0f);
                else				fz = (kappa*zr + 16.0f) / 116.0f;

                outimg[index].x = 116.0f*fy - 16.0f;
                outimg[index].y = 500.0f*(fx - fy);
                outimg[index].z = 200.0f*(fy - fz);
                break;
        }
    }
}

template<typename Dtype>
void GenSlicinfoLayer<Dtype>::Cvt_Img_Space(const Blob<Dtype>* inimg, Blob<Vector4f>* outimg,
	   	int color_space) {
	const Dtype* inimg_ptr = inimg->gpu_data();
	Vector4f* outimg_ptr = outimg->mutable_gpu_data();
    const int count = outimg->count();

	Cvt_Img_Space_device<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, inimg_ptr, outimg_ptr, img_size_, color_space);
}

// ----------------------------------------------------
//
//	Cluster Center Initialisation
//
// ----------------------------------------------------

__device__ inline void init_cluster_centers_shared(const Vector4f* inimg,
        spixel_info* out_spixel, Vector2i map_size,
        Vector2i img_size, int spixel_size, int x, int y) {
	int cluster_idx = y * map_size.x + x;
	int img_x = x * spixel_size + spixel_size / 2;
	int img_y = y * spixel_size + spixel_size / 2;
	img_x = img_x >= img_size.x ? (x * spixel_size + img_size.x) / 2 : img_x;
	img_y = img_y >= img_size.y ? (y * spixel_size + img_size.y) / 2 : img_y;

	// TODO: go one step towards gradients direction
	out_spixel[cluster_idx].id = cluster_idx;
	out_spixel[cluster_idx].center = Vector2f((float)img_x, (float)img_y);
	out_spixel[cluster_idx].color_info = inimg[img_y*img_size.x + img_x];
	out_spixel[cluster_idx].no_pixels = 0;
}

__global__ void Init_Cluster_Centers_device(const int nthreads, const Vector4f* inimg,
        spixel_info* out_spixel, Vector2i map_size, Vector2i img_size, int spixel_size) {
    CUDA_KERNEL_LOOP(index, nthreads) { // index : N x 1 x map_size x map_size
        const int x = index % map_size.x;
        const int y = (index / map_size.x) % map_size.y;
        const int n = (index / map_size.x) / map_size.y;
        const int start_index_img_n = n*img_size.y*img_size.x;
        const int start_index_map_n = n*map_size.y*map_size.x;
        const Vector4f* inimg_n = &(inimg[start_index_img_n]);
        spixel_info* out_spixel_n = &(out_spixel[start_index_map_n]);
	    init_cluster_centers_shared(inimg_n, out_spixel_n, map_size, img_size,
                spixel_size, x, y);
    }
}

template<typename Dtype>
void GenSlicinfoLayer<Dtype>::Init_Cluster_Centers() {
	spixel_info* spixel_list = spixel_map_.mutable_gpu_data();
	const Vector4f* img_ptr = cvt_img_.gpu_data();
    const int count = spixel_map_.count();

	Init_Cluster_Centers_device<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, img_ptr, spixel_list, map_size_, img_size_, spixel_size_);
}

// ----------------------------------------------------
//
//	Finding the Cluster Associations
//
// ----------------------------------------------------

__device__ inline float compute_slic_distance(const Vector4f& pix, int x, int y,
        const spixel_info& center_info, float weight, float normalizer_xy,
        float normalizer_color) {
	float dcolor = (pix.x - center_info.color_info.x)*(pix.x - center_info.color_info.x)
				 + (pix.y - center_info.color_info.y)*(pix.y - center_info.color_info.y)
				 + (pix.z - center_info.color_info.z)*(pix.z - center_info.color_info.z);

	float dxy = (x - center_info.center.x) * (x - center_info.center.x)
			  + (y - center_info.center.y) * (y - center_info.center.y);

	float retval = dcolor * normalizer_color + weight * dxy * normalizer_xy;

	return sqrtf(retval);
}

template<typename Dtype>
__device__ inline void find_center_association_shared(const Vector4f* inimg,
        const spixel_info* in_spixel_map, Dtype* out_idx_img, 
        Vector2i map_size, Vector2i img_size, int spixel_size, float weight,
        int n, int x, int y, float max_xy_dist, float max_color_dist) {
	int idx_img = (n*img_size.y + y)*img_size.x + x;
	int ctr_x = x / spixel_size;
	int ctr_y = y / spixel_size;
	Dtype minidx = -1;
	float dist = 999999.9999f;

	// search 3x3 neighborhood
	for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++)
	{
		int ctr_x_check = ctr_x + j;
		int ctr_y_check = ctr_y + i;
		if (ctr_x_check >= 0 && ctr_y_check >= 0 && ctr_x_check < map_size.x &&
                ctr_y_check < map_size.y) {
			int ctr_idx = (n*map_size.y + ctr_y_check)*map_size.x + ctr_x_check;
			float cdist = compute_slic_distance(inimg[idx_img], x, y, in_spixel_map[ctr_idx],
                    weight, max_xy_dist, max_color_dist);
			if (cdist < dist) {
				dist = cdist;
				minidx = in_spixel_map[ctr_idx].id;
			}
		}
	}
	if (minidx >= 0) out_idx_img[idx_img] = minidx;
}

template<typename Dtype>
__global__ void Find_Center_Association_device(const int nthreads, const Vector4f* inimg,
        const spixel_info* in_spixel_map, Dtype* out_idx_img, Vector2i map_size, 
        Vector2i img_size, int spixel_size, float weight, float max_xy_dist,
        float max_color_dist) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // index: (n, h, w)
        const int x = index % img_size.x;
        const int y = (index / img_size.x) % img_size.y;
        const int n = (index / img_size.x) / img_size.y;
	    find_center_association_shared(inimg, in_spixel_map, out_idx_img, map_size, img_size,
                spixel_size, weight, n, x, y, max_xy_dist, max_color_dist);
    }
}

template<typename Dtype>
void GenSlicinfoLayer<Dtype>::Find_Center_Association(Blob<Dtype>* idx_blob) {
	spixel_info* spixel_list = spixel_map_.mutable_gpu_data();
	const Vector4f* img_ptr = cvt_img_.gpu_data();
	Dtype* idx_ptr = idx_blob->mutable_gpu_data();
    const int count = cvt_img_.count();

	Find_Center_Association_device<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, img_ptr, spixel_list, idx_ptr, map_size_, img_size_, spixel_size_,
            coh_weight_, max_xy_dist_, max_color_dist_);
}

// ----------------------------------------------------
//
//	Updating the cluster center
//
// ----------------------------------------------------

template<typename Dtype>
__global__ void Update_Cluster_Center_device(const Vector4f* inimg,
        const Dtype* in_idx_img, spixel_info* accum_map, Vector2i map_size, Vector2i img_size,
        int spixel_size, int no_blocks_per_line) {
	int local_id = threadIdx.y * blockDim.x + threadIdx.x; // thread index of one superpixel
    // blockIdx.x : image index(N), blockIdx.y : superpixel index(mapsize^2)
    // blockIdx.z = block index(num_blocks_per_superpixel)
    const int num_blocks_per_superpixel_ = gridDim.z;
    const int BLOCK_DIM = BLOCK_SEARCH_RANGE_;
	int spixel_id = blockIdx.y; // superpixel index
    int spixel_index_x = spixel_id % map_size.x;
    int spixel_index_y = spixel_id / map_size.x;


	__shared__ Vector4f color_shared[BLOCK_DIM*BLOCK_DIM];
	__shared__ Vector2f xy_shared[BLOCK_DIM*BLOCK_DIM];
	__shared__ int count_shared[BLOCK_DIM*BLOCK_DIM];
	__shared__ bool should_add; 

	color_shared[local_id] = Vector4f(0, 0, 0, 0);
	xy_shared[local_id] = Vector2f(0, 0);
	count_shared[local_id] = 0;
	should_add = false;
	__syncthreads();


	// compute the relative position in the search window
	int block_x = blockIdx.z % no_blocks_per_line;
	int block_y = blockIdx.z / no_blocks_per_line;

	int x_offset = block_x * BLOCK_DIM + threadIdx.x;
	int y_offset = block_y * BLOCK_DIM + threadIdx.y;

	if (x_offset < spixel_size * 3 && y_offset < spixel_size * 3)
	{
		// compute the start of the search window
		int x_start = spixel_index_x * spixel_size - spixel_size;	
		int y_start = spixel_index_y * spixel_size - spixel_size;

		int x_img = x_start + x_offset;
		int y_img = y_start + y_offset;

		if (x_img >= 0 && x_img < img_size.x && y_img >= 0 && y_img < img_size.y)
		{
			int img_idx = (blockIdx.x*img_size.y + y_img)*img_size.x + x_img;
			if (in_idx_img[img_idx] == spixel_id)
			{
				color_shared[local_id] = inimg[img_idx];
				xy_shared[local_id] = Vector2f(x_img, y_img);
				count_shared[local_id] = 1;
				should_add = true;
			}
		}
	}
	__syncthreads();

	if (should_add)
	{
		if (local_id < 128)
		{
			color_shared[local_id] += color_shared[local_id + 128];
			xy_shared[local_id] += xy_shared[local_id + 128];
			count_shared[local_id] += count_shared[local_id + 128];
		}
		__syncthreads();

		if (local_id < 64)
		{
			color_shared[local_id] += color_shared[local_id + 64];
			xy_shared[local_id] += xy_shared[local_id + 64];
			count_shared[local_id] += count_shared[local_id + 64];
		}
		__syncthreads();

		if (local_id < 32)
		{
			color_shared[local_id] += color_shared[local_id + 32];
			color_shared[local_id] += color_shared[local_id + 16];
			color_shared[local_id] += color_shared[local_id + 8];
			color_shared[local_id] += color_shared[local_id + 4];
			color_shared[local_id] += color_shared[local_id + 2];
			color_shared[local_id] += color_shared[local_id + 1];

			xy_shared[local_id] += xy_shared[local_id + 32];
			xy_shared[local_id] += xy_shared[local_id + 16];
			xy_shared[local_id] += xy_shared[local_id + 8];
			xy_shared[local_id] += xy_shared[local_id + 4];
			xy_shared[local_id] += xy_shared[local_id + 2];
			xy_shared[local_id] += xy_shared[local_id + 1];

			count_shared[local_id] += count_shared[local_id + 32];
			count_shared[local_id] += count_shared[local_id + 16];
			count_shared[local_id] += count_shared[local_id + 8];
			count_shared[local_id] += count_shared[local_id + 4];
			count_shared[local_id] += count_shared[local_id + 2];
			count_shared[local_id] += count_shared[local_id + 1];
		}
	}
	__syncthreads();

	if (local_id == 0)
	{
        const int n = blockIdx.x;
		int accum_map_idx = ((n*num_blocks_per_superpixel_+blockIdx.z)*map_size.y + 
            spixel_index_y)*map_size.x + spixel_index_x;
		accum_map[accum_map_idx].center = xy_shared[0];
		accum_map[accum_map_idx].color_info = color_shared[0];
		accum_map[accum_map_idx].no_pixels = count_shared[0];
	}
}

__device__ inline void finalize_reduction_result_shared(
        const spixel_info* accum_map,
        spixel_info* spixel_list, Vector2i map_size,
        int no_blocks_per_spixel, int x, int y, int n) {
	int spixel_idx = (n*map_size.y + y)*map_size.x + x;

	spixel_list[spixel_idx].center = Vector2f(0, 0);
	spixel_list[spixel_idx].color_info = Vector4f(0, 0, 0, 0);
	spixel_list[spixel_idx].no_pixels = 0;

	for (int i = 0; i < no_blocks_per_spixel; i++)
	{
		int accum_list_idx = ((n*no_blocks_per_spixel + i)*map_size.y + y)*map_size.x + x;
		spixel_list[spixel_idx].center += accum_map[accum_list_idx].center;
		spixel_list[spixel_idx].color_info += accum_map[accum_list_idx].color_info;
		spixel_list[spixel_idx].no_pixels += accum_map[accum_list_idx].no_pixels;
	}

	if (spixel_list[spixel_idx].no_pixels != 0)
	{
		spixel_list[spixel_idx].center /= (float)spixel_list[spixel_idx].no_pixels;
		spixel_list[spixel_idx].color_info /= (float)spixel_list[spixel_idx].no_pixels;
	}
}

__global__ void Finalize_Reduction_Result_device(const int nthreads, 
        const spixel_info* accum_map, spixel_info* spixel_list, Vector2i map_size,
        int no_blocks_per_spixel) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // index from (N, smap_x, smap_y)
        const int x = index % map_size.x;
        const int y = (index / map_size.x) % map_size.y;
        const int n = (index / map_size.x) / map_size.y;
	    finalize_reduction_result_shared(accum_map, spixel_list, map_size, 
            no_blocks_per_spixel, x, y, n);
    }
}

template<typename Dtype>
void GenSlicinfoLayer<Dtype>::Update_Cluster_Center(Blob<Dtype>* idx_blob) {
	spixel_info* accum_map_ptr = accum_map_.mutable_gpu_data();
	spixel_info* spixel_list_ptr = spixel_map_.mutable_gpu_data();
	const Vector4f* img_ptr = cvt_img_.gpu_data();
	const Dtype* idx_ptr = idx_blob->gpu_data();
  const int N = idx_blob->shape(0);

  int no_blocks_per_line = spixel_size_ * 3 / BLOCK_SEARCH_RANGE_;

  // Cannot use Caffe thread setting (use shared variable)
	dim3 blockSize(BLOCK_SEARCH_RANGE_, BLOCK_SEARCH_RANGE_);
  dim3 gridSize(N, map_size_.x * map_size_.y, num_block_per_superpixel_);
	// BLOCK_DIM: BLOCK_SEARCH_RANGE

  const int count = spixel_map_.count();

	Update_Cluster_Center_device<<<gridSize, blockSize>>>(img_ptr, idx_ptr,
            accum_map_ptr, map_size_, img_size_, spixel_size_, no_blocks_per_line);

	Finalize_Reduction_Result_device<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, accum_map_ptr, spixel_list_ptr, map_size_, num_block_per_superpixel_);
}

// ----------------------------------------------------
//
//	Enforce connectivity
//
// ----------------------------------------------------
template<typename Dtype>
void GenSlicinfoLayer<Dtype>::Enforce_Connectivity(Blob<Dtype>* idx_blob,
		Blob<Dtype>* dead_blob)
{
	const int N = idx_blob->shape(0);
	// Merge superpixels which have too small number of pixels
	// Indicate dead superpixels
	for (int n = 0; n < N; n++) {
		Dtype* idx_ptr = idx_blob->mutable_cpu_data() + idx_blob->offset(n, 0, 0, 0);
		Dtype* dead_ptr = dead_blob->mutable_cpu_data() + dead_blob->offset(n, 0, 0, 0);

		// loop superpixel index
		for (int slic_yind = 0; slic_yind < map_size_.y; slic_yind++) {
			for (int slic_xind = 0; slic_xind < map_size_.x; slic_xind++) {
				int slic_index = slic_yind*map_size_.x + slic_xind;
				const int hstart = max((slic_yind - 4) * spixel_size_, 0);
				const int wstart = max((slic_xind - 4) * spixel_size_, 0);
				const int hend = min((slic_yind + 4) * spixel_size_, img_size_.y);
				const int wend = min((slic_xind + 4) * spixel_size_, img_size_.x);
				int count = 0;
				for (int h = hstart; h < hend; ++h) {
					for (int w = wstart; w < wend; ++w) {
						if (idx_ptr[h * img_size_.x + w] == slic_index) {
							count++;
						}
					}
				}
				if (count < min_pixel_num_) {
					// merge this superpixel with near other one.
					dead_ptr[slic_index] = 1;
					int new_slic_index;
					if (slic_xind != map_size_.x - 1) {
						new_slic_index = slic_index + 1;
					} else if (slic_yind != map_size_.y - 1) {
						new_slic_index = (slic_yind + 1)*map_size_.x + slic_xind;
					} else {
						new_slic_index = slic_index - 1;
					}
					// find surrounded labels and pick most similar color sp
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							if (idx_ptr[h * img_size_.x + w] == slic_index) {
								idx_ptr[h * img_size_.x + w] = new_slic_index;
							}
						}
					}
				} else {
					dead_ptr[slic_index] = 0;
				}
			}
		}
	}
}

template<typename Dtype>
void Visualize(Blob<Dtype>* top, int spixel_xdim, int spixel_ydim) {
	// visualize... 3rd one?
	int img_index = 0;
	const Dtype* data = top->cpu_data();
	Mat vismap(top->shape(2), top->shape(3), CV_8UC3);
	for (int r = 0; r < top->shape(2); r++) {
		for (int c = 0; c < top->shape(3); c++) {
			int index = (img_index*top->shape(2) + r)*top->shape(3) + c;
			Dtype value = data[index];
			int slic_xindex = (int)value % spixel_xdim;
			int slic_yindex = (int)value / spixel_xdim;
			Vec3b color( slic_xindex/(float)spixel_xdim*255,
					slic_yindex/(float)spixel_ydim*255, 0);
			//Vec3b color( slic_xindex/(float)spixel_xdim*255,slic_xindex/(float)spixel_xdim*255,
			//			slic_xindex/(float)spixel_xdim*255);

			vismap.at<Vec3b>(r, c) = color;
		}
	}
	imshow("ahahaha", vismap);
	waitKey(0);
}
template<typename Dtype>
void Visualize22(Blob<Dtype>* top, int spixel_xdim, int spixel_ydim, Blob<Dtype>* img) {
	// visualize... 3rd one?
	int img_index = 0;
	const Dtype* data = top->cpu_data();
  const Dtype* img_data = img->cpu_data();
	Mat vismap(top->shape(2), top->shape(3), CV_8UC3);
  int means[3] = {104, 117, 123};
  for (int ch = 0; ch < 3; ch++) {
    for (int r = 0; r < top->shape(2); r++) {
      for (int c = 0; c < top->shape(3); c++) {
        vismap.at<Vec3b>(r, c)[ch] = static_cast<uchar>(means[ch] +
            *(img_data + ((img_index * 3 + ch)*top->shape(2) + r )*top->shape(3) + c));
      }
    }
  }
	for (int r = 1; r < top->shape(2)-1; r++) {
		for (int c = 1; c < top->shape(3)-1; c++) {
			int index = (img_index*top->shape(2) + r)*top->shape(3) + c;
			Dtype value = data[index];
      bool boundary = false;
      for (int i = -1; i <= 0; i++) {
        for (int j = -1; j <= 0; j++) {
          if (i == 0 && j == 0) {
          } else {
			      int tmp_index = (img_index*top->shape(2) + (r+i))
              *top->shape(3) + (c+j);
            Dtype val2 = data[tmp_index];
            if (val2 != value) {
              boundary = true;
            }
          }
        }
      }
      if (boundary) {
			  Vec3b color(0,0,255);
			  vismap.at<Vec3b>(r, c) = color;
      }
		}
	}
	imshow("ahahaha2", vismap);
	waitKey(0);
}

// ---------------------------------------------------
// 
//  Caffe forward implementation
//
// ---------------------------------------------------
template<typename Dtype>
void GenSlicinfoLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
	boost::chrono::system_clock::time_point start;
	start = boost::chrono::system_clock::now();
    // Convert Image space from BGR to XYZ
    Cvt_Img_Space(bottom[0], &cvt_img_, COLOR_CHAN_);
    Init_Cluster_Centers();
    Find_Center_Association(top[0]);

	//Visualize(top[0], map_size_.x, map_size_.y);
    for (int i = 0; i < 3; i++) { // TODO: iter_num hardcoded
        Update_Cluster_Center(top[0]);
        Find_Center_Association(top[0]);
    }
    Enforce_Connectivity(top[0], top[1]);// <- need to solve slic index problem.
	//Visualize(top[0], map_size_.x, map_size_.y);
	//Visualize22(top[0], map_size_.x, map_size_.y, bottom[0]);
    // synchronize?
    // cudaThreadSynchronize();
	boost::chrono::duration<double> sec = boost::chrono::system_clock::now() - start;
	//LOG(INFO) << "time : " << sec.count() << "s";
}
INSTANTIATE_LAYER_GPU_FUNCS(GenSlicinfoLayer);
}
