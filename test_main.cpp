#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/net.hpp>
#include <caffe/util/io.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <boost/filesystem.hpp>
#include <boost/chrono.hpp>

namespace fs = boost::filesystem;
using namespace std;
using namespace cv;
using namespace caffe;

float MEAN_BGR[3] = {104, 117, 123};

void CustomResize(const Mat& img, Mat& output, cv::Size size) {

  Mat newmap(size, CV_8UC1);
  float mul_x = img.cols / (float) (size.width);
  float mul_y = img.rows / (float) (size.height);
  for (int i = 0; i < size.height; i++) {
    for (int j = 0; j < size.width; j ++) {
      newmap.at<uchar>(i, j) = static_cast<uchar>(img.at<float>((int)(i*mul_y), 
          (int)(j*mul_x)) * 255);
    }
  }
  output = newmap.clone();
}

void CustomCVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;

  for (int ch = 0; ch < datum_channels; ch++) {
	  for (int r = 0; r < datum_height; r++) {
		  for (int c = 0; c < datum_width; c++) {
			  unsigned char mat_val = cv_img.at<Vec3b>(r, c)[ch];
			  datum->add_float_data(static_cast<float>(mat_val) - MEAN_BGR[ch]);
		  }
	  }
  }
}

/*
 * argv[1]: prototxt path
 * argv[2]: binary model path
 * argv[3]: image directory path
 * argv[4]: save directory path
 */
int main(int argc, char** argv) {
  const int fixed_size = 324;
	string net_proto_path(argv[1]);
	string net_binary_path(argv[2]);
  double total_count = 0;

	std::unique_ptr<Net<float>> caffe_test_net;
	Caffe::set_mode(Caffe::Brew::GPU);
	Caffe::SetDevice(0);
	caffe_test_net.reset(new Net<float>(net_proto_path, caffe::TEST));
	caffe_test_net->CopyTrainedLayersFrom(net_binary_path);

	string image_dirpath = argv[3];
  string save_dirpath = argv[4];
	// collect filename
	fs::directory_iterator imgdir_iter = fs::directory_iterator(image_dirpath);	
	fs::directory_iterator end_iter;
	vector<string> img_filenames;
	while (imgdir_iter != end_iter) {
        string imgname = imgdir_iter->path().string();
        img_filenames.push_back(imgname);
        imgdir_iter++;
  }
	std::sort(img_filenames.begin(), img_filenames.end());

	int total_num = img_filenames.size();

	for (int imgidx = 0; imgidx < total_num; imgidx++) {

		string img_path = img_filenames[imgidx];
		Mat image =	cv::imread(img_path);
		boost::chrono::system_clock::time_point start;
		start = boost::chrono::system_clock::now();	

    fs::path savepath(save_dirpath);
    fs::path img_path_p(img_path);

		cv::Size original_size = cv::Size(image.cols, image.rows);
		cv::resize(image, image, cv::Size(fixed_size, fixed_size));
		vector<Datum> input_datum;
		input_datum.emplace_back();
		CustomCVMatToDatum(image, &(input_datum[0]));
		
		boost::shared_ptr<MemoryDataLayer<float>> input_layer =
			boost::static_pointer_cast<MemoryDataLayer<float>>(
					caffe_test_net->layer_by_name("data"));

		input_layer->AddDatumVector(input_datum);

		vector<Blob<float>*> empty_vec;

		caffe_test_net->Forward(empty_vec);

		const boost::shared_ptr<Blob<float>> slic_blob =
			caffe_test_net->blob_by_name("slic");
      
		const boost::shared_ptr<Blob<float>> score_blob =
			caffe_test_net->blob_by_name("score");

    const float* score_ptr = score_blob->cpu_data();

		Mat result(fixed_size, fixed_size, CV_8UC1);

    const float* slic_ptr = slic_blob->cpu_data();
		for (int i = 0; i < fixed_size; i++) {
			for (int j = 0; j < fixed_size; j++) {
				const float index = slic_ptr[i*fixed_size + j];
        const float score = score_ptr[(int)(index)];
			  result.at<uchar>(i, j) = static_cast<uchar>(score*255);
			}
		}


		resize(result, result, original_size);
	  boost::chrono::duration<double> sec2 = boost::chrono::system_clock::now() - start;
		cout << "time : " << sec2.count() << "s" << endl;
    total_count += (double)sec2.count();

    string outimg = img_path_p.filename().string();
    outimg.replace(outimg.end()-4, outimg.end(), "_ELD.png");
    fs::path filename(outimg);
    fs::path outputpath = savepath / outimg;
    imwrite(outputpath.string(), result);
        
		cout << outputpath << endl;
	}

  cout <<  "AVG time : " << total_count / total_num << "s" << endl;

	return 0;
}
