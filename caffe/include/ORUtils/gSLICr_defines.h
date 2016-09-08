// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once

#include "ORUtils/Vector.h"
#include "ORUtils/MathUtils.h"

namespace gSLICr
{
	//------------------------------------------------------
	// 
	// math defines
	//
	//------------------------------------------------------

	typedef unsigned char uchar;
	typedef unsigned short ushort;
	typedef unsigned int uint;
	typedef unsigned long ulong;

    typedef class ORUtils::Vector2<short> Vector2s;
	typedef class ORUtils::Vector2<int> Vector2i;
	typedef class ORUtils::Vector2<float> Vector2f;
	typedef class ORUtils::Vector2<double> Vector2d;

	typedef class ORUtils::Vector3<short> Vector3s;
	typedef class ORUtils::Vector3<double> Vector3d;
	typedef class ORUtils::Vector3<int> Vector3i;
	typedef class ORUtils::Vector3<uint> Vector3ui;
	typedef class ORUtils::Vector3<uchar> Vector3u;
	typedef class ORUtils::Vector3<float> Vector3f;

	typedef class ORUtils::Vector4<float> Vector4f;
	typedef class ORUtils::Vector4<int> Vector4i;
	typedef class ORUtils::Vector4<short> Vector4s;
	typedef class ORUtils::Vector4<uchar> Vector4u;

	//------------------------------------------------------
	// 
	// Other defines
	//
	//------------------------------------------------------

	typedef enum 
	{
		CIELAB = 0,
		XYZ,
		RGB
	} COLOR_SPACE;

	typedef enum
	{
		GIVEN_NUM = 0,
		GIVEN_SIZE

	} SEG_METHOD;

  struct spixel_info
  {
      Vector2f center;
      Vector4f color_info;
      int id;
      int no_pixels;
  };

  struct Handcrafted
  {
    float color_mean[9];
    float color_histogram[9][8];
    float gabor_val[24];
    float max_gabor_val;
    float center_location[2];
  };
}
