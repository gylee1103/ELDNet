// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <math.h>
#include <ostream>
#include "MathUtils.h"
namespace ORUtils {
	//////////////////////////////////////////////////////////////////////////
	//						Basic Vector Structure
	//////////////////////////////////////////////////////////////////////////

	template <class T> struct Vector2_{
		union {
			struct { T x, y; }; // standard names for components
			struct { T s, t; }; // standard names for components
			struct { T width, height; };
			T v[2];     // array access
		};
	};

	template <class T> struct Vector3_{
		union {
			struct{ T x, y, z; }; // standard names for components
			struct{ T r, g, b; }; // standard names for components
			struct{ T s, t, p; }; // standard names for components
			T v[3];
		};
	};

	template <class T> struct Vector4_ {
		union {
			struct { T x, y, z, w; }; // standard names for components
			struct { T r, g, b, a; }; // standard names for components
			struct { T s, t, p, q; }; // standard names for components
			T v[4];
		};
	};

	template <class T> struct Vector6_ {
		//union {
		T v[6];
		//};
	};

	template<class T, int s> struct VectorX_
	{
		int vsize;
		T v[s];
	};

	//////////////////////////////////////////////////////////////////////////
	// Vector class with math operators: +, -, *, /, +=, -=, /=, [], ==, !=, T*(), etc.
	//////////////////////////////////////////////////////////////////////////
	template <class T> class Vector2 : public Vector2_ < T >
	{
	public:
		typedef T value_type;
		__device__ inline int size() const { return 2; }

		////////////////////////////////////////////////////////
		//  Constructors
		////////////////////////////////////////////////////////
		__device__ Vector2(){} // Default constructor
		__device__ Vector2(const T &t) { this->x = t; this->y = t; } // Scalar constructor
		__device__ Vector2(const T *tp) { this->x = tp[0]; this->y = tp[1]; } // Construct from array			            
		__device__ Vector2(const T v0, const T v1) { this->x = v0; this->y = v1; } // Construct from explicit values
		__device__ Vector2(const Vector2_<T> &v) { this->x = v.x; this->y = v.y; }// copy constructor

		__device__ explicit Vector2(const Vector3_<T> &u)  { this->x = u.x; this->y = u.y; }
		__device__ explicit Vector2(const Vector4_<T> &u)  { this->x = u.x; this->y = u.y; }

		__device__ inline Vector2<int> toInt() const {
			return Vector2<int>((int)ROUND(this->x), (int)ROUND(this->y));
		}

		__device__ inline Vector2<int> toIntFloor() const {
			return Vector2<int>((int)floor(this->x), (int)floor(this->y));
		}

		__device__ inline Vector2<unsigned char> toUChar() const {
			Vector2<int> vi = toInt(); return Vector2<unsigned char>((unsigned char)CLAMP(vi.x, 0, 255), (unsigned char)CLAMP(vi.y, 0, 255));
		}

		__device__ inline Vector2<float> toFloat() const {
			return Vector2<float>((float)this->x, (float)this->y);
		}

		__device__ const T *getValues() const { return this->v; }
		__device__ Vector2<T> &setValues(const T *rhs) { this->x = rhs[0]; this->y = rhs[1]; return *this; }

		// indexing operators
		__device__ T &operator [](int i) { return this->v[i]; }
		__device__ const T &operator [](int i) const { return this->v[i]; }

		// type-cast operators
		__device__ operator T *() { return this->v; }
		__device__ operator const T *() const { return this->v; }

		////////////////////////////////////////////////////////
		//  Math operators
		////////////////////////////////////////////////////////

		// scalar multiply assign
		__device__ friend Vector2<T> &operator *= (const Vector2<T> &lhs, T d) {
			lhs.x *= d; lhs.y *= d; return lhs;
		}

		// component-wise vector multiply assign
		__device__ friend Vector2<T> &operator *= (Vector2<T> &lhs, const Vector2<T> &rhs) {
			lhs.x *= rhs.x; lhs.y *= rhs.y; return lhs;
		}

		// scalar divide assign
		__device__ friend Vector2<T> &operator /= (Vector2<T> &lhs, T d) {
			if (d == 0) return lhs; lhs.x /= d; lhs.y /= d; return lhs;
		}

		// component-wise vector divide assign
		__device__ friend Vector2<T> &operator /= (Vector2<T> &lhs, const Vector2<T> &rhs) {
			lhs.x /= rhs.x; lhs.y /= rhs.y;	return lhs;
		}

		// component-wise vector add assign
		__device__ friend Vector2<T> &operator += (Vector2<T> &lhs, const Vector2<T> &rhs) {
			lhs.x += rhs.x; lhs.y += rhs.y;	return lhs;
		}

		// component-wise vector subtract assign
		__device__ friend Vector2<T> &operator -= (Vector2<T> &lhs, const Vector2<T> &rhs) {
			lhs.x -= rhs.x; lhs.y -= rhs.y;	return lhs;
		}

		// unary negate
		__device__ friend Vector2<T> operator - (const Vector2<T> &rhs) {
			Vector2<T> rv;	rv.x = -rhs.x; rv.y = -rhs.y; return rv;
		}

		// vector add
		__device__ friend Vector2<T> operator + (const Vector2<T> &lhs, const Vector2<T> &rhs)  {
			Vector2<T> rv(lhs); return rv += rhs;
		}

		// vector subtract
		__device__ friend Vector2<T> operator - (const Vector2<T> &lhs, const Vector2<T> &rhs) {
			Vector2<T> rv(lhs); return rv -= rhs;
		}

		// scalar multiply
		__device__ friend Vector2<T> operator * (const Vector2<T> &lhs, T rhs) {
			Vector2<T> rv(lhs); return rv *= rhs;
		}

		// scalar multiply
		__device__ friend Vector2<T> operator * (T lhs, const Vector2<T> &rhs) {
			Vector2<T> rv(lhs); return rv *= rhs;
		}

		// vector component-wise multiply
		__device__ friend Vector2<T> operator * (const Vector2<T> &lhs, const Vector2<T> &rhs) {
			Vector2<T> rv(lhs); return rv *= rhs;
		}

		// scalar multiply
		__device__ friend Vector2<T> operator / (const Vector2<T> &lhs, T rhs) {
			Vector2<T> rv(lhs); return rv /= rhs;
		}

		// vector component-wise multiply
		__device__ friend Vector2<T> operator / (const Vector2<T> &lhs, const Vector2<T> &rhs) {
			Vector2<T> rv(lhs); return rv /= rhs;
		}

		////////////////////////////////////////////////////////
		//  Comparison operators
		////////////////////////////////////////////////////////

		// equality
		__device__ friend bool operator == (const Vector2<T> &lhs, const Vector2<T> &rhs) {
			return (lhs.x == rhs.x) && (lhs.y == rhs.y);
		}

		// inequality
		__device__ friend bool operator != (const Vector2<T> &lhs, const Vector2<T> &rhs) {
			return (lhs.x != rhs.x) || (lhs.y != rhs.y);
		}

		friend std::ostream& operator<<(std::ostream& os, const Vector2<T>& dt){
			os << dt.x << ", " << dt.y;
			return os;
		}
	};

	template <class T> class Vector3 : public Vector3_ < T >
	{
	public:
		typedef T value_type;
		__device__ inline int size() const { return 3; }

		////////////////////////////////////////////////////////
		//  Constructors
		////////////////////////////////////////////////////////
		__device__ Vector3(){} // Default constructor
		__device__ Vector3(const T &t)	{ this->x = t; this->y = t; this->z = t; } // Scalar constructor
		__device__ Vector3(const T *tp) { this->x = tp[0]; this->y = tp[1]; this->z = tp[2]; } // Construct from array
		__device__ Vector3(const T v0, const T v1, const T v2) { this->x = v0; this->y = v1; this->z = v2; } // Construct from explicit values
		__device__ explicit Vector3(const Vector4_<T> &u)	{ this->x = u.x; this->y = u.y; this->z = u.z; }
		__device__ explicit Vector3(const Vector2_<T> &u, T v0) { this->x = u.x; this->y = u.y; this->z = v0; }

		__device__ inline Vector3<int> toIntRound() const {
			return Vector3<int>((int)ROUND(this->x), (int)ROUND(this->y), (int)ROUND(this->z));
		}

		__device__ inline Vector3<int> toInt() const {
			return Vector3<int>((int)(this->x), (int)(this->y), (int)(this->z));
		}

		__device__ inline Vector3<int> toInt(Vector3<float> &residual) const {
			Vector3<int> intRound = toInt();
			residual = Vector3<float>(this->x - intRound.x, this->y - intRound.y, this->z - intRound.z);
			return intRound;
		}

		__device__ inline Vector3<short> toShortRound() const {
			return Vector3<short>((short)ROUND(this->x), (short)ROUND(this->y), (short)ROUND(this->z));
		}

		__device__ inline Vector3<short> toShortFloor() const {
			return Vector3<short>((short)floor(this->x), (short)floor(this->y), (short)floor(this->z));
		}

		__device__ inline Vector3<int> toIntFloor() const {
			return Vector3<int>((int)floor(this->x), (int)floor(this->y), (int)floor(this->z));
		}

		__device__ inline Vector3<int> toIntFloor(Vector3<float> &residual) const {
			Vector3<float> intFloor(floor(this->x), floor(this->y), floor(this->z));
			residual = *this - intFloor;
			return Vector3<int>((int)intFloor.x, (int)intFloor.y, (int)intFloor.z);
		}

		__device__ inline Vector3<unsigned char> toUChar() const {
			Vector3<int> vi = toIntRound(); return Vector3<unsigned char>((unsigned char)CLAMP(vi.x, 0, 255), (unsigned char)CLAMP(vi.y, 0, 255), (unsigned char)CLAMP(vi.z, 0, 255));
		}

		__device__ inline Vector3<float> toFloat() const {
			return Vector3<float>((float)this->x, (float)this->y, (float)this->z);
		}

		__device__ inline Vector3<float> normalised() const {
			float norm = 1.0f / sqrt((float)(this->x * this->x + this->y * this->y + this->z * this->z));
			return Vector3<float>((float)this->x * norm, (float)this->y * norm, (float)this->z * norm);
		}

		__device__ const T *getValues() const	{ return this->v; }
		__device__ Vector3<T> &setValues(const T *rhs) { this->x = rhs[0]; this->y = rhs[1]; this->z = rhs[2]; return *this; }

		// indexing operators
		__device__ T &operator [](int i) { return this->v[i]; }
		__device__ const T &operator [](int i) const { return this->v[i]; }

		// type-cast operators
		__device__ operator T *()	{ return this->v; }
		__device__ operator const T *() const { return this->v; }

		////////////////////////////////////////////////////////
		//  Math operators
		////////////////////////////////////////////////////////

		// scalar multiply assign
		__device__ friend Vector3<T> &operator *= (Vector3<T> &lhs, T d)	{
			lhs.x *= d; lhs.y *= d; lhs.z *= d; return lhs;
		}

		// component-wise vector multiply assign
		__device__ friend Vector3<T> &operator *= (Vector3<T> &lhs, const Vector3<T> &rhs) {
			lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; return lhs;
		}

		// scalar divide assign
		__device__ friend Vector3<T> &operator /= (Vector3<T> &lhs, T d) {
			lhs.x /= d; lhs.y /= d; lhs.z /= d; return lhs;
		}

		// component-wise vector divide assign
		__device__ friend Vector3<T> &operator /= (Vector3<T> &lhs, const Vector3<T> &rhs)	{
			lhs.x /= rhs.x; lhs.y /= rhs.y; lhs.z /= rhs.z; return lhs;
		}

		// component-wise vector add assign
		__device__ friend Vector3<T> &operator += (Vector3<T> &lhs, const Vector3<T> &rhs)	{
			lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs;
		}

		// component-wise vector subtract assign
		__device__ friend Vector3<T> &operator -= (Vector3<T> &lhs, const Vector3<T> &rhs) {
			lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs;
		}

		// unary negate
		__device__ friend Vector3<T> operator - (const Vector3<T> &rhs)	{
			Vector3<T> rv; rv.x = -rhs.x; rv.y = -rhs.y; rv.z = -rhs.z; return rv;
		}

		// vector add
		__device__ friend Vector3<T> operator + (const Vector3<T> &lhs, const Vector3<T> &rhs){
			Vector3<T> rv(lhs); return rv += rhs;
		}

		// vector subtract
		__device__ friend Vector3<T> operator - (const Vector3<T> &lhs, const Vector3<T> &rhs){
			Vector3<T> rv(lhs); return rv -= rhs;
		}

		// scalar multiply
		__device__ friend Vector3<T> operator * (const Vector3<T> &lhs, T rhs) {
			Vector3<T> rv(lhs); return rv *= rhs;
		}

		// scalar multiply
		__device__ friend Vector3<T> operator * (T lhs, const Vector3<T> &rhs) {
			Vector3<T> rv(lhs); return rv *= rhs;
		}

		// vector component-wise multiply
		__device__ friend Vector3<T> operator * (const Vector3<T> &lhs, const Vector3<T> &rhs)	{
			Vector3<T> rv(lhs); return rv *= rhs;
		}

		// scalar multiply
		__device__ friend Vector3<T> operator / (const Vector3<T> &lhs, T rhs) {
			Vector3<T> rv(lhs); return rv /= rhs;
		}

		// vector component-wise multiply
		__device__ friend Vector3<T> operator / (const Vector3<T> &lhs, const Vector3<T> &rhs) {
			Vector3<T> rv(lhs); return rv /= rhs;
		}

		////////////////////////////////////////////////////////
		//  Comparison operators
		////////////////////////////////////////////////////////

		// inequality
		__device__ friend bool operator != (const Vector3<T> &lhs, const Vector3<T> &rhs) {
			return (lhs.x != rhs.x) || (lhs.y != rhs.y) || (lhs.z != rhs.z);
		}

		////////////////////////////////////////////////////////////////////////////////
		// dimension specific operations
		////////////////////////////////////////////////////////////////////////////////

		// cross product
		__device__ friend Vector3<T> cross(const Vector3<T> &lhs, const Vector3<T> &rhs) {
			Vector3<T> r;
			r.x = lhs.y * rhs.z - lhs.z * rhs.y;
			r.y = lhs.z * rhs.x - lhs.x * rhs.z;
			r.z = lhs.x * rhs.y - lhs.y * rhs.x;
			return r;
		}

		friend std::ostream& operator<<(std::ostream& os, const Vector3<T>& dt){
			os << dt.x << ", " << dt.y << ", " << dt.z;
			return os;
		}
	};

	////////////////////////////////////////////////////////
	//  Non-member comparison operators
	////////////////////////////////////////////////////////

	// equality
	template <typename T1, typename T2> __device__ inline bool operator == (const Vector3<T1> &lhs, const Vector3<T2> &rhs){
		return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
	}

	template <class T> class Vector4 : public Vector4_ < T >
	{
	public:
		typedef T value_type;
		__device__ inline int size() const { return 4; }

		////////////////////////////////////////////////////////
		//  Constructors
		////////////////////////////////////////////////////////

		__device__ Vector4() {} // Default constructor
		__device__ Vector4(const T &t) { this->x = t; this->y = t; this->z = t; this->w = t; } //Scalar constructor
		__device__ Vector4(const T *tp) { this->x = tp[0]; this->y = tp[1]; this->z = tp[2]; this->w = tp[3]; } // Construct from array
		__device__ Vector4(const T v0, const T v1, const T v2, const T v3) { this->x = v0; this->y = v1; this->z = v2; this->w = v3; } // Construct from explicit values
		__device__ explicit Vector4(const Vector3_<T> &u, T v0) { this->x = u.x; this->y = u.y; this->z = u.z; this->w = v0; }
		__device__ explicit Vector4(const Vector2_<T> &u, T v0, T v1) { this->x = u.x; this->y = u.y; this->z = v0; this->w = v1; }

		__device__ inline Vector4<int> toIntRound() const {
			return Vector4<int>((int)ROUND(this->x), (int)ROUND(this->y), (int)ROUND(this->z), (int)ROUND(this->w));
		}

		__device__ inline Vector4<unsigned char> toUChar() const {
			Vector4<int> vi = toIntRound(); return Vector4<unsigned char>((unsigned char)CLAMP(vi.x, 0, 255), (unsigned char)CLAMP(vi.y, 0, 255), (unsigned char)CLAMP(vi.z, 0, 255), (unsigned char)CLAMP(vi.w, 0, 255));
		}

		__device__ inline Vector4<float> toFloat() const {
			return Vector4<float>((float)this->x, (float)this->y, (float)this->z, (float)this->w);
		}

		__device__ inline Vector4<T> homogeneousCoordinatesNormalize() const {
			return (this->w <= 0) ? *this : Vector4<T>(this->x / this->w, this->y / this->w, this->z / this->w, 1);
		}

		__device__ inline Vector3<T> toVector3() const {
			return Vector3<T>(this->x, this->y, this->z);
		}

		__device__ const T *getValues() const { return this->v; }
		__device__ Vector4<T> &setValues(const T *rhs) { this->x = rhs[0]; this->y = rhs[1]; this->z = rhs[2]; this->w = rhs[3]; return *this; }

		// indexing operators
		__device__ T &operator [](int i) { return this->v[i]; }
		__device__ const T &operator [](int i) const { return this->v[i]; }

		// type-cast operators
		__device__ operator T *() { return this->v; }
		__device__ operator const T *() const { return this->v; }

		////////////////////////////////////////////////////////
		//  Math operators
		////////////////////////////////////////////////////////

		// scalar multiply assign
		__device__ friend Vector4<T> &operator *= (Vector4<T> &lhs, T d) {
			lhs.x *= d; lhs.y *= d; lhs.z *= d; lhs.w *= d; return lhs;
		}

		// component-wise vector multiply assign
		__device__ friend Vector4<T> &operator *= (Vector4<T> &lhs, const Vector4<T> &rhs) {
			lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; lhs.w *= rhs.w; return lhs;
		}

		// scalar divide assign
		__device__ friend Vector4<T> &operator /= (Vector4<T> &lhs, T d){
			lhs.x /= d; lhs.y /= d; lhs.z /= d; lhs.w /= d; return lhs;
		}

		// component-wise vector divide assign
		__device__ friend Vector4<T> &operator /= (Vector4<T> &lhs, const Vector4<T> &rhs) {
			lhs.x /= rhs.x; lhs.y /= rhs.y; lhs.z /= rhs.z; lhs.w /= rhs.w; return lhs;
		}

		// component-wise vector add assign
		__device__ friend Vector4<T> &operator += (Vector4<T> &lhs, const Vector4<T> &rhs)	{
			lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; lhs.w += rhs.w; return lhs;
		}

		// component-wise vector subtract assign
		__device__ friend Vector4<T> &operator -= (Vector4<T> &lhs, const Vector4<T> &rhs)	{
			lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; lhs.w -= rhs.w; return lhs;
		}

		// unary negate
		__device__ friend Vector4<T> operator - (const Vector4<T> &rhs)	{
			Vector4<T> rv; rv.x = -rhs.x; rv.y = -rhs.y; rv.z = -rhs.z; rv.w = -rhs.w; return rv;
		}

		// vector add
		__device__ friend Vector4<T> operator + (const Vector4<T> &lhs, const Vector4<T> &rhs) {
			Vector4<T> rv(lhs); return rv += rhs;
		}

		// vector subtract
		__device__ friend Vector4<T> operator - (const Vector4<T> &lhs, const Vector4<T> &rhs) {
			Vector4<T> rv(lhs); return rv -= rhs;
		}

		// scalar multiply
		__device__ friend Vector4<T> operator * (const Vector4<T> &lhs, T rhs) {
			Vector4<T> rv(lhs); return rv *= rhs;
		}

		// scalar multiply
		__device__ friend Vector4<T> operator * (T lhs, const Vector4<T> &rhs) {
			Vector4<T> rv(lhs); return rv *= rhs;
		}

		// vector component-wise multiply
		__device__ friend Vector4<T> operator * (const Vector4<T> &lhs, const Vector4<T> &rhs) {
			Vector4<T> rv(lhs); return rv *= rhs;
		}

		// scalar divide
		__device__ friend Vector4<T> operator / (const Vector4<T> &lhs, T rhs) {
			Vector4<T> rv(lhs); return rv /= rhs;
		}

		// vector component-wise divide
		__device__ friend Vector4<T> operator / (const Vector4<T> &lhs, const Vector4<T> &rhs) {
			Vector4<T> rv(lhs); return rv /= rhs;
		}

		////////////////////////////////////////////////////////
		//  Comparison operators
		////////////////////////////////////////////////////////

		// equality
		__device__ friend bool operator == (const Vector4<T> &lhs, const Vector4<T> &rhs) {
			return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z) && (lhs.w == rhs.w);
		}

		// inequality
		__device__ friend bool operator != (const Vector4<T> &lhs, const Vector4<T> &rhs) {
			return (lhs.x != rhs.x) || (lhs.y != rhs.y) || (lhs.z != rhs.z) || (lhs.w != rhs.w);
		}

		friend std::ostream& operator<<(std::ostream& os, const Vector4<T>& dt){
			os << dt.x << ", " << dt.y << ", " << dt.z << ", " << dt.w;
			return os;
		}
	};

	template <class T> class Vector6 : public Vector6_ < T >
	{
	public:
		typedef T value_type;
		__device__ inline int size() const { return 6; }

		////////////////////////////////////////////////////////
		//  Constructors
		////////////////////////////////////////////////////////

		__device__ Vector6() {} // Default constructor
		__device__ Vector6(const T &t) { this->v[0] = t; this->v[1] = t; this->v[2] = t; this->v[3] = t; this->v[4] = t; this->v[5] = t; } //Scalar constructor
		__device__ Vector6(const T *tp) { this->v[0] = tp[0]; this->v[1] = tp[1]; this->v[2] = tp[2]; this->v[3] = tp[3]; this->v[4] = tp[4]; this->v[5] = tp[5]; } // Construct from array
		__device__ Vector6(const T v0, const T v1, const T v2, const T v3, const T v4, const T v5) { this->v[0] = v0; this->v[1] = v1; this->v[2] = v2; this->v[3] = v3; this->v[4] = v4; this->v[5] = v5; } // Construct from explicit values
		__device__ explicit Vector6(const Vector4_<T> &u, T v0, T v1) { this->v[0] = u.x; this->v[1] = u.y; this->v[2] = u.z; this->v[3] = u.w; this->v[4] = v0; this->v[5] = v1; }
		__device__ explicit Vector6(const Vector3_<T> &u, T v0, T v1, T v2) { this->v[0] = u.x; this->v[1] = u.y; this->v[2] = u.z; this->v[3] = v0; this->v[4] = v1; this->v[5] = v2; }
		__device__ explicit Vector6(const Vector2_<T> &u, T v0, T v1, T v2, T v3) { this->v[0] = u.x; this->v[1] = u.y; this->v[2] = v0; this->v[3] = v1; this->v[4] = v2, this->v[5] = v3; }

		__device__ inline Vector6<int> toIntRound() const {
			return Vector6<int>((int)ROUND(this[0]), (int)ROUND(this[1]), (int)ROUND(this[2]), (int)ROUND(this[3]), (int)ROUND(this[4]), (int)ROUND(this[5]));
		}

		__device__ inline Vector6<unsigned char> toUChar() const {
			Vector6<int> vi = toIntRound(); return Vector6<unsigned char>((unsigned char)CLAMP(vi[0], 0, 255), (unsigned char)CLAMP(vi[1], 0, 255), (unsigned char)CLAMP(vi[2], 0, 255), (unsigned char)CLAMP(vi[3], 0, 255), (unsigned char)CLAMP(vi[4], 0, 255), (unsigned char)CLAMP(vi[5], 0, 255));
		}

		__device__ inline Vector6<float> toFloat() const {
			return Vector6<float>((float)this[0], (float)this[1], (float)this[2], (float)this[3], (float)this[4], (float)this[5]);
		}

		__device__ const T *getValues() const { return this->v; }
		__device__ Vector6<T> &setValues(const T *rhs) { this[0] = rhs[0]; this[1] = rhs[1]; this[2] = rhs[2]; this[3] = rhs[3]; this[4] = rhs[4]; this[5] = rhs[5]; return *this; }

		// indexing operators
		__device__ T &operator [](int i) { return this->v[i]; }
		__device__ const T &operator [](int i) const { return this->v[i]; }

		// type-cast operators
		__device__ operator T *() { return this->v; }
		__device__ operator const T *() const { return this->v; }

		////////////////////////////////////////////////////////
		//  Math operators
		////////////////////////////////////////////////////////

		// scalar multiply assign
		__device__ friend Vector6<T> &operator *= (Vector6<T> &lhs, T d) {
			lhs[0] *= d; lhs[1] *= d; lhs[2] *= d; lhs[3] *= d; lhs[4] *= d; lhs[5] *= d; return lhs;
		}

		// component-wise vector multiply assign
		__device__ friend Vector6<T> &operator *= (Vector6<T> &lhs, const Vector6<T> &rhs) {
			lhs[0] *= rhs[0]; lhs[1] *= rhs[1]; lhs[2] *= rhs[2]; lhs[3] *= rhs[3]; lhs[4] *= rhs[4]; lhs[5] *= rhs[5]; return lhs;
		}

		// scalar divide assign
		__device__ friend Vector6<T> &operator /= (Vector6<T> &lhs, T d){
			lhs[0] /= d; lhs[1] /= d; lhs[2] /= d; lhs[3] /= d; lhs[4] /= d; lhs[5] /= d; return lhs;
		}

		// component-wise vector divide assign
		__device__ friend Vector6<T> &operator /= (Vector6<T> &lhs, const Vector6<T> &rhs) {
			lhs[0] /= rhs[0]; lhs[1] /= rhs[1]; lhs[2] /= rhs[2]; lhs[3] /= rhs[3]; lhs[4] /= rhs[4]; lhs[5] /= rhs[5]; return lhs;
		}

		// component-wise vector add assign
		__device__ friend Vector6<T> &operator += (Vector6<T> &lhs, const Vector6<T> &rhs)	{
			lhs[0] += rhs[0]; lhs[1] += rhs[1]; lhs[2] += rhs[2]; lhs[3] += rhs[3]; lhs[4] += rhs[4]; lhs[5] += rhs[5]; return lhs;
		}

		// component-wise vector subtract assign
		__device__ friend Vector6<T> &operator -= (Vector6<T> &lhs, const Vector6<T> &rhs)	{
			lhs[0] -= rhs[0]; lhs[1] -= rhs[1]; lhs[2] -= rhs[2]; lhs[3] -= rhs[3]; lhs[4] -= rhs[4]; lhs[5] -= rhs[5];  return lhs;
		}

		// unary negate
		__device__ friend Vector6<T> operator - (const Vector6<T> &rhs)	{
			Vector6<T> rv; rv[0] = -rhs[0]; rv[1] = -rhs[1]; rv[2] = -rhs[2]; rv[3] = -rhs[3]; rv[4] = -rhs[4]; rv[5] = -rhs[5];  return rv;
		}

		// vector add
		__device__ friend Vector6<T> operator + (const Vector6<T> &lhs, const Vector6<T> &rhs) {
			Vector6<T> rv(lhs); return rv += rhs;
		}

		// vector subtract
		__device__ friend Vector6<T> operator - (const Vector6<T> &lhs, const Vector6<T> &rhs) {
			Vector6<T> rv(lhs); return rv -= rhs;
		}

		// scalar multiply
		__device__ friend Vector6<T> operator * (const Vector6<T> &lhs, T rhs) {
			Vector6<T> rv(lhs); return rv *= rhs;
		}

		// scalar multiply
		__device__ friend Vector6<T> operator * (T lhs, const Vector6<T> &rhs) {
			Vector6<T> rv(lhs); return rv *= rhs;
		}

		// vector component-wise multiply
		__device__ friend Vector6<T> operator * (const Vector6<T> &lhs, const Vector6<T> &rhs) {
			Vector6<T> rv(lhs); return rv *= rhs;
		}

		// scalar divide
		__device__ friend Vector6<T> operator / (const Vector6<T> &lhs, T rhs) {
			Vector6<T> rv(lhs); return rv /= rhs;
		}

		// vector component-wise divide
		__device__ friend Vector6<T> operator / (const Vector6<T> &lhs, const Vector6<T> &rhs) {
			Vector6<T> rv(lhs); return rv /= rhs;
		}

		////////////////////////////////////////////////////////
		//  Comparison operators
		////////////////////////////////////////////////////////

		// equality
		__device__ friend bool operator == (const Vector6<T> &lhs, const Vector6<T> &rhs) {
			return (lhs[0] == rhs[0]) && (lhs[1] == rhs[1]) && (lhs[2] == rhs[2]) && (lhs[3] == rhs[3]) && (lhs[4] == rhs[4]) && (lhs[5] == rhs[5]);
		}

		// inequality
		__device__ friend bool operator != (const Vector6<T> &lhs, const Vector6<T> &rhs) {
			return (lhs[0] != rhs[0]) || (lhs[1] != rhs[1]) || (lhs[2] != rhs[2]) || (lhs[3] != rhs[3]) || (lhs[4] != rhs[4]) || (lhs[5] != rhs[5]);
		}

		friend std::ostream& operator<<(std::ostream& os, const Vector6<T>& dt){
			os << dt[0] << ", " << dt[1] << ", " << dt[2] << ", " << dt[3] << ", " << dt[4] << ", " << dt[5];
			return os;
		}
	};


	template <class T, int s> class VectorX : public VectorX_ < T, s >
	{
	public:
		typedef T value_type;
		__device__ inline int size() const { return this->vsize; }

		////////////////////////////////////////////////////////
		//  Constructors
		////////////////////////////////////////////////////////

		__device__ VectorX() { this->vsize = s; } // Default constructor
		__device__ VectorX(const T &t) { for (int i = 0; i < s; i++) this->v[i] = t; } //Scalar constructor
		__device__ VectorX(const T *tp) { for (int i = 0; i < s; i++) this->v[i] = tp[i]; } // Construct from array

		// indexing operators
		__device__ T &operator [](int i) { return this->v[i]; }
		__device__ const T &operator [](int i) const { return this->v[i]; }


		__device__ inline VectorX<int, s> toIntRound() const {
			VectorX<int, s> retv;
			for (int i = 0; i < s; i++) retv[i] = (int)ROUND(this->v[i]);
			return retv;
		}

		__device__ inline VectorX<unsigned char, s> toUChar() const {
			VectorX<int, s> vi = toIntRound();
			VectorX<unsigned char, s> retv;
			for (int i = 0; i < s; i++) retv[i] = (unsigned char)CLAMP(vi[0], 0, 255);
			return retv;
		}

		__device__ inline VectorX<float, s> toFloat() const {
			VectorX<float, s> retv;
			for (int i = 0; i < s; i++) retv[i] = (float) this->v[i];
			return retv;
		}

		__device__ const T *getValues() const { return this->v; }
		__device__ VectorX<T, s> &setValues(const T *rhs) { for (int i = 0; i < s; i++) this->v[i] = rhs[i]; return *this; }
		__device__ void Clear(T v){
			for (int i = 0; i < s; i++)
				this->v[i] = v;
		}


		// type-cast operators
		__device__ operator T *() { return this->v; }
		__device__ operator const T *() const { return this->v; }

		////////////////////////////////////////////////////////
		//  Math operators
		////////////////////////////////////////////////////////

		// scalar multiply assign
		__device__ friend VectorX<T, s> &operator *= (VectorX<T, s> &lhs, T d) {
			for (int i = 0; i < s; i++) lhs[i] *= d; return lhs;
		}

		// component-wise vector multiply assign
		__device__ friend VectorX<T, s> &operator *= (VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			for (int i = 0; i < s; i++) lhs[i] *= rhs[i]; return lhs;
		}

		// scalar divide assign
		__device__ friend VectorX<T, s> &operator /= (VectorX<T, s> &lhs, T d){
			for (int i = 0; i < s; i++) lhs[i] /= d; return lhs;
		}

		// component-wise vector divide assign
		__device__ friend VectorX<T, s> &operator /= (VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			for (int i = 0; i < s; i++) lhs[i] /= rhs[i]; return lhs;
		}

		// component-wise vector add assign
		__device__ friend VectorX<T, s> &operator += (VectorX<T, s> &lhs, const VectorX<T, s> &rhs)	{
			for (int i = 0; i < s; i++) lhs[i] += rhs[i]; return lhs;
		}

		// component-wise vector subtract assign
		__device__ friend VectorX<T, s> &operator -= (VectorX<T, s> &lhs, const VectorX<T, s> &rhs)	{
			for (int i = 0; i < s; i++) lhs[i] -= rhs[i]; return lhs;
		}

		// unary negate
		__device__ friend VectorX<T, s> operator - (const VectorX<T, s> &rhs)	{
			VectorX<T, s> rv; for (int i = 0; i < s; i++) rv[i] = -rhs[i]; return rv;
		}

		// vector add
		__device__ friend VectorX<T, s> operator + (const VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			VectorX<T, s> rv(lhs); return rv += rhs;
		}

		// vector subtract
		__device__ friend VectorX<T, s> operator - (const VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			VectorX<T, s> rv(lhs); return rv -= rhs;
		}

		// scalar multiply
		__device__ friend VectorX<T, s> operator * (const VectorX<T, s> &lhs, T rhs) {
			VectorX<T, s> rv(lhs); return rv *= rhs;
		}

		// scalar multiply
		__device__ friend VectorX<T, s> operator * (T lhs, const VectorX<T, s> &rhs) {
			VectorX<T, s> rv(lhs); return rv *= rhs;
		}

		// vector component-wise multiply
		__device__ friend VectorX<T, s> operator * (const VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			VectorX<T, s> rv(lhs); return rv *= rhs;
		}

		// scalar divide
		__device__ friend VectorX<T, s> operator / (const VectorX<T, s> &lhs, T rhs) {
			VectorX<T, s> rv(lhs); return rv /= rhs;
		}

		// vector component-wise divide
		__device__ friend VectorX<T, s> operator / (const VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			VectorX<T, s> rv(lhs); return rv /= rhs;
		}

		////////////////////////////////////////////////////////
		//  Comparison operators
		////////////////////////////////////////////////////////

		// equality
		__device__ friend bool operator == (const VectorX<T, s> &lhs, const VectorX<T, s> &rhs) {
			for (int i = 0; i < s; i++) if (lhs[i] != rhs[i]) return false;
			return true;
		}

		// inequality
		__device__ friend bool operator != (const VectorX<T, s> &lhs, const Vector6<T> &rhs) {
			for (int i = 0; i < s; i++) if (lhs[i] != rhs[i]) return true;
			return false;
		}

		friend std::ostream& operator<<(std::ostream& os, const VectorX<T, s>& dt){
			for (int i = 0; i < s; i++) os << dt[i] << "\n";
			return os;
		}
	};

	////////////////////////////////////////////////////////////////////////////////
	// Generic vector operations
	////////////////////////////////////////////////////////////////////////////////

	template< class T> __device__ inline T sqr(const T &v) { return v*v; }

	// compute the dot product of two vectors
	template<class T> __device__ inline typename T::value_type dot(const T &lhs, const T &rhs) {
		typename T::value_type r = 0;
		for (int i = 0; i < lhs.size(); i++)
			r += lhs[i] * rhs[i];
		return r;
	}

	// return the length of the provided vector
	template< class T> __device__ inline typename T::value_type length(const T &vec) {
		return sqrt(dot(vec, vec));
	}

	// return the normalized version of the vector
	template< class T> __device__ inline T normalize(const T &vec)	{
		typename T::value_type sum = length(vec);
		return sum == 0 ? T(typename T::value_type(0)) : vec / sum;
	}

	//template< class T> __device__ inline T min(const T &lhs, const T &rhs) {
	//	return lhs <= rhs ? lhs : rhs;
	//}

	//template< class T> __device__ inline T max(const T &lhs, const T &rhs) {
	//	return lhs >= rhs ? lhs : rhs;
	//}

	//component wise min
	template< class T> __device__ inline T minV(const T &lhs, const T &rhs) {
		T rv;
		for (int i = 0; i < lhs.size(); i++)
			rv[i] = min(lhs[i], rhs[i]);
		return rv;
	}

	// component wise max
	template< class T>
	__device__ inline T maxV(const T &lhs, const T &rhs)	{
		T rv;
		for (int i = 0; i < lhs.size(); i++)
			rv[i] = max(lhs[i], rhs[i]);
		return rv;
	}
};
