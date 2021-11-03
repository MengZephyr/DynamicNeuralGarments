#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

struct R_Mesh
{
	std::vector<cv::Vec3f> verts;
	std::vector<cv::Vec3f> normals;
	std::vector<cv::Vec3i> faceInds;

	int numV() { return verts.size(); }
	int numF() { return faceInds.size();}

	void scaleMesh(float sx, float sy, float sz)
	{
		for (int v = 0; v < numV(); v++)
		{
			cv::Vec3f pos = verts[v];
			pos = cv::Vec3f(pos[0] * sx, pos[1] * sy, pos[2] * sz);
			verts[v] = pos;
		}
	}
};

struct CameraModel
{
	CameraModel(const std::vector<cv::Vec4f>& matArray, int ImgH, int ImgW)
	{
		cv::Matx33f& R = this->RotMat;
		cv::Vec3f& T = this->TransVec;
		for (int pi = 0; pi < 3; pi++)
		{
			R(pi, 0) = matArray[pi][0];
			R(pi, 1) = matArray[pi][1];
			R(pi, 2) = matArray[pi][2];
			T[pi] = matArray[pi][3];
		}
		this->rx = ImgW;
		this->ry = ImgH;
		this->cx = 0.5 * (this->rx - 1.);
		this->cy = 0.5 * (this->ry - 1.);
		this->fx = -matArray[4][0] * 0.5 * (this->rx - 1.);
		this->fy = matArray[5][1] * 0.5 * (this->ry - 1.);
		//printf("%f, %f\n", fx, fy);
	}

	cv::Matx33f RotMat;
	cv::Vec3f   TransVec;
	float fx, fy, cx, cy;
	int rx, ry;

	cv::Vec3f proj(const cv::Vec3f& p, bool ifLeft=true)
	{
		cv::Vec3f pos = ifLeft ? cv::Vec3f(p[0], -p[2], p[1]) : cv::Vec3f(p[0], p[1], p[2]);
		pos = RotMat * pos + TransVec;
		pos = cv::Vec3f(fx * pos[0] / pos[2] + cx, fy * pos[1] / pos[2] + cy, pos[2]);
		return pos;
	}

	cv::Vec3f orthProj(const cv::Vec3f& p, bool ifLeft = false)
	{
		float fs = 10. * 512. / 36.;
		cv::Vec3f pos = ifLeft ? cv::Vec3f(p[0], -p[2], p[1]) : cv::Vec3f(p[0], p[1], p[2]);
		pos = RotMat * pos + TransVec;
		pos = cv::Vec3f(-1.* fx * pos[0] + cx, ry - 1.* fy* pos[1] - cy, pos[2]);
		return pos;
	}

	cv::Vec3f projNorm(const cv::Vec3f& n, bool ifLeft = true)
	{
		cv::Vec3f nn = ifLeft ? cv::Vec3f(n[0], -n[2], n[1]) : cv::Vec3f(n[0], n[1], n[2]);
		nn = RotMat * n;
		nn = normalize(n);
		return nn;
	}

	std::vector<cv::Vec3f> projVertArray(const std::vector<cv::Vec3f>& verts, bool ifOri = false)
	{
		std::vector<cv::Vec3f> projVertArray(verts.size());
		for (int i = 0; i < verts.size(); i++)
			projVertArray[i] = ifOri? orthProj(verts[i]) : proj(verts[i]);
		
		return projVertArray;
	}
};