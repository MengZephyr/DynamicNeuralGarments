#pragma once
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

std::vector<cv::Vec4f> readMatrixFile(std::string fName, bool ifColor=false)
{
	std::vector<cv::Vec4f> mmArray;
	std::ifstream vfs(fName);
	if (!vfs.is_open())
	{
		printf("Fail to read matrix file.\n");
		return std::vector<cv::Vec4f>();
	}
	std::string line;
	while (getline(vfs, line))
	{
		if (line.empty())
			continue;
		stringstream ss;
		string cmd;
		ss << line;
		if (mmArray.size() < 9)
		{
			cv::Vec4f m;
			if(ss >> m[0] >> m[1] >> m[2] >> m[3])
				mmArray.push_back(m);
		}
		else if (ifColor)
		{
			cv::Vec4f m(0., 0., 0., 0.);
			if(ss >> m[0] >> m[1] >> m[2])
				mmArray.push_back(m);
		}
	}
	return mmArray;
}

std::vector<cv::Vec3f> readJointsFile(std::string fName)
{
	std::ifstream jfs(fName);
	if (!jfs.is_open())
	{
		printf("Fail to read matrix file.\n");
		return std::vector<cv::Vec3f>();
	}
	std::string line;
	getline(jfs, line);
	stringstream ss;
	ss << line;
	int numJ = 0;
	ss >> numJ;
	std::vector<cv::Vec3f> jointArray(numJ);
	for (int j = 0; j < numJ; j++)
	{
		std::string line;
		getline(jfs, line);
		stringstream ss;
		ss << line;
		ss >> jointArray[j][0] >> jointArray[j][1] >> jointArray[j][2];
	} // end for j
	return jointArray;
}

void readPly(std::string fName, std::vector<cv::Vec3f>& verts, std::vector<cv::Vec3i>& faces)
{
	ifstream plyStream(fName);
	if (!plyStream.is_open())
	{
		printf("Fail to read Ply file.\n");
		return;
	}
	int numV = 0;
	int numF = 0;
	string line;
	while (getline(plyStream, line))
	{
		stringstream ss;
		string cmd;
		ss << line;
		ss >> cmd;
		if (cmd == "end_header")
			break;
		if (cmd == "element")
		{
			ss >> cmd;
			if (cmd == "vertex")
				ss >> numV;
			if (cmd == "face")
				ss >> numF;
		}
	}
	verts.clear();
	faces.clear();
	int count = 0;
	while (getline(plyStream, line))
	{
		stringstream ss;
		ss << line;
		cv::Vec3f v;
		ss >> v[0] >> v[1] >> v[2];
		verts.push_back(v);
		count++;
		if (count == numV)
			break;
	}
	count = 0;
	while (getline(plyStream, line))
	{
		stringstream ss;
		ss << line;
		int n = 0;
		ss >> n;
		if (n == 3)
		{
			cv::Vec3i f;
			ss >> f[0] >> f[1] >> f[2];
			faces.push_back(f);
		}
		count++;
		if (count == numF)
			break;
	}
	plyStream.close();
}

void readObjVertArray(std::string fName, std::vector<cv::Vec3f>& vArray)
{
	ifstream InfoStream(fName);
	if (!InfoStream.is_open())
	{
		printf("Fail to read Obj file.\n");
		return;
	}
	vArray.clear();
	string line;
	while (getline(InfoStream, line))
	{
		stringstream ss;
		string cmd;
		ss << line;
		ss >> cmd;
		if (cmd == "v")
		{
			cv::Vec3f v(0., 0., 0.);
			ss >> v[0] >> v[1] >> v[2];
			vArray.push_back(v);
		}
	}
}

void readObjVNArray(std::string fName, std::vector<cv::Vec3f>& vArray, std::vector<cv::Vec3f>& nArray)
{
	ifstream InfoStream(fName);
	if (!InfoStream.is_open())
	{
		printf("Fail to read Obj file.\n");
		return;
	}
	vArray.clear();
	nArray.clear();
	string line;
	while (getline(InfoStream, line))
	{
		stringstream ss;
		string cmd;
		ss << line;
		ss >> cmd;
		if (cmd == "v")
		{
			cv::Vec3f v(0., 0., 0.);
			ss >> v[0] >> v[1] >> v[2];
			vArray.push_back(v);
		}
		if (cmd == "vn")
		{
			cv::Vec3f n(0., 0., 0.);
			ss >> n[0] >> n[1] >> n[2];
			nArray.push_back(n);
		}
	}
}

void savePlyFile(std::string fName, ::vector<cv::Vec3f> verts, std::vector<cv::Vec3i> colors, std::vector<cv::Vec3i> faceID)
{
	int numV = verts.size();
	int numF = faceID.size();
	ofstream plyStream(fName);
	plyStream << "ply" << endl
		<< "format ascii 1.0" << endl;
	plyStream << "element vertex " << numV << endl;
	plyStream << "property float x" << endl
		<< "property float y" << endl
		<< "property float z" << endl
		<< "property uchar red" << endl
		<< "property uchar green" << endl
		<< "property uchar blue" << endl
		<< "property uchar alpha" << endl;
	plyStream << "element face " << numF << endl;
	plyStream << "property list uchar int vertex_indices" << endl;
	plyStream << "end_header" << endl;
	for (int v = 0; v < numV; v++)
		plyStream << verts[v][0] << " " << verts[v][1] << " " << verts[v][2] << " "
		<< colors[v][0] << " " << colors[v][1] << " " << colors[v][2] << " " << "255" << endl;
	for (int f = 0; f < numF; f++)
		plyStream << "3 " << faceID[f][0] << " " << faceID[f][1] << " " << faceID[f][2] << endl;
	plyStream.close();
	plyStream.clear();
}

void savePixelSampleMap(std::string fName, int numLevel, int* levelH, int* levelW, const std::vector<std::vector<cv::Vec2f>>& pixel_uv,
	const std::vector<std::vector<cv::Vec3i>>& pixel_vID, const std::vector<std::vector<cv::Vec2i>>& pixelValid, bool ifColor=false, cv::Vec3f colorVec = cv::Vec3f(0., 0., 0.))
{
	ofstream MapStream(fName);
	if (ifColor)
		MapStream << colorVec[0] << " " << colorVec[1] << " " << colorVec[2] << endl;
	MapStream << numLevel << endl;
	for (int l = 0; l < numLevel; l++)
		MapStream << levelH[l] << " ";
	MapStream << endl;
	for (int l = 0; l < numLevel; l++)
		MapStream << levelW[l] << " ";
	MapStream << endl;

	for (int l = 0; l < numLevel; l++)
	{
		assert(pixelValid[l].size() == pixel_uv[l].size());
		assert(pixelValid[l].size() == pixel_vID[l].size());
		int nump = pixelValid[l].size();
		MapStream <<l << " " << nump << endl;
		for (int p = 0; p < nump; p++)
			MapStream << pixelValid[l][p][0] << " " << pixelValid[l][p][1] << " "
			<< pixel_vID[l][p][0] << " " << pixel_vID[l][p][1] << " " << pixel_vID[l][p][2] << " "
			<< pixel_uv[l][p][0] << " " << pixel_uv[l][p][1] << endl;
	} // end for l
	MapStream.close();
	MapStream.clear();
}

void saveJointFeatFile(std::string fName, std::vector<std::vector<float>>& feat)
{
	int numVert = feat.size();
	int numFeat = feat[0].size();
	ofstream wStream(fName);
	wStream << numVert << endl;
	wStream << numFeat << endl;
	for (int vi = 0; vi < numVert; vi++)
	{
		for (int ji = 0; ji < numFeat; ji++)
		{
			wStream << feat[vi][ji];
			if (ji == numFeat - 1)
				wStream << endl;
			else
				wStream << " ";
		} // end for ji
	} // end for vi
	wStream.close();
	wStream.clear();
}

void save3dVectors(std::string fName, std::vector<cv::Vec3f>& Verts)
{
	int numVert = Verts.size();
	ofstream wStream(fName);
	wStream << numVert << endl;
	for (int vi = 0; vi < numVert; vi++)
		wStream << Verts[vi][0] << " " << Verts[vi][1] << " " << Verts[vi][2] << endl;
	wStream.close();
	wStream.clear();
}

std::vector<cv::Vec3f> read3dVectors(std::string fName)
{
	std::vector<cv::Vec3f> verts;
	ifstream RStream(fName);
	int numV = 0;
	string line;
	getline(RStream, line);
	stringstream ss;
	string cmd;
	ss << line;
	ss >> numV;
	while (getline(RStream, line))
	{
		stringstream ss;
		ss << line;
		cv::Vec3f idx;
		ss >> idx[0] >> idx[1] >> idx[2];
		verts.push_back(cv::Vec3f(idx[0] - 1, idx[1] - 1, idx[2] - 1));
	}
	assert(verts.size() == numV);
	RStream.close();
	return verts;
}

void readFaceFile(std::string fName, std::vector<cv::Vec3i>& fInds)
{
	ifstream faceStream(fName);
	int numF = 0;
	string line;
	getline(faceStream, line);
	stringstream ss;
	string cmd;
	ss << line;
	ss >> numF;
	fInds.clear();
	while (getline(faceStream, line))
	{
		stringstream ss;
		ss << line;
		cv::Vec3i idx;
		ss >> idx[0] >> idx[1] >> idx[2];
		fInds.push_back(cv::Vec3i(idx[0] - 1, idx[1] - 1, idx[2] - 1));
	}
	assert(fInds.size() == numF);
	faceStream.close();
}
