#include "DataIO.h"
#include "mesh.h"
#include "rayTracer.h"
#include <ctime>

cv::Mat blurGapTexture(cv::Mat oriImg, cv::Mat mask, int numIter = 3)
{
	int iter = numIter; // enlarge iter pixels
	cv::Mat mm = mask.clone();
	cv::Mat rstImg = oriImg.clone();
	while ((iter--) > 0)
	{
		cv::Mat iterImg = rstImg.clone();
		cv::Mat iterMask = mm.clone();
		for (int y = 0; y < oriImg.rows; y++)
		{
			for (int x = 0; x < oriImg.cols; x++)
			{
				cv::Vec3f color(0., 0., 0.);
				int cc = 0;
				if (mask.at<int>(y, x) > 0)
					color = oriImg.at<cv::Vec3f>(y, x);
				else
				{
					for (int hy = -1; hy <= 1; hy++)
					{
						for (int hx = -1; hx <= 1; hx++)
						{
							int px = MAX(MIN(x + hx, oriImg.cols - 1), 0);
							int py = MAX(MIN(y + hy, oriImg.rows - 1), 0);
							if (iterMask.at<int>(py, px) > 0)
							{
								color += iterImg.at<cv::Vec3f>(py, px);
								cc += 1;
							}
						} // end for hx
					} // end for hy
					if (cc > 0)
					{
						color = color / float(cc);
						mm.at<int>(y, x) = 255;
					}
				}
				rstImg.at<cv::Vec3f>(y, x) = color;
			} // end for x
		} //end for y
	}
	return rstImg;
}

void detectFrontArm()
{
	string jointRoot = "C:/DynamicNerualGarment/data/realCase/7/template/JJ/";
	string cameraRoot = "C:/DynamicNerualGarment/data/realCase/7/template/case/";
	string saveRoot = "C:/DynamicNerualGarment/data/realCase/7/template/case/FrontArm/";
	int hipID = 4, RHandID = 8, LHandID = 12;
	int frame0 = 37;
	int frame1 = 99;
	for (int fID = frame0; fID < frame1 + 1; fID++)
	{
		char _Jbuff[8];
		std::snprintf(_Jbuff, sizeof(_Jbuff), "%07d", fID);
		std::vector<cv::Vec3f> JointArray = readJointsFile(jointRoot + string(_Jbuff) + ".txt");
		std::vector<cv::Vec4f> matArray = readMatrixFile(cameraRoot + "camera.txt");
		CameraModel vCamera(matArray, 512, 512);
		/*cv::Vec3f p_hip = vCamera.proj(JointArray[hipID], false);
		cv::Vec3f p_RHand = vCamera.proj(JointArray[RHandID], false);
		cv::Vec3f p_LHand = vCamera.proj(JointArray[LHandID], false);*/
		cv::Vec3f p_hip = vCamera.orthProj(JointArray[hipID], false);
		cv::Vec3f p_RHand = vCamera.orthProj(JointArray[RHandID], false);
		cv::Vec3f p_LHand = vCamera.orthProj(JointArray[LHandID], false);

		printf("Frame %d: ", fID);
		printf("hip: (%f, %f, %f)...", p_hip[0], p_hip[1], p_hip[2]);
		printf("right: (%f, %f, %f)...", p_RHand[0], p_RHand[1], p_RHand[2]);
		printf("left: (%f, %f, %f) \n", p_LHand[0], p_LHand[1], p_LHand[2]);

		int RFront = p_hip[2] < p_RHand[2] ? 1 : 0;
		int LFront = p_hip[2] < p_LHand[2] ? 1 : 0;

		ofstream flagStream(saveRoot + string(_Jbuff) + "_0.txt");
		flagStream << RFront << " " << LFront << endl;
		flagStream.close();
		flagStream.clear();

		cv::Mat testM = cv::Mat::zeros(512, 512, CV_8UC3);
		if (RFront > 0)
			cv::circle(testM, cv::Point(p_RHand[0], p_RHand[1]), 5, cv::Scalar(0.007 * 255, 0.158 * 255, 0.8 * 255), -1, 8);
		if (LFront > 0)
			cv::circle(testM, cv::Point(p_LHand[0], p_LHand[1]), 5, cv::Scalar(0.102 * 255, 0.8 * 255, 0.28 * 255), -1, 8);
		cv::imwrite(saveRoot + string(_Jbuff) + "_0.png", testM);

	} // end for fI
}

void MultiLevelProjMap(const std::vector<cv::Vec4f>& matArray, int numLevel,
	int* img_levelHeight, int* img_levelWidth, R_Mesh& objMesh, R_Mesh& uvMesh, std::string saveName, bool ifColor=false)
{
	std::vector<std::vector<cv::Vec2f>> Pixel_texUV(numLevel);
	std::vector<std::vector<cv::Vec3i>> Pixel_texVID(numLevel);
	std::vector<std::vector<cv::Vec2i>> Pixel_Valid(numLevel);

	//clock_t start = std::clock();
	for (int le = 0; le < numLevel; le++)
	{
		CameraModel vCamera(matArray, img_levelHeight[le], img_levelWidth[le]);
		R_Mesh vProjMesh;
		vProjMesh.verts = vCamera.projVertArray(objMesh.verts);
		vProjMesh.faceInds = objMesh.faceInds;

		//cv::Mat tMat = cv::Mat::zeros(img_levelHeight[le], img_levelWidth[le], CV_32FC1);

		RayIntersection myTracer;
		myTracer.addObj(&vProjMesh);
		for (int y = 0; y < img_levelHeight[le]; y++)
		{
			for (int x = 0; x < img_levelWidth[le]; x++)
			{
				cv::Vec3f ori(x, y, 10.);
				cv::Vec3f dir(0., 0., -1.);
				RTCHit h = myTracer.rayIntersection(ori, dir);
				int fID = h.primID;
				if (fID < 0)
					continue;
				else
				{
					//tMat.at<float>(y, x) = 255;
					cv::Vec3i uvFace = uvMesh.faceInds[fID];
					cv::Vec2f uvCoor(h.u, h.v);
					Pixel_texUV[le].push_back(uvCoor);
					Pixel_texVID[le].push_back(uvFace);
					Pixel_Valid[le].push_back(cv::Vec2i(x, y));
				}
			} // end for x
		} // end for y
		/*cv::imwrite(std::string("tt.png"), tMat);
		exit(1);*/
	} // end for le
	//printf("%.4f sec", (std::clock() - start) / (float)CLOCKS_PER_SEC);
	

	cv::Vec3f colorV(0., 0., 0.);
	if (ifColor)
		colorV = cv::Vec3f(matArray[9][0], matArray[9][1], matArray[9][2]);

	savePixelSampleMap(saveName,
		numLevel, img_levelHeight, img_levelWidth, Pixel_texUV, Pixel_texVID, Pixel_Valid, ifColor, colorV);
}

void MultiC_gen_motionSampleViewMap()
{
	//please check camera frameID
	string viewRoot = "train/";
	string objRoot = "train/";

	int img_Height[5] = { 512, 256, 128, 64, 32 };
	int img_Width[5] = { 512, 256, 128, 64, 32 };
	int numLevel = 1;

	string uvName = objRoot + "/30_uvMesh.ply";
	R_Mesh uvMesh;
	readPly(uvName, uvMesh.verts, uvMesh.faceInds);

	string geoName = objRoot + "/30_uvMesh.ply";
	R_Mesh geoMesh;
	readPly(geoName, geoMesh.verts, geoMesh.faceInds);

	int frameLow = 2;

	string meshNamePre = "coarse_garment/30_L/PD30_";

//#define ONEBEFORE
	bool ifColor = false;
	int FD = 0;
	int frame0 = 2;
	int frame1 = 850;
	int numViews = 10;
	frameLow = MAX(frameLow + FD, 0);
	for (int fID = frame0; fID < frame1 + 1; fID++)
	{
		char _fbuffer[8];
		std::snprintf(_fbuffer, sizeof(_fbuffer), "%07d", fID + FD);
		char _ffbuffer[8];
		std::snprintf(_ffbuffer, sizeof(_ffbuffer), "%d", fID);

		//--current frame
		R_Mesh objMesh;
		readObjVertArray(objRoot + meshNamePre + string(_fbuffer) + ".obj", objMesh.verts);
		objMesh.faceInds = geoMesh.faceInds;
		//--previouse frame
		char _prevfbuffer[8];
		std::snprintf(_prevfbuffer, sizeof(_prevfbuffer), "%07d", MAX(fID - 1 + FD, frameLow));
		R_Mesh pre_objMesh;
		readObjVertArray(objRoot + meshNamePre + string(_prevfbuffer) + ".obj", pre_objMesh.verts);
		pre_objMesh.faceInds = geoMesh.faceInds;

#ifdef ONEBEFORE
		//--one before previouse frame
		char _beprebuffer[8];
		std::snprintf(_beprebuffer, sizeof(_beprebuffer), "%07d", MAX(fID - 2 + FD, frameLow));
		R_Mesh be_objMesh;
		readObjVertArray(objRoot + meshNamePre + string(_beprebuffer) + ".obj", be_objMesh.verts);
		be_objMesh.faceInds = uvMesh.faceInds;
#endif // ONEBEFORE

		char camera_fbuffer[8];
		std::snprintf(camera_fbuffer, sizeof(camera_fbuffer), "%07d", fID);
		
		for (int vID = 0; vID < numViews; vID++)
		{
			char viewbuffer[8];
			std::snprintf(viewbuffer, sizeof(viewbuffer), "%d", vID);

			string vMatName = viewRoot + "cameras/" + string(camera_fbuffer) + "_" + string(viewbuffer) + "_c.txt";
			std::vector<cv::Vec4f> matArray = readMatrixFile(vMatName, ifColor);
			/*if (ifColor)
			{
				printf("%f, %f, %f, %f\n", matArray[9][0], matArray[9][1], matArray[9][2], matArray[9][3]);
				while (true);
			}*/

			//--current frame
			MultiLevelProjMap(matArray, numLevel, img_Height, img_Width, objMesh, uvMesh,
				viewRoot + "C_map_2287/" + string(_ffbuffer) + "_" + string(viewbuffer) + "_m.txt", ifColor);
			//--previouse frame
			MultiLevelProjMap(matArray, numLevel, img_Height, img_Width, pre_objMesh, uvMesh,
				viewRoot + "C_map_2287/p_" + string(_ffbuffer) + "_" + string(viewbuffer) + "_m.txt", ifColor);

#ifdef ONEBEFORE
			//--one before prev. frame
			MultiLevelProjMap(matArray, numLevel, img_Height, img_Width, be_objMesh,
				viewRoot + "/C_map_2287/bp_" + string(_ffbuffer) + "_" + string(viewbuffer) + "_m.txt", ifColor);
#endif // ONEBEFORE
			
		} // end for vID
	
	} // end for fID
}

# define CHECK 1
void gen_motionSampleViewMap()
{
	string viewRoot = "D:/models/NR/Data/ver_4/Data/case_1/V_4/";
	string objRoot = "D:/models/MD/DataModel/DressOri/case_1/Chamuse/";

	int img_Height[5] = { 512, 256, 128, 64, 32 };
	int img_Width[5] = { 512, 256, 128, 64, 32 };
	int numLevel = 5;

	string uvName = objRoot + "uv/30_uvMesh.ply";
	R_Mesh uvMesh;
	readPly(uvName, uvMesh.verts, uvMesh.faceInds);

	int frame0 = 2;
	int frame1 = 850;
	int numViews = 1;
	for (int fID = frame0; fID < frame1+1; fID++)
	{
		char _fbuffer[8];
		std::snprintf(_fbuffer, sizeof(_fbuffer), "%07d", fID);
		R_Mesh objMesh;
		readObjVertArray(objRoot + "30_L/PD30_" + string(_fbuffer) + ".obj", objMesh.verts);
		objMesh.faceInds = uvMesh.faceInds;
		char _ffbuffer[8];
		std::snprintf(_ffbuffer, sizeof(_ffbuffer), "%d", fID);
#ifdef CHECK
		string uvImgName = objRoot + "t_30_L/" + string(_fbuffer) + ".png";
		cv::Mat uvImg = cv::imread(uvImgName, cv::IMREAD_COLOR);
		uvImg.convertTo(uvImg, CV_32FC3);
		int texH = uvImg.rows;
		int texW = uvImg.cols;
		uvMesh.scaleMesh(float(texW), float(texH), 0.);
#endif // CHECK

		for (int vi = 0; vi < numViews; vi++)
		{
			string vMatName = "";
			char _vbuffer[8];
			std::snprintf(_vbuffer, sizeof(_vbuffer), "%d", vi);
			//if (numViews > 1)
				//vMatName = viewRoot + "/cameras/" + string(_ffbuffer) + "_" + string(_vbuffer) + ".txt";
			//else
				//vMatName = viewRoot + "/cameras/" + string(_ffbuffer) + ".txt";
			vMatName = viewRoot + "/cameras.txt";
			std::vector<cv::Vec4f> matArray = readMatrixFile(vMatName);
			std::vector<std::vector<cv::Vec2f>> Pixel_texUV(numLevel);
			std::vector<std::vector<cv::Vec3i>> Pixel_texVID(numLevel);
			std::vector<std::vector<cv::Vec2i>> Pixel_Valid(numLevel);

			for (int le = 0; le < numLevel; le++)
			{
				CameraModel vCamera(matArray, img_Height[le], img_Width[le]);
				R_Mesh vProjMesh;
				vProjMesh.verts = vCamera.projVertArray(objMesh.verts);
				vProjMesh.faceInds = objMesh.faceInds;

				RayIntersection myTracer;
				myTracer.addObj(&vProjMesh);
#ifdef CHECK
				cv::Mat testImg = cv::Mat::zeros(img_Height[le], img_Width[le], CV_32FC3);
#endif
				for (int y = 0; y < img_Height[le]; y++)
				{
					for (int x = 0; x < img_Width[le]; x++)
					{
						cv::Vec3f ori(x, y, 10.);
						cv::Vec3f dir(0., 0., -1.);
						RTCHit h = myTracer.rayIntersection(ori, dir);
						int fID = h.primID;
						if (fID < 0)
							continue;
						else
						{
							cv::Vec3i uvFace = uvMesh.faceInds[fID];
							cv::Vec2f uvCoor(h.u, h.v);
							Pixel_texUV[le].push_back(uvCoor);
							Pixel_texVID[le].push_back(uvFace);
							Pixel_Valid[le].push_back(cv::Vec2i(x, y));
#ifdef CHECK
							cv::Vec3f tv0 = uvMesh.verts[uvFace[0]];
							cv::Vec3f tv1 = uvMesh.verts[uvFace[1]];
							cv::Vec3f tv2 = uvMesh.verts[uvFace[2]];
							cv::Vec3f tp = (1. - h.u - h.v) * tv0 + h.u * tv1 + h.v * tv2;
							cv::Vec3f color = getInfoFromMat_3f(cv::Vec2f(tp[0], float(texH) - tp[1]), uvImg);
							testImg.at<cv::Vec3f>(y, x) = color;
#endif // CHECK

						}
					} // end for x
				} // end for y
#ifdef CHECK
				cv::imwrite("D:/models/NR/Code/test/t.png", testImg);
				printf("CHECK_DONE");
				while (1);
#endif // CHECK

			} // end for le
			if(numViews > 1)
				savePixelSampleMap(viewRoot + "/C_map_2287/" + string(_ffbuffer) + "_" + string(_vbuffer) + "_m.txt", 
					numLevel, img_Height, img_Width, Pixel_texUV, Pixel_texVID, Pixel_Valid);
			else
				savePixelSampleMap(viewRoot + "/C_map_2287/" + string(_ffbuffer) + "_m.txt",
					numLevel, img_Height, img_Width, Pixel_texUV, Pixel_texVID, Pixel_Valid);
		} // end for vi

	} // end for fID
}

void gen_vertSampleViewMap()
{
	string viewRoot = "D:/models/NR/Data/ver_2/CTest/G_2/";
	string objRoot = "D:/models/NR/Data/ver_2/";

	int img_Height[5] = { 512, 256, 128, 64, 32};
	int img_Width[5] = {512, 256, 128, 64, 32};
	int numLevel = 5;
	
	string oVertName = objRoot + "C_0000120.obj";
	R_Mesh objMesh;
	readObjVertArray(oVertName, objMesh.verts);

	string uvName = objRoot + "C_uvMesh.ply";
	R_Mesh uvMesh;
	readPly(uvName, uvMesh.verts, uvMesh.faceInds);
	objMesh.faceInds = uvMesh.faceInds; // vert Face == txt Face

#ifdef CHECK
	string uvImgName = objRoot + "0000120_C.png";
	cv::Mat uvImg = cv::imread(uvImgName, cv::IMREAD_COLOR);
	uvImg.convertTo(uvImg, CV_32FC3);
	int texH = uvImg.rows;
	int texW = uvImg.cols;
	uvMesh.scaleMesh(float(texW), float(texH), 0.);
#endif // CHECK

	int numViews = 200;
	for (int vi = 0; vi < numViews; vi++)
	{
		char _buffer[8];
		std::snprintf(_buffer, sizeof(_buffer), "%d", vi);
		string vMatName = viewRoot + "/cameras/" + string(_buffer) + ".txt";
		std::vector<cv::Vec4f> matArray = readMatrixFile(vMatName);
		std::vector<std::vector<cv::Vec2f>> Pixel_texUV(numLevel);
		std::vector<std::vector<cv::Vec3i>> Pixel_texVID(numLevel);
		std::vector<std::vector<cv::Vec2i>> Pixel_Valid(numLevel);
		for (int le = 0; le < numLevel; le++)
		{
			CameraModel vCamera(matArray, img_Height[le], img_Width[le]);
			R_Mesh vProjMesh;
			vProjMesh.verts = vCamera.projVertArray(objMesh.verts);
			vProjMesh.faceInds = objMesh.faceInds;

			RayIntersection myTracer;
			myTracer.addObj(&vProjMesh);
#ifdef CHECK
			cv::Mat testImg = cv::Mat::zeros(img_Height[le], img_Width[le], CV_32FC3);
#endif
			for (int y = 0; y < img_Height[le]; y++)
			{
				for (int x = 0; x < img_Width[le]; x++)
				{
					cv::Vec3f ori(x, y, 10.);
					cv::Vec3f dir(0., 0., -1.);
					RTCHit h = myTracer.rayIntersection(ori, dir);
					int fID = h.primID;
					if (fID < 0)
						continue;
					else
					{
						cv::Vec3i uvFace = uvMesh.faceInds[fID];
						cv::Vec2f uvCoor(h.u, h.v);
						Pixel_texUV[le].push_back(uvCoor);
						Pixel_texVID[le].push_back(uvFace);
						Pixel_Valid[le].push_back(cv::Vec2i(x, y));
#ifdef CHECK
						cv::Vec3f tv0 = uvMesh.verts[uvFace[0]];
						cv::Vec3f tv1 = uvMesh.verts[uvFace[1]];
						cv::Vec3f tv2 = uvMesh.verts[uvFace[2]];
						cv::Vec3f tp = (1. - h.u - h.v) * tv0 + h.u * tv1 + h.v * tv2;
						cv::Vec3f color = getInfoFromMat_3f(cv::Vec2f(tp[0], float(texH) - tp[1]), uvImg);
						testImg.at<cv::Vec3f>(y, x) = color;
#endif // CHECK

					}
				} // end for x
			} // end for y
#ifdef CHECK
			cv::imwrite("D:/models/NR/Code/test/t.png", testImg);
			printf("CHECK_DONE");
			while (1);
#endif // CHECK

		} // end for le
		savePixelSampleMap(viewRoot + "/C_map_1728/" + string(_buffer) + "_m.txt", numLevel, img_Height, img_Width,
			Pixel_texUV, Pixel_texVID, Pixel_Valid);
	} // end for vi
}

void gen_normViewMap()
{
	string viewRoot = "D:/models/NR/Data/ver_2/CTest/G_2/";
	string objRoot = "D:/models/NR/Data/ver_2/";

	int img_Height = 512;
	int img_Width = 512;

	string oVertName = objRoot + "C_0000120.obj";
	R_Mesh objMesh;
	readObjVertArray(oVertName, objMesh.verts);

	string uvName = objRoot + "C_uvMesh.ply";
	R_Mesh uvMesh;
	readPly(uvName, uvMesh.verts, uvMesh.faceInds);
	objMesh.faceInds = uvMesh.faceInds; // vert Face == txt Face

	string uvImgName = objRoot + "0000120_C.png";
	cv::Mat uvImg = cv::imread(uvImgName, cv::IMREAD_COLOR);
	string uvMaskName = objRoot + "Mask_C.png";
	cv::Mat uvMask = cv::imread(uvMaskName, cv::IMREAD_GRAYSCALE);
	uvImg.convertTo(uvImg, CV_32FC3);
	uvMask.convertTo(uvMask, CV_32SC1);
	uvImg = blurGapTexture(uvImg, uvMask, 4);
	printf("%d, %d \n", uvImg.cols, uvImg.rows);
	int texH = uvImg.rows;
	int texW = uvImg.cols;
	uvMesh.scaleMesh(float(texW), float(texH), 0.);

	int numViews = 200;
	for (int vi = 0; vi < numViews; vi++)
	{
		char _buffer[8];
		std::snprintf(_buffer, sizeof(_buffer), "%d", vi);
		string vMatName = viewRoot + "/cameras/" + string(_buffer) + ".txt";
		std::vector<cv::Vec4f> matArray = readMatrixFile(vMatName);

		CameraModel vCamera(matArray, img_Height, img_Width);
		R_Mesh vProjMesh;
		vProjMesh.verts = vCamera.projVertArray(objMesh.verts);
		vProjMesh.faceInds = objMesh.faceInds;

		RayIntersection myTracer;
		myTracer.addObj(&vProjMesh);
		cv::Mat testImg = cv::Mat::zeros(img_Height, img_Width, CV_32FC3);
		for (int y = 0; y < img_Height; y++)
		{
			for (int x = 0; x < img_Width; x++)
			{
				cv::Vec3f ori(x, y, 10.);
				cv::Vec3f dir(0., 0., -1.);
				RTCHit h = myTracer.rayIntersection(ori, dir);
				int fID = h.primID;
				if (fID < 0)
					continue;
				else
				{
					cv::Vec3i uvFace = uvMesh.faceInds[fID];
					cv::Vec3f tv0 = uvMesh.verts[uvFace[0]];
					cv::Vec3f tv1 = uvMesh.verts[uvFace[1]];
					cv::Vec3f tv2 = uvMesh.verts[uvFace[2]];
					cv::Vec3f tp = (1. - h.u - h.v) * tv0 + h.u * tv1 + h.v * tv2;
					cv::Vec3f color = getInfoFromMat_3f(cv::Vec2f(tp[0], float(texH) - tp[1]), uvImg);
					testImg.at<cv::Vec3f>(y, x) = color;
				}
			} // end for xi
		} // end for yi
		cv::imwrite(viewRoot + "/normal_C/" + string(_buffer) + "_o.png", testImg);
	} // end for vi
}

void gen_viewAlignNormalMap()
{
	string objRoot = "D:/models/MD/DataModel/DressOri/case_5/Chamuse/";
	string dataRoot = "D:/models/NR/Data/ver_4/Data/case_5/template/";

	int img_Height = 512;
	int img_Width = 512;
	/*string vMatName = dataRoot + "cameras.txt";
	std::vector<cv::Vec4f> matArray = readMatrixFile(vMatName);
	CameraModel vCamera(matArray, img_Height, img_Width);*/

	int frame0 = 1;
	int frame1 = 585;

	string uvName = objRoot + "uv/30_uvMesh.ply";
	R_Mesh uvMesh;
	readPly(uvName, uvMesh.verts, uvMesh.faceInds);
	int numVerts = uvMesh.numV();
	std::vector<cv::Vec3f> vColors = read3dVectors(objRoot + "uv/vColors.txt");
	/*std::vector<cv::Vec3i> vC(vColors.size());
	for (int vi = 0; vi < numVerts; vi++)
		vC[vi] = cv::Vec3i(vColors[vi][2], vColors[vi][1], vColors[vi][0]);
	savePlyFile(objRoot + "uv/colorUV.ply", uvMesh.verts, vC, uvMesh.faceInds);*/
	/*std::vector<cv::Vec3f> vColors(numVerts);
	for (int vi = 0; vi < numVerts; vi++)
		vColors[vi] = cv::Vec3f(rand() % 255, rand() % 255, rand() % 255);
	save3dVectors(objRoot + "uv/vColors.txt", vColors);
	while (1);*/

	for (int fID = frame0; fID < frame1 + 1; fID++)
	{
		char _fbuffer[8];
		std::snprintf(_fbuffer, sizeof(_fbuffer), "%07d", fID);

		//string vMatName = dataRoot + "cameras/" + string(_fbuffer) + "_7_c.txt";
		string vMatName = dataRoot + "camera.txt";
		std::vector<cv::Vec4f> matArray = readMatrixFile(vMatName);
		CameraModel vCamera(matArray, img_Height, img_Width);

		R_Mesh objMesh;
		readObjVNArray(objRoot + "30_L/PD30_" + string(_fbuffer) + ".obj", objMesh.verts, objMesh.normals);
		objMesh.faceInds = uvMesh.faceInds;
		char _ffbuffer[8];
		std::snprintf(_ffbuffer, sizeof(_ffbuffer), "%d", fID);
		R_Mesh projMesh;
		projMesh.verts = vCamera.projVertArray(objMesh.verts);
		projMesh.faceInds = objMesh.faceInds;
		RayIntersection myTracer;
		myTracer.addObj(&projMesh);
		cv::Mat testImg = cv::Mat::zeros(img_Height, img_Width, CV_32FC4);
		for (int y = 0; y < img_Height; y++)
		{
			for (int x = 0; x < img_Width; x++)
			{
				testImg.at<cv::Vec4f>(y, x) = cv::Vec4f(255, 255, 255, 255);
				cv::Vec3f ori(x, y, 10.);
				cv::Vec3f dir(0., 0., -1.);
				RTCHit h = myTracer.rayIntersection(ori, dir);
				int fID = h.primID;
				if (fID < 0)
					continue;
				else
				{
					cv::Vec3i uvFace = objMesh.faceInds[fID];
					cv::Vec3f nv0 = vColors[uvFace[0]];
					cv::Vec3f nv1 = vColors[uvFace[1]];
					cv::Vec3f nv2 = vColors[uvFace[2]];
					cv::Vec3f np = (1. - h.u - h.v) * nv0 + h.u * nv1 + h.v * nv2;
					testImg.at<cv::Vec4f>(y, x) = cv::Vec4f(np[0], np[1], np[2], 200.);
					//np = vCamera.projNorm(np, false);
					//testImg.at<cv::Vec3f>(y, x) = (cv::Vec3f(np[2], np[1], np[0]) + cv::Vec3f(1., 1., 1.)) * 0.5;
				}
			} // end for x
		} // end for y
		char _fbuffer1[8];
		std::snprintf(_fbuffer1, sizeof(_fbuffer1), "%07d", fID);
		std::string savename = dataRoot + "template/" + string(_fbuffer1) + ".png";
		cv::imwrite(savename, testImg);
	} // end for fID
}

void gen_uvjointInfo()
{
	string objRoot = "C:/DynamicNerualGarment/data/TShirtPants/case_5/";
	string jointRoot = "D:/models/MD/DataModel/Motions/JJ/case_5/";
	int frame0 = 1;
	int frame1 = 586;
	int minFrame = 1;

	int DF = 0;
	int JTemW = 5;
	string namePref = "30_L/t_";
	string uvName = objRoot + "uv/30_uvMesh.ply";
	R_Mesh uvMesh;
	readPly(uvName, uvMesh.verts, uvMesh.faceInds);
	string geoName = objRoot + "uv/30_geo.ply";
	R_Mesh geoMesh;
	readPly(geoName, geoMesh.verts, geoMesh.faceInds);
	assert(uvMesh.numF() == geoMesh.numF());

	for (int fID = frame0; fID < frame1 + 1; fID++)
	{
		char _ttfbufferObj[8];
		std::snprintf(_ttfbufferObj, sizeof(_ttfbufferObj), "%06d", fID+DF);
		R_Mesh objMesh;
		readObjVertArray(objRoot + namePref + string(_ttfbufferObj) + ".obj", objMesh.verts);

		std::vector<std::vector<float>> JFeatures(objMesh.numV(), std::vector<float>());
		for (int JId = 0; JId < JTemW; JId++)
		{
			char _Jbuff[8];
			std::snprintf(_Jbuff, sizeof(_Jbuff), "%07d", MAX(fID - 2 * JId, minFrame));
			std::vector<cv::Vec3f> JointArray = readJointsFile(jointRoot + string(_Jbuff) + ".txt");
			for (int vi = 0; vi < objMesh.numV(); vi++)
			{
				cv::Vec3f vpos = objMesh.verts[vi];
				vpos = cv::Vec3f(vpos[0], -vpos[2], vpos[1]);
				for (int ji = 0; ji < JointArray.size(); ji++)
				{
					float dist = norm(JointArray[ji] - vpos);
					//JFeatures[vi].push_back(exp(-(dist * dist) / 1000.));
					JFeatures[vi].push_back(exp(-(dist * dist) / 0.1));
					//JFeatures[vi][ji] = exp(-(dist * dist)/0.1);
				}
			} // end for vi
		} // end for JId

		//std::vector<cv::Vec3i> tColorA(uvMesh.numV());
		std::vector<std::vector<float>> UVJFeatures(uvMesh.numV(), std::vector<float>());
		std::vector<int> uv_VVFlage(uvMesh.numV(), -1);
		for (int face = 0; face < uvMesh.numF(); face++)
		{
			cv::Vec3i uvFace = uvMesh.faceInds[face];
			cv::Vec3i geoFace = geoMesh.faceInds[face];
			for (int d = 0; d < 3; d++)
			{
				if (uv_VVFlage[uvFace[d]] < 0)
				{
					UVJFeatures[uvFace[d]] = JFeatures[geoFace[d]];
					uv_VVFlage[uvFace[d]] = 1;
					/*tColorA[uvFace[d]] = cv::Vec3i(UVJFeatures[uvFace[d]][13] * 255, UVJFeatures[uvFace[d]][14] * 255,
						                           UVJFeatures[uvFace[d]][18] * 255);*/
				}
			} // end for d
		} // end for face

		/*savePlyFile(objRoot + "t.ply", uvMesh.verts, tColorA, uvMesh.faceInds);
		printf("...Done.");
		while (1);*/

		char _fbufferObj[8];
		std::snprintf(_fbufferObj, sizeof(_fbufferObj), "%07d", fID);
		saveJointFeatFile(objRoot + "30_JFeat_2_10/" + string(_fbufferObj) + ".txt", UVJFeatures);

	} // end for fID
}

void gen_jointInfo()
{
	string objRoot = "C:/DynamicNerualGarment/data/TShirtPants/case_5/";
	string jointRoot = "D:/models/MD/DataModel/Motions/JJ/case_5/";
	int frame0 = 1;
	int frame1 = 585;
	int minFrame = 1;

	int JTemW = 5;
	string namePref = "30_L/PD30_";
	string uvName = objRoot + "uv/30_uvMesh.ply";
	R_Mesh uvMesh;
	readPly(uvName, uvMesh.verts, uvMesh.faceInds);
	for (int fID = frame0; fID < frame1 + 1; fID++)
	{
		char _fbufferObj[8];
		std::snprintf(_fbufferObj, sizeof(_fbufferObj), "%07d", fID);
		R_Mesh objMesh;
		readObjVertArray(objRoot + namePref + string(_fbufferObj) + ".obj", objMesh.verts);

		/*char _ttfbufferObj[8];
		std::snprintf(_ttfbufferObj, sizeof(_ttfbufferObj), "%06d", fID - 1);
		R_Mesh objMesh;
		readObjVertArray(objRoot + namePref + string(_ttfbufferObj) + ".obj", objMesh.verts);*/

		std::vector<cv::Vec3i> tColorA(objMesh.numV());

		std::vector<std::vector<float>> JFeatures(objMesh.numV(), std::vector<float>());
		for (int JId = 0; JId < JTemW; JId++)
		{
			char _Jbuff[8];
			std::snprintf(_Jbuff, sizeof(_Jbuff), "%07d", MAX(fID - 2 * JId, minFrame));
			std::vector<cv::Vec3f> JointArray = readJointsFile(jointRoot + string(_Jbuff) + ".txt");
			for (int vi = 0; vi < objMesh.numV(); vi++)
			{
				cv::Vec3f vpos = objMesh.verts[vi];
				vpos = cv::Vec3f(vpos[0], -vpos[2], vpos[1]);
				for (int ji = 0; ji < JointArray.size(); ji++)
				{
					float dist = norm(JointArray[ji] - vpos);
					//JFeatures[vi].push_back(exp(-(dist * dist) / 1000.));
					JFeatures[vi].push_back(exp(-(dist * dist) / 0.1));
					//JFeatures[vi][ji] = exp(-(dist * dist)/0.1);
				}
				tColorA[vi] = cv::Vec3i(JFeatures[vi][13] * 255, JFeatures[vi][14] * 255, JFeatures[vi][18] * 255);
			} // end for vi

		} // end for JId

		/*savePlyFile("D:/models/NR/Code/test/t.ply", objMesh.verts, tColorA, uvMesh.faceInds);
		printf("...Done.");
		while (1);*/

		saveJointFeatFile(objRoot + "30_JFeat_2_10/" + string(_fbufferObj) + ".txt", JFeatures);

	} // end for fID
}

void blankUVGen()
{
	int minx = 0, maxx =1000, miny = 0, maxy = 1000;
	int width = 1000, height = 1000;
	cv::Mat uvimg = cv::Mat::zeros(height, width, CV_32FC3);
	float LenX = maxx - minx, LenY = maxy - miny;
	for (int y = miny; y < maxy; y++)
		for (int x = minx; x < maxx; x++)
			uvimg.at<cv::Vec3f>(y, x) = cv::Vec3f(0, (float(y) - float(miny)) / LenY, (float(x) - float(minx)) / LenX);
	cv::imwrite("uv.png", uvimg * 255);
}

void perViewAlbedoDraw(CameraModel& vCamera, R_Mesh& objMesh, cv::Mat& maskImg, cv::Mat& oriImg, cv::Mat& oriGray, 
	R_Mesh& uvMesh, cv::Mat& uvImg, string saveName_a, string saveName_s)
{
	int uvH = uvImg.rows, uvW = uvImg.cols;
	int img_Height = maskImg.rows, img_Width = maskImg.cols;
	cv::Mat compImg = oriImg.clone();
	cv::Mat compGray = oriGray.clone();

	R_Mesh projMesh;
	projMesh.verts = vCamera.projVertArray(objMesh.verts);
	projMesh.faceInds = objMesh.faceInds;
	RayIntersection myTracer;
	myTracer.addObj(&projMesh);

	for (int y = 0; y < img_Height; y++)
	{
		for (int x = 0; x < img_Width; x++)
		{
			if (maskImg.at<uchar>(y, x) < 150)
				continue;

			cv::Vec3f ori(x, y, 10.);
			cv::Vec3f dir(0., 0., -1.);
			RTCHit h = myTracer.rayIntersection(ori, dir);
			int fID = h.primID;
			if (fID < 0)
				continue;
			else
			{
				cv::Vec3i uvFace = objMesh.faceInds[fID];
				cv::Vec3f uv0 = uvMesh.verts[uvFace[0]];
				cv::Vec3f uv1 = uvMesh.verts[uvFace[1]];
				cv::Vec3f uv2 = uvMesh.verts[uvFace[2]];
				cv::Vec3f myuv = (1. - h.u - h.v) * uv0 + h.u * uv1 + h.v * uv2;
				cv::Vec2f uvImgPos(myuv[0], float(uvH) - myuv[1]);
				cv::Vec3f uvColor = getInfoFromMat_3f(uvImgPos, uvImg);
				cv::Vec3f oriColor = oriImg.at<cv::Vec3f>(y, x);
				float shV = (oriColor[1] + oriColor[2]) / (uvColor[1] + uvColor[2]);
				shV = MIN(shV, 2.);
				compImg.at<cv::Vec3f>(y, x) = cv::Vec3f(0., uvColor[1], uvColor[2]);
				compGray.at<float>(y, x) = shV * 255. / 2.;
			}
		}
	}
	cv::imwrite(saveName_a, compImg);
	cv::imwrite(saveName_s, compGray);
}

//#define NEXTAS 1
//#define PREVAS 1
void albedoDraw()
{
	string viewRoot = "D:/models/NR/Data/ver_4/Data/case_1/tango/V_t/";
	string objRoot = "D:/models/MD/DataModel/DressOri/case_1/tango/";

	string uvName = objRoot + "uv/uv.ply";
	R_Mesh uvMesh;
	readPly(uvName, uvMesh.verts, uvMesh.faceInds);
	string uvImgName = objRoot + "uv.png";
	cv::Mat uvImg = cv::imread(uvImgName, cv::IMREAD_COLOR);
	uvImg.convertTo(uvImg, CV_32FC3);
	int uvH = uvImg.rows, uvW = uvImg.cols;
	uvMesh.scaleMesh(float(uvW), float(uvH), 0.);

	int img_Height = 512;
	int img_Width = 512;

	int frameLow = 2;
	int frameHigh = 850;

	int frame0 = 51, frame1 = 251;
	int numViews = 1;
	
	for (int fID = frame0; fID < frame1+1; fID++)
	{
		if (fID > frameHigh)
			break;
		//--current frame
		char _fbufferObj[8];
		std::snprintf(_fbufferObj, sizeof(_fbufferObj), "%07d", fID);
		string objName = objRoot + "10_L/PD10_" + string(_fbufferObj) + ".obj";
		R_Mesh objMesh;
		readObjVertArray(objName, objMesh.verts);
		objMesh.faceInds = uvMesh.faceInds;

#ifdef PREVAS
		//--previouse frame
		char _prevfbuffer[8];
		std::snprintf(_prevfbuffer, sizeof(_prevfbuffer), "%07d", MAX(fID - 1, frameLow));
		R_Mesh pre_objMesh;
		readObjVertArray(objRoot + "10_L/PD10_" + string(_prevfbuffer) + ".obj", pre_objMesh.verts);
		pre_objMesh.faceInds = uvMesh.faceInds;
#endif // DEBUG

#ifdef NEXTAS
		//--next frame
		char _Nextbuffer[8];
		std::snprintf(_Nextbuffer, sizeof(_Nextbuffer), "%07d", MIN(fID + 1, frameHigh));
		R_Mesh next_objMesh;
		readObjVertArray(objRoot + "10_L/PD10_" + string(_Nextbuffer) + ".obj", next_objMesh.verts);
		next_objMesh.faceInds = uvMesh.faceInds;
#endif // NEXTAS

		for (int vID = 0; vID < numViews; vID++)
		{
			char viewbuffer[8];
			std::snprintf(viewbuffer, sizeof(viewbuffer), "%d", vID);

			string vMatName = viewRoot + "camera.txt";
			//string vMatName = viewRoot + "cameras/" + string(_fbufferObj) + "_" + string(viewbuffer) + "_c.txt";
			std::vector<cv::Vec4f> matArray = readMatrixFile(vMatName);
			CameraModel vCamera(matArray, img_Height, img_Width);

			//--current frame
			cv::Mat currMask = cv::imread(viewRoot + "mask/" + string(_fbufferObj) + "_" + string(viewbuffer) + ".png", 
				cv::IMREAD_GRAYSCALE);
			cv::Mat currImg = cv::imread(viewRoot + "render_1/" + string(_fbufferObj) + "_" + string(viewbuffer) + ".png", 
				cv::IMREAD_COLOR);
			cv::Mat currGray;
			cv::cvtColor(currImg, currGray, cv::COLOR_BGR2GRAY);
			currGray.convertTo(currGray, CV_32FC1);
			currImg.convertTo(currImg, CV_32FC3);
			perViewAlbedoDraw(vCamera, objMesh, currMask, currImg, currGray, uvMesh, uvImg,
				viewRoot + "render_a/" + string(_fbufferObj) + "_" + string(viewbuffer) + ".png", 
				viewRoot + "render_s/" + string(_fbufferObj) + "_" + string(viewbuffer) + ".png");

#ifdef PREVAS
			//--prev frame 
			cv::Mat preMask = cv::imread(viewRoot + "mask/p_" + string(_fbufferObj) + "_" + string(viewbuffer) + ".png",
				cv::IMREAD_GRAYSCALE);
			cv::Mat preImg = cv::imread(viewRoot + "render_1/p_" + string(_fbufferObj) + "_" + string(viewbuffer) + ".png",
				cv::IMREAD_COLOR);
			cv::Mat preGray;
			cv::cvtColor(preImg, preGray, cv::COLOR_BGR2GRAY);
			preGray.convertTo(preGray, CV_32FC1);
			preImg.convertTo(preImg, CV_32FC3);
			perViewAlbedoDraw(vCamera, pre_objMesh, preMask, preImg, preGray, uvMesh, uvImg,
				viewRoot + "render_a/p_" + string(_fbufferObj) + "_" + string(viewbuffer) + ".png",
				viewRoot + "render_s/p_" + string(_fbufferObj) + "_" + string(viewbuffer) + ".png");
#endif // PREVAS

#ifdef NEXTAS
			//--next frame 
			cv::Mat nextMask = cv::imread(viewRoot + "mask/a_" + string(_fbufferObj) + "_" + string(viewbuffer) + ".png",
				cv::IMREAD_GRAYSCALE);
			cv::Mat nextImg = cv::imread(viewRoot + "render_1/a_" + string(_fbufferObj) + "_" + string(viewbuffer) + ".png",
				cv::IMREAD_COLOR);
			cv::Mat nextGray;
			cv::cvtColor(nextImg, nextGray, cv::COLOR_BGR2GRAY);
			nextGray.convertTo(nextGray, CV_32FC1);
			nextImg.convertTo(nextImg, CV_32FC3);
			perViewAlbedoDraw(vCamera, next_objMesh, nextMask, nextImg, nextGray, uvMesh, uvImg,
				viewRoot + "render_a/a_" + string(_fbufferObj) + "_" + string(viewbuffer) + ".png",
				viewRoot + "render_s/a_" + string(_fbufferObj) + "_" + string(viewbuffer) + ".png");
#endif // NEXTAS
		}
		
	} // end for fID
}

void shaderAbtr()
{
	string rootName = "D:/models/NR/Data/ver_4/Data/case_1/tango/V_t/";
	string oriImgName = rootName + "render_4/0000104_0.png";
	string albImgName = rootName + "render_a/0000104_0.png";
	string maskImgName = rootName + "mask/0000104_0.png";
	cv::Mat oriImg = cv::imread(oriImgName, cv::IMREAD_COLOR);
	oriImg.convertTo(oriImg, CV_32FC3);
	/*cv::Mat shaImg = cv::imread(rootName + "render_s/0000104_0.png", cv::IMREAD_GRAYSCALE);
	shaImg.convertTo(shaImg, CV_32FC1);*/
	cv::Mat albImg = cv::imread(albImgName, cv::IMREAD_COLOR);
	albImg.convertTo(albImg, CV_32FC3);
	cv::Mat maskImg = cv::imread(maskImgName, cv::IMREAD_GRAYSCALE);

	cv::Mat shaImg = oriImg.clone();
	//cv::Mat reOriImg = cv::Mat(maskImg.size(), CV_32FC3);
	int cc = 0;
	for ( int yi = 0; yi < maskImg.rows; yi++)
	{
		for (int xi = 0; xi < maskImg.cols; xi++)
		{
			if (maskImg.at<uchar>(yi, xi) > 180)
			{
				cv::Vec3f oriC = oriImg.at<cv::Vec3f>(yi, xi);
				cv::Vec3f albC = albImg.at<cv::Vec3f>(yi, xi);
				float shaV = (oriC[1] + oriC[2]) / (albC[1] + albC[2]);
				shaImg.at<cv::Vec3f>(yi, xi) = cv::Vec3f(shaV * 255. / 2., albC[1], albC[2]);
				//float shaV = shaImg.at<float>(yi, xi) * 2. / 255;
				//reOriImg.at<cv::Vec3f>(yi, xi) = shaV * albC;
				//shaImg.at<float>(yi, xi) = (oriC[0] + oriC[1] + oriC[2]) / (albC[0] + albC[1] + albC[2]);
			}
		}
	}
	cv::imwrite(rootName + "render_s/0000104_0_c.png", shaImg);
	//cv::imwrite(rootName + "render_s/0000104_0r.png", reOriImg);
}

void postTex()
{
	string rootName = "D:/models/NR/Data/ver_4/Data/case_1/tango/V_t/";
	string txtName = "D:/models/MD/DataModel/DressOri/case_1/tango/PD10_rrr.png";
	cv::Mat txtImg = cv::imread(txtName, cv::IMREAD_COLOR);
	txtImg.convertTo(txtImg, CV_32FC3);

	int frame0 = 51;
	int frame1 = 251;
	for (int fID = frame0; fID < frame1+1; fID++)
	{
		char fbuffer[8];
		std::snprintf(fbuffer, sizeof(fbuffer), "%d", fID);

		string shaImgName = rootName + "rst/netRst_1/" + string(fbuffer) + "_0.png";
		string maskImgName = rootName + "rst/netMask_1/" + string(fbuffer) + "_0.png";
		cv::Mat shaImg = cv::imread(shaImgName, cv::IMREAD_COLOR);
		shaImg.convertTo(shaImg, CV_32FC3);
		cv::Mat maskImg = cv::imread(maskImgName, cv::IMREAD_GRAYSCALE);
		maskImg.convertTo(maskImg, CV_64FC1);
		double vmax, vmin;
		cv::Point pmax, pmin;
		cv::minMaxLoc(maskImg, &vmin, &vmax, &pmin, &pmax);
		maskImg = 255 * (maskImg - vmin) / (vmax - vmin);
		//printf("min: %f, max: %f", vmin, vmax);
		//cv::imwrite(rootName + "rst/comRst/m.png", maskImg);

		cv::Mat rendImg = shaImg.clone();

		int minx = 0, maxx = 850, miny = 150, maxy = 1000;
		float LenX = maxx - minx, LenY = maxy - miny;
		std::vector<cv::Vec2i> badPixel;
		for (int yi = 0; yi < maskImg.rows; yi++)
		{
			for (int xi = 0; xi < maskImg.cols; xi++)
			{
				if (maskImg.at<double>(yi, xi) >= 150)
				{
					cv::Vec3f info = shaImg.at<cv::Vec3f>(yi, xi) / 255.;
					//float shV = info[0] * 2.;
					/*if (shV < 0.000001)
						shV = 0.01;*/
					float indy = info[1] * LenY + float(miny);
					float indx = info[2] * LenX + float(minx);
					//printf("%d, %d--> %f, %f, %f\n", xi, yi, indx, indy, shV);
					//while (1);
					indy = MAX(MIN(indy, maxy - 1), miny);
					indx = MAX(MIN(indx, maxx - 1), minx);
					cv::Vec3f color = getInfoFromMat_3f(cv::Vec2f(indx, indy), txtImg);
					if (color[0] < 0.)
					{
						badPixel.push_back(cv::Vec2i(xi, yi));
						continue;
					}
					rendImg.at<cv::Vec3f>(yi, xi) = color;
				}
			}
		}
		//printf("badP: %d\n", badPixel.size());
		cv::imwrite(rootName + "rst/comRst_2/" + string(fbuffer) + "_0.png", rendImg);
	}
	
}


int main()
{
	//gen_jointInfo();
	//gen_uvjointInfo();

	MultiC_gen_motionSampleViewMap();

	//detectFrontArm();

	//blankUVGen();
	//shaderAbtr();
	//postTex();
	//albedoDraw();
	printf("Done.\n");
	while (1);
}
