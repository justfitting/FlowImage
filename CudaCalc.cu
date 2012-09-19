#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cublas_v2.h>
#include "MatrixBSplineD.h"
#include "CommonData.h"

#include <cstdio>
#include <cstdlib>

using namespace std;

#define TESTSIZE 100000			// 测试数据所用数组的大小
#define TEST 16					// 测试时每个线程使用的数组大小

int totalMemD = 0;

/* 测试用数据结构 */

ofstream fout("cuda.txt");
cudaError_t cymError;

void callCudaThreadSynchronize()
{
	cudaThreadSynchronize();
}

/* B 样条体求值所需的矩阵 */
static __device__ float  MB2fD[2 * 2], MB30fD[3 * 3], MB31fD[3 * 3], MB32fD[3 * 3],  MB33fD[3 * 3];
static __device__ float MB40fD[4 * 4], MB41fD[4 * 4], MB42fD[4 * 4], MB43fD[4 * 4], MB44fD[4 * 4], MB45fD[4 * 4], MB46fD[4 * 4], MB47fD[4 * 4], MB48fD[4 * 4];

/* 根据阶数、控制顶点数、左端节点的编号返回相应的 B 样条矩阵（用于 B 样条体求值） */
__device__ float *matrixCaseD(int order, int ctrlPointNum, int leftIdx)
{
	if (order == 2)
		return MB2fD;
	else if (order == 3)
	{
		if (ctrlPointNum == 3)
			return MB30fD;
		else
		{
			if (leftIdx == 2)
				return MB31fD;
			else if (leftIdx == ctrlPointNum - 1)
				return MB32fD;
			else
				return MB33fD;
		}
	}
	else
	{
		if (ctrlPointNum == 4)
			return MB40fD;
		else if (ctrlPointNum == 5)
		{
			if (leftIdx == 3)
				return MB41fD;
			else
				return MB42fD;
		}
		else if (ctrlPointNum == 6)
		{
			if (leftIdx == 3)
				return MB43fD;
			else if (leftIdx == 4)
				return MB44fD;
			else
				return MB45fD;
		}
		else
		{
			if (leftIdx == 3)
				return MB43fD;
			else if (leftIdx == 4)
				return MB46fD;
			else if (leftIdx == ctrlPointNum - 2)
				return MB47fD;
			else if (leftIdx == ctrlPointNum - 1)
				return MB45fD;
			else
				return MB48fD;
		}
	}
}

static __device__ float3 ctrlPointD[15 * 15 * 15];
static __device__ float knotListD[3 * 20];							// 节点序列

/* 使用矩阵乘法求 B 样条体的值 */
__device__ float3 BSplineVolumeValueMatrixD(float u, float v, float w,
											int leftUIdx, int leftVIdx, int leftWIdx,
											int orderU, int orderV, int orderW,
											int ctrlPointNumU, int ctrlPointNumV, int ctrlPointNumW)
{
	float3 result;
	float3 tempCtrlPoint1[4];
	float3 tempCtrlPoint2[4][4];

	float *M, temp[4], mul1[4];

	float tempKnot = knotListD[leftUIdx];
	u = (u - tempKnot) / (knotListD[leftUIdx + 1] - tempKnot);
	tempKnot = knotListD[20 + leftVIdx];
	v = (v - tempKnot) / (knotListD[20 + leftVIdx + 1] - tempKnot);
	tempKnot = knotListD[40 + leftWIdx];
	w = (w - tempKnot) / (knotListD[40 + leftWIdx + 1] - tempKnot);

	// 由三维控制顶点算出二维临时控制顶点
	temp[0] = 1.0f;
	temp[1] = w;
	temp[2] = w * w;
	temp[3] = temp[2] * w;

	M = matrixCaseD(orderW, ctrlPointNumW, leftWIdx);

	for (int i = 0; i < orderW; ++i)
	{
		mul1[i] = 0.0f;
		for (int j = 0; j < orderW; ++j)
		{
			mul1[i] += temp[j] * M[j * orderW + i];
		}
	}
	for (int i = 0; i < orderU; ++i)
	{
		for (int j = 0; j < orderV; ++j)
		{
			tempCtrlPoint2[i][j].x = 0.0f;
			tempCtrlPoint2[i][j].y = 0.0f;
			tempCtrlPoint2[i][j].z = 0.0f;
			for (int k = 0; k < orderW; ++k)
			{
				float3 cp = ctrlPointD[(leftUIdx - i) * 15 * 15 + (leftVIdx - j) * 15 + leftWIdx - k];
				tempCtrlPoint2[i][j].x += cp.x * mul1[orderW - 1 - k];
				tempCtrlPoint2[i][j].y += cp.y * mul1[orderW - 1 - k];
				tempCtrlPoint2[i][j].z += cp.z * mul1[orderW - 1 - k];
			}
		}
	}

	// 由二维临时控制顶点算出一维临时控制顶点
	temp[1] = v;
	temp[2] = v * v;
	temp[3] = temp[2] * v;

	M = matrixCaseD(orderV, ctrlPointNumV, leftVIdx);

	for (int i = 0; i < orderV; ++i)
	{
		mul1[i] = 0.0;
		for (int j = 0; j < orderV; ++j)
		{
			mul1[i] += temp[j] * M[j * orderV + i];
		}
	}
	for (int i = 0; i < orderU; ++i)
	{
		tempCtrlPoint1[i].x = 0.0f;
		tempCtrlPoint1[i].y = 0.0f;
		tempCtrlPoint1[i].z = 0.0f;
		for (int j = 0; j < orderV; ++j)
		{
			tempCtrlPoint1[i].x += tempCtrlPoint2[i][j].x * mul1[orderV - 1 - j];
			tempCtrlPoint1[i].y += tempCtrlPoint2[i][j].y * mul1[orderV - 1 - j];
			tempCtrlPoint1[i].z += tempCtrlPoint2[i][j].z * mul1[orderV - 1 - j];
		}
	}

	// 由一维临时控制顶点算出结果
	temp[1] = u;
	temp[2] = u * u;
	temp[3] = temp[2] * u;

	M = matrixCaseD(orderU, ctrlPointNumU, leftUIdx);

	for (int i = 0; i < orderU; ++i)
	{
		mul1[i] = 0.0;
		for (int j = 0; j < orderU; ++j)
		{
			mul1[i] += temp[j] * M[j * orderU + i];
		}
	}
	result.x = 0.0f;
	result.y = 0.0f;
	result.z = 0.0f;
	for (int i = 0; i < orderU; ++i)
	{
		result.x += tempCtrlPoint1[i].x * mul1[orderU - 1 - i];
		result.y += tempCtrlPoint1[i].y * mul1[orderU - 1 - i];
		result.z += tempCtrlPoint1[i].z * mul1[orderU - 1 - i];
	}
	return result;
}

/* kernel，计算三个方向参数分别为 u, v, w 的点的 B 样条体值 */
__global__ void fromParamToCoordOnePoint(float3 *vertexCoordListD, float3 *vertexParamListD,
										 int vertexCount, int orderU, int orderV, int orderW,
										 int ctrlPointNumU, int ctrlPointNumV, int ctrlPointNumW,
										 int knotIntervalCountU, int knotIntervalCountV, int knotIntervalCountW)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= vertexCount)
		return;

	float3 tempVertexParam = vertexParamListD[idx];
	float u = tempVertexParam.x;
	float v = tempVertexParam.y;
	float w = tempVertexParam.z;

	// 预先将其值设为最大，将末端点归入最后一段
	int leftUIdx, leftVIdx, leftWIdx;
	leftUIdx = orderU - 1 + knotIntervalCountU - 1;
	leftVIdx = orderV - 1 + knotIntervalCountV - 1;
	leftWIdx = orderW - 1 + knotIntervalCountW - 1;

	// 沿 U 方向查找当前点所在的节点区间
	for (int i = orderU - 1; i <= orderU - 1 + knotIntervalCountU - 1; ++i)
	{
		if (u >= knotListD[i] && u < knotListD[i + 1])
		{
			leftUIdx = i;
			break;
		}
	}
	// 沿 V 方向查找当前点所在的节点区间
	for (int j = orderV - 1; j <= orderV - 1 + knotIntervalCountV - 1; ++j)
	{
		if (v >= knotListD[20 + j] && v < knotListD[20 + j + 1])
		{
			leftVIdx = j;
			break;
		}
	}
	// 沿 W 方向查找当前点所在的节点区间
	for (int k = orderW - 1; k <= orderW - 1 + knotIntervalCountW - 1; ++k)
	{
		if (w >= knotListD[40 + k] && w < knotListD[40 + k + 1])
		{
			leftWIdx = k;
			break;
		}
	}
	vertexCoordListD[idx] = BSplineVolumeValueMatrixD(u, v, w, leftUIdx, leftVIdx, leftWIdx,
													  orderU, orderV, orderW,
													  ctrlPointNumU, ctrlPointNumV, ctrlPointNumW);
}

float3 *vertexParamListD = 0;					// 模型顶点参数序列
float3 *vertexCoordListD = 0;					// 模型顶点坐标序列

int order[3], ctrlPointNum[3], knotIntervalCount[3], knotCount[3];		// 三个方向的阶数、控制顶点数、节点区间数、节点数
float knotList[3][20];														// 三个方向的节点向量
float3 ctrlPoint[15][15][15];												// 控制顶点
cublasHandle_t cublas_handle = 0;

/* 将 B 样条矩阵载入显存 */
void loadMBD()
{
	cudaMemcpyToSymbol(MB2fD,  MB2f,  sizeof(float) * 2 * 2);
	cudaMemcpyToSymbol(MB30fD, MB30f, sizeof(float) * 3 * 3);
	cudaMemcpyToSymbol(MB31fD, MB31f, sizeof(float) * 3 * 3);
	cudaMemcpyToSymbol(MB32fD, MB32f, sizeof(float) * 3 * 3);
	cudaMemcpyToSymbol(MB33fD, MB33f, sizeof(float) * 3 * 3);
	cudaMemcpyToSymbol(MB40fD, MB40f, sizeof(float) * 4 * 4);
	cudaMemcpyToSymbol(MB41fD, MB41f, sizeof(float) * 4 * 4);
	cudaMemcpyToSymbol(MB42fD, MB42f, sizeof(float) * 4 * 4);
	cudaMemcpyToSymbol(MB43fD, MB43f, sizeof(float) * 4 * 4);
	cudaMemcpyToSymbol(MB44fD, MB44f, sizeof(float) * 4 * 4);
	cudaMemcpyToSymbol(MB45fD, MB45f, sizeof(float) * 4 * 4);
	cudaMemcpyToSymbol(MB46fD, MB46f, sizeof(float) * 4 * 4);
	cudaMemcpyToSymbol(MB47fD, MB47f, sizeof(float) * 4 * 4);
	cudaMemcpyToSymbol(MB48fD, MB48f, sizeof(float) * 4 * 4);

	cublasCreate(&cublas_handle);
	cout << "handle = " << (long)cublas_handle << endl;
	cout << "cublasCreate" << endl;
}

void copyCtrlPointD(CommonData *commonData)
{
	for (int i = 0; i < ctrlPointNum[U]; ++i)
	{
		for (int j = 0; j < ctrlPointNum[V]; ++j)
		{
			for (int k = 0; k < ctrlPointNum[W]; ++k)
			{
				ctrlPoint[i][j][k].x = (float)commonData->getCtrlPoint(i, j, k).x();
				ctrlPoint[i][j][k].y = (float)commonData->getCtrlPoint(i, j, k).y();
				ctrlPoint[i][j][k].z = (float)commonData->getCtrlPoint(i, j, k).z();
			}
		}
	}
	cymError = cudaMemcpyToSymbol(ctrlPointD, &ctrlPoint[0][0][0], sizeof(float3) * 15 * 15 * 15);
	if (cymError)
		cerr << "copyCtrlPointD = " << cymError << endl;
}

/*
 * 根据所有顶点的参数，计算出相应的 B 样条体值
 * 仅用于 FFD 算法
 */
void fromParamToCoordD(CommonData *commonData)
{
	int vertexCount = commonData->vertexCount();
	int threadCount = commonData->ffdThreadCount();
	fromParamToCoordOnePoint<<<vertexCount / threadCount + 1, threadCount>>>(
													vertexCoordListD, vertexParamListD,
													vertexCount, order[U], order[V], order[W],
													ctrlPointNum[U], ctrlPointNum[V], ctrlPointNum[W],
													knotIntervalCount[U], knotIntervalCount[V], knotIntervalCount[W]);
	float3 *vertexCoordList = new float3[vertexCount];
	cudaMemcpy(vertexCoordList, vertexCoordListD, sizeof(float3) * vertexCount, cudaMemcpyDeviceToHost);
	for (int i = 0; i < vertexCount; ++i)
		commonData->setVertexCoord(i, vertexCoordList[i].x, vertexCoordList[i].y, vertexCoordList[i].z);
	delete []vertexCoordList;
}

/* 把数字a转换成一个逗号分节的string */
string longNumber(int a)
{
	string result;
	for (; a > 0; a /= 1000)
	{
		string num3;
		ostringstream oss(num3);

		int remainder = a % 1000;
		if (a >= 1000)
		{
			if (remainder < 10)
				oss << "00" << remainder;
			else if (remainder >= 10 && remainder < 100)
				oss << "0" << remainder;
			else 
				oss << remainder;
		}
		else
			oss << remainder;

		if (result.size() == 0)
			result = oss.str();
		else
			result = oss.str() + "," + result;
	}
	return result;
}

/* 打印显存使用量 */
void printMemD(const char *file, const char *function, int line, int memSize)
{
	/* 只取文件名部分，路径舍弃 */
	string filePath(file);
	int lastSlashPos = filePath.rfind('/');
	filePath = filePath.substr(lastSlashPos + 1, filePath.size());

	totalMemD += memSize;
	cout << "文件"<< filePath << "，函数" << function << ", 第" << line << "行，申请显存" << longNumber(memSize) << "字节, "
		 << "目前累计使用显存" << longNumber(totalMemD) << "字节" << endl;
	//cout << "!!!!!!"<< function << ", 第" << line << "行，申请显存" << memSize << "字节, "
		 //<< "目前累计使用显存" << totalMemD << "字节" << endl;
}

/* 预计算，将内存中的数据拷贝到相应的显存空间中 */
void preCalcD(CommonData *commonData)
{
	for (int i = 0; i < 3; ++i)
	{
		order[i] = commonData->order(i);
		ctrlPointNum[i] = commonData->ctrlPointCount(i);
		knotIntervalCount[i] = commonData->knotIntervalCount(i);
		knotCount[i] = order[i] + ctrlPointNum[i];
	}
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < knotCount[i]; ++j)
			knotList[i][j] = (float)commonData->getKnot(i, j);
	cymError = cudaMemcpyToSymbol(knotListD, &knotList[0][0], sizeof(float) * 3 * 20);
	if (cymError)
		cerr << "preCalcD0, cymError = " << cymError << endl;

	int vertexCount = commonData->vertexCount();
	float3 *vertexParamListAlloc = new float3[vertexCount];
	for (int i = 0; i < vertexCount; ++i)
	{
		vertexParamListAlloc[i].x = (float)commonData->vertexParam(i).u();
		vertexParamListAlloc[i].y = (float)commonData->vertexParam(i).v();
		vertexParamListAlloc[i].z = (float)commonData->vertexParam(i).w();
	}
	cymError = cudaMalloc((void**)&vertexParamListD, sizeof(float3) * vertexCount);
	cout << "@原始模型上所有顶点的参数，仅用于FFD" << endl << "\t";
	printMemD(__FILE__, __FUNCTION__, __LINE__, sizeof(float3) * vertexCount);
	if (cymError)
		cerr << "preCalcD1, cymError = " << cymError << endl;

	cymError = cudaMemcpy(vertexParamListD, vertexParamListAlloc, sizeof(float3) * vertexCount, cudaMemcpyHostToDevice);
	if (cymError)
		cerr << "preCalcD2, cymError = " << cymError << endl;
	delete []vertexParamListAlloc;
	vertexParamListAlloc = 0;

	cymError = cudaMalloc((void**)&vertexCoordListD, sizeof(float3) * vertexCount);
	cout << "@原始模型上所有顶点的坐标，仅用于FFD" << endl << "\t";
	printMemD(__FILE__, __FUNCTION__, __LINE__, sizeof(float3) * vertexCount);
	if (cymError)
		cerr << "preCalcD3, cymError = " << cymError << endl;

	copyCtrlPointD(commonData);
}

/*static __device__ float matrixEdgeD[203];*/
/*static __device__ float matrixInteriorD[1596];*/
float *matrixEdgeD;
float *matrixInteriorD;

void loadTriangleMatrixD()
{
	extern float matrixEdge[203];
	cudaMalloc((void**)&matrixEdgeD, sizeof(float) * 203);
	/*cudaMemcpyToSymbol(matrixEdgeD, matrixEdge, sizeof(float) * 203);*/
	cudaMemcpy(matrixEdgeD, matrixEdge, sizeof(float) * 203, cudaMemcpyHostToDevice);

	extern float matrixInterior[1596];
	cudaMalloc((void**)&matrixInteriorD, sizeof(float) * 1596);
	/*cudaMemcpyToSymbol(matrixInteriorD, matrixInterior, sizeof(float) * 1596);*/
	cudaMemcpy(matrixInteriorD, matrixInterior, sizeof(float) * 1596, cudaMemcpyHostToDevice);
}

struct TriangleD
{
	float3 v[3];
	float2 vt[3];
	int boxIdx;
};

TriangleD *triangleListD;
int degree, triangleCtrlPointNum, triangleNum;

int blockSizeStep1 = 256;

int blockNumCorner;

int activeThreadNumEdge;
int trianglePerBlockEdge;
int matrixStartIdxEdge;
int blockNumEdge;

int activeThreadNumInterior;
int trianglePerBlockInterior;
int matrixStartIdxInterior;
int blockNumInterior;

int threadPerTriangleInterior;

float *edgeTD, *edgeCtrlPointD;
float *interiorTD, *triangleCtrlPointD;

__host__ __device__ inline int index2c(int i, int j, int stride)
{
	return j * stride + i;
}

void loadTriangleListD(const vector<Triangle> &triangleList, int deg)
{
	degree = deg;
	triangleNum = triangleList.size();

	TriangleD *tempTriangleList = new TriangleD[triangleNum];
	for (vector<Triangle>::size_type i = 0; i < triangleNum; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			tempTriangleList[i].v[j].x = triangleList[i].v[j].x();
			tempTriangleList[i].v[j].y = triangleList[i].v[j].y();
			tempTriangleList[i].v[j].z = triangleList[i].v[j].z();
			tempTriangleList[i].vt[j].x = triangleList[i].vt[j].u();
			tempTriangleList[i].vt[j].y = triangleList[i].vt[j].v();
		}
		tempTriangleList[i].boxIdx = triangleList[i].boxIdx;
	}
	cudaMalloc((void**)&triangleListD, sizeof(TriangleD) * triangleNum);
	cout << "@原始模型上所有三角形信息" << endl << "\t";
	printMemD(__FILE__, __FUNCTION__, __LINE__, sizeof(TriangleD) * triangleNum);

	cudaMemcpy(triangleListD, tempTriangleList, sizeof(TriangleD) * triangleNum, cudaMemcpyHostToDevice);

	triangleCtrlPointNum = (deg + 1) * (deg + 2) / 2;

	delete []tempTriangleList;

	////////////////////////////////////
	blockNumCorner = triangleNum / blockSizeStep1;
	if (blockNumCorner * blockSizeStep1 != triangleNum)
		++blockNumCorner;

	activeThreadNumEdge = blockSizeStep1 / (degree - 1) * (degree - 1);
	trianglePerBlockEdge = activeThreadNumEdge / (degree - 1);
	blockNumEdge = triangleNum / trianglePerBlockEdge;
	if (blockNumEdge * trianglePerBlockEdge != triangleNum)
		++blockNumEdge;

	threadPerTriangleInterior = (deg - 2) * (deg - 1) / 2;
	activeThreadNumInterior = blockSizeStep1 / threadPerTriangleInterior * threadPerTriangleInterior;
	trianglePerBlockInterior = activeThreadNumInterior / threadPerTriangleInterior;
	blockNumInterior = triangleNum / trianglePerBlockInterior;
	if (blockNumInterior * trianglePerBlockInterior != triangleNum)
		++blockNumInterior;
	////////////////////////////////////

	/*cudaMalloc(&sampleValueD, sizeof(float) * triangleCtrlPointNum * triangleNum * 3);*/
	/*cout << "@为了求Bezier曲面片的控制顶点，需要在其上进行采样，结果放在这里。即第二个矩阵乘法用到的矩阵T" << endl << "\t";*/
	/*printMemD(__FILE__, __FUNCTION__, __LINE__, sizeof(float) * triangleCtrlPointNum * triangleNum * 3);*/

	/*activeThreadNumStep0 = triangleCtrlPointNum * triangleNum;*/
	/*blockNumStep0 = ceil(static_cast<double>(activeThreadNumStep0) / blockSizeStep0);*/

	/*extern float matrixTriangle[7][55*55];*/
	/*float *temp = new float[triangleCtrlPointNum * triangleCtrlPointNum];*/
	/*for (int i = 0; i < triangleCtrlPointNum; ++i)*/
	/*{*/
		/*for (int j = 0; j < triangleCtrlPointNum; ++j)*/
		/*{*/
			/*temp[index2c(i, j, triangleCtrlPointNum)] = matrixTriangle[degree - 3][i * triangleCtrlPointNum + j];*/
		/*}*/
	/*}*/
	/*cudaMalloc(&B_1D, sizeof(float) * triangleCtrlPointNum * triangleCtrlPointNum);*/
	/*cout << "@第一个矩阵乘法用到的矩阵(B-1)T存放在这里" << endl << "\t";*/
	/*printMemD(__FILE__, __FUNCTION__, __LINE__, sizeof(float) * triangleCtrlPointNum * triangleCtrlPointNum);*/
	/*cudaMemcpy(B_1D, temp, sizeof(float) * triangleCtrlPointNum * triangleCtrlPointNum, cudaMemcpyHostToDevice);*/
	/*delete temp;*/

	matrixStartIdxEdge = 0;
	matrixStartIdxInterior = 0;
	for (int i = 3; i < degree; ++i)
	{
		matrixStartIdxEdge += (i - 1) * (i - 1);
		matrixStartIdxInterior += (i - 1) * (i - 1) * (i - 2) * (i - 2) / 4;
	}

	/***************************************************************************/

	cudaMalloc((void**)&edgeTD, sizeof(float) * 3 * triangleNum * (degree - 1) * 3);
	cudaMalloc((void**)&edgeCtrlPointD, sizeof(float) * 3 * triangleNum * (degree - 1) * 3);

	/***************************************************************************/

	cudaMalloc((void**)&interiorTD, sizeof(float) * triangleNum * threadPerTriangleInterior * 3);
	cudaMalloc((void**)&triangleCtrlPointD, sizeof(float) * triangleNum * triangleCtrlPointNum * 3);

	/***************************************************************************/

	cout << "degree = " << degree << endl;
	cout << "triangleCtrlPointNum = " << triangleCtrlPointNum << endl;
	cout << "triangleNum = " << triangleNum << endl << endl;
	/*cout << "activeThreadNumStep0 = " << activeThreadNumStep0 << endl;*/
	/*cout << "blockNumStep0 = " << blockNumStep0 << endl;*/
}

void printCudaError(const char *function, int lineNum)
{
	cymError = cudaGetLastError();
	cout << "print, 函数" << function << ", " << lineNum << "行，CUDA错误：" << cudaGetErrorString(cymError) << endl;
}

void printCudaError(int lineNum)
{
	cymError = cudaGetLastError();
	cout << "print" << lineNum << "行，CUDA错误：" << cudaGetErrorString(cymError) << endl;
}

__device__ float3 operator+(const float3 &a, const float3 &b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator*(float a, const float3 &b)
{
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ float3 operator*(const float3 &a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ void operator+=(float3 &a, const float3 &b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

__device__ void operator-=(float3 &a, const float3 &b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

/* 使用矩阵乘法求 B 样条体的值，和上面一个类似函数的区别在于不负责 u、v、w 重新参数化的工作，
 * 而且也不负责求合适的 B 样条矩阵，这两项工作均需调用函数之前完成，参数列表得到简化 */
__device__ float3 BSplineVolumeValueMatrixD2(float *Mu, float *Mv, float *Mw,
											 float u, float v, float w, float3 *pointValue,
											 int leftUIdx, int leftVIdx, int leftWIdx,
											 int orderU, int orderV, int orderW)
{
	float *mul1 = (float *)&pointValue[2 * blockDim.x];
	float *mul2 = (float *)&mul1[4 * blockDim.x];
	float temp[3], tempV[3];

	// 由三维控制顶点算出二维临时控制顶点
	temp[0] = w;
	temp[1] = w * w;
	temp[2] = w * w * w;

	for (int i = 0; i < orderW; ++i)
	{
		mul1[4 * threadIdx.x + i] = Mw[i];
		for (int j = 1; j < orderW; ++j)
			mul1[4 * threadIdx.x + i] += temp[j - 1] * Mw[j * orderW + i];
	}

	// 由二维临时控制顶点算出一维临时控制顶点
	tempV[0] = v;
	tempV[1] = v * v;
	tempV[2] = v * v * v;

	for (int i = 0; i < orderV; ++i)
	{
		mul2[4 * threadIdx.x + i] = Mv[i];
		for (int j = 1; j < orderV; ++j)
			mul2[4 * threadIdx.x + i] += tempV[j - 1] * Mv[j * orderV + i];
	}

	float3 tempCtrlPoint2[4];

	float3 tempCtrlPoint1[4];
	for (int i = 0; i < orderU; ++i)
	{
		for (int j = 0; j < orderV; ++j)
		{
			tempCtrlPoint2[j] = make_float3(0.0f, 0.0f, 0.0f);
			for (int k = 0; k < orderW; ++k)
			{
				float3 cp = ctrlPointD[(leftUIdx - i) * 15 * 15 + (leftVIdx - j) * 15 + leftWIdx - k];
				tempCtrlPoint2[j] += cp * mul1[4 * threadIdx.x + orderW - 1 - k];
			}
		}
		tempCtrlPoint1[i] = make_float3(0.0f, 0.0f, 0.0f);
		for (int j = 0; j < orderV; ++j)
			tempCtrlPoint1[i] += tempCtrlPoint2[j] * mul2[4 * threadIdx.x + orderV - 1 - j];
	}

	// 由一维临时控制顶点算出结果
	temp[0] = u;
	temp[1] = u * u;
	temp[2] = u * u * u;

	for (int i = 0; i < orderU; ++i)
	{
		mul1[4 * threadIdx.x + i] = Mu[i];
		for (int j = 1; j < orderU; ++j)
			mul1[4 * threadIdx.x + i] += temp[j - 1] * Mu[j * orderU + i];
	}
	float3 result = make_float3(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < orderU; ++i)
		result += tempCtrlPoint1[i] * mul1[4 * threadIdx.x + orderU - 1 - i];
	return result;
}

__global__ void calcBezierCtrlPointCorner(TriangleD *triangleListD, float *triangleCtrlPointD,
										  int ctrlPointNum, int triangleNum,
										  int degree, int orderU, int orderV, int orderW,
										  int ctrlPointCountU, int ctrlPointCountV, int ctrlPointCountW)
{
	int triangleIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (triangleIdx >= triangleNum)
		return;

	TriangleD &triangle = triangleListD[triangleIdx];
	int boxIdx = triangle.boxIdx;
	int i = boxIdx >> 16;
	int j = boxIdx >> 8 & 0xFF;
	int k = boxIdx & 0xFF;

	/* 确定此 block 需要的 u、v、w 三个方向的 B 样条矩阵 */
	float *Mu = matrixCaseD(orderU, ctrlPointCountU, i + orderU - 1);
	float *Mv = matrixCaseD(orderV, ctrlPointCountV, j + orderV - 1);
	float *Mw = matrixCaseD(orderW, ctrlPointCountW, k + orderW - 1);

	extern __shared__ float shared_array[];
	__shared__ float3 *pointValue;
	for (int cornerIdx = 0; cornerIdx < 3; ++cornerIdx)
	{
		float u = triangle.v[cornerIdx].x;
		float v = triangle.v[cornerIdx].y;
		float w = triangle.v[cornerIdx].z;

		float tmpKnot = knotListD[i + orderU - 1];
		float tmpKnot1 = knotListD[i + orderU];
		u = (u - tmpKnot) / (tmpKnot1 - tmpKnot);

		tmpKnot = knotListD[20 + j + orderV - 1];
		tmpKnot1 = knotListD[20 + j + orderV];
		v = (v - tmpKnot) / (tmpKnot1 - tmpKnot);

		tmpKnot = knotListD[40 + k + orderW - 1];
		tmpKnot1 = knotListD[40 + k + orderW];
		w = (w - tmpKnot) / (tmpKnot1 - tmpKnot);

		pointValue = (float3 *)shared_array;
		/* 算出该线程负责的采样点的 B 样条体值 */
		pointValue[threadIdx.x] = BSplineVolumeValueMatrixD2(Mu, Mv, Mw,
															 u, v, w, pointValue,
															 i + orderU - 1, j + orderV - 1, k + orderW - 1,
															 orderU, orderV, orderW);
		triangleCtrlPointD[cornerIdx * triangleNum * 3 + triangleIdx] = pointValue[threadIdx.x].x;
		triangleCtrlPointD[cornerIdx * triangleNum * 3 + triangleNum + triangleIdx] = pointValue[threadIdx.x].y;
		triangleCtrlPointD[cornerIdx * triangleNum * 3 + triangleNum * 2 + triangleIdx] = pointValue[threadIdx.x].z;
		__syncthreads();
	}
}

__device__ float power(float a, int n)
{
	if (n <= 0)
		return 1.0;
	float result = a;
	for (int i = 1; i < n; ++i)
		result *= a;
	return result;
}

double power(double a, int n)
{
	if (n <= 0)
		return 1.0;
	double result = a;
	for (int i = 1; i < n; ++i)
		result *= a;
	return result;
}

__global__ void calcBezierCtrlPointEdge(TriangleD *triangleListD, float *matrixEdgeD,
										float *edgeTD, float *triangleCtrlPointD,
										int activeThreadNum, int trianglePerBlock,
										int threadPerTriangleEdge, int matrixStartIdx,
										int ctrlPointNum, int triangleNum,
										int degree, int orderU, int orderV, int orderW,
										int ctrlPointCountU, int ctrlPointCountV, int ctrlPointCountW)
{
	if (threadIdx.x >= activeThreadNum)
		return;
	int triangleIdx = trianglePerBlock * blockIdx.x + threadIdx.x / threadPerTriangleEdge;
	if (triangleIdx >= triangleNum)
		return;

	TriangleD &triangle = triangleListD[triangleIdx];
	int boxIdx = triangle.boxIdx;
	int i = boxIdx >> 16;
	int j = boxIdx >> 8 & 0xFF;
	int k = boxIdx & 0xFF;

	float3 vec0 = triangle.v[0];
	float3 vec1 = triangle.v[1];
	float3 vec2 = triangle.v[2];

	/* 确定此 block 需要的 u、v、w 三个方向的 B 样条矩阵 */
	float *Mu = matrixCaseD(orderU, ctrlPointCountU, i + orderU - 1);
	float *Mv = matrixCaseD(orderV, ctrlPointCountV, j + orderV - 1);
	float *Mw = matrixCaseD(orderW, ctrlPointCountW, k + orderW - 1);

	extern __shared__ float shared_array[];
	/*__shared__ float3 *pointValue, *result;*/
	__shared__ float3 *pointValue;
	int localIdx = threadIdx.x % threadPerTriangleEdge;
	for (int edgeIdx = 0; edgeIdx < 3; ++edgeIdx)
	{
		float3 barycentric_coord = make_float3(0.0f, 0.0f, 0.0f);
		if (edgeIdx == 0)
		{
			barycentric_coord.z = (float)(localIdx + 1) / degree;
			barycentric_coord.y = 1.0f - barycentric_coord.z;
			/*barycentric_coord.y = 1.0f;*/ 			// 刺猬
		}
		else if (edgeIdx == 1)
		{
			barycentric_coord.x = (float)(localIdx + 1) / degree;
			barycentric_coord.z = 1.0f - barycentric_coord.x;
			/*barycentric_coord.z = 1.0f;*/ 			// 刺猬
		}
		else
		{
			barycentric_coord.y = (float)(localIdx + 1) / degree;
			barycentric_coord.x = 1.0f - barycentric_coord.y;
			/*barycentric_coord.x = 1.0f;*/ 			// 刺猬
		}

		float u = vec0.x * barycentric_coord.x + vec1.x * barycentric_coord.y + vec2.x * barycentric_coord.z;
		float v = vec0.y * barycentric_coord.x + vec1.y * barycentric_coord.y + vec2.y * barycentric_coord.z;
		float w = vec0.z * barycentric_coord.x + vec1.z * barycentric_coord.y + vec2.z * barycentric_coord.z;

		float tmpKnot = knotListD[i + orderU - 1];
		float tmpKnot1 = knotListD[i + orderU];
		u = (u - tmpKnot) / (tmpKnot1 - tmpKnot);

		tmpKnot = knotListD[20 + j + orderV - 1];
		tmpKnot1 = knotListD[20 + j + orderV];
		v = (v - tmpKnot) / (tmpKnot1 - tmpKnot);

		tmpKnot = knotListD[40 + k + orderW - 1];
		tmpKnot1 = knotListD[40 + k + orderW];
		w = (w - tmpKnot) / (tmpKnot1 - tmpKnot);

		pointValue = (float3 *)shared_array;
		/* 算出该线程负责的采样点的 B 样条体值 */
		pointValue[threadIdx.x] = BSplineVolumeValueMatrixD2(Mu, Mv, Mw,
															 u, v, w, pointValue,
															 i + orderU - 1, j + orderV - 1, k + orderW - 1,
															 orderU, orderV, orderW);

		float t = (float)(localIdx + 1) / degree;
		float Bn0t = power(1 - t, degree);
		float Bnnt = power(t, degree);
		pointValue[threadIdx.x].x -= triangleCtrlPointD[(edgeIdx + 1) % 3 * triangleNum * 3 + triangleIdx] * Bn0t;
		pointValue[threadIdx.x].x -= triangleCtrlPointD[(edgeIdx + 2) % 3 * triangleNum * 3 + triangleIdx] * Bnnt;
		pointValue[threadIdx.x].y -= triangleCtrlPointD[(edgeIdx + 1) % 3 * triangleNum * 3 + triangleNum + triangleIdx] * Bn0t;
		pointValue[threadIdx.x].y -= triangleCtrlPointD[(edgeIdx + 2) % 3 * triangleNum * 3 + triangleNum + triangleIdx] * Bnnt;
		pointValue[threadIdx.x].z -= triangleCtrlPointD[(edgeIdx + 1) % 3 * triangleNum * 3 + triangleNum * 2 + triangleIdx] * Bn0t;
		pointValue[threadIdx.x].z -= triangleCtrlPointD[(edgeIdx + 2) % 3 * triangleNum * 3 + triangleNum * 2 + triangleIdx] * Bnnt;
		__syncthreads();
		/*if (triangleIdx == 0)*/
		/*{*/
			/*printf("tid = %d, eid = %d, (%f, %f, %f), edgeTDIdx = %d\n",*/
					/*threadIdx.x, edgeIdx, pointValue[threadIdx.x].x,*/
					/*pointValue[threadIdx.x].y,*/
					/*pointValue[threadIdx.x].z,*/
					/*localIdx * triangleNum * 3 + triangleIdx * 3 + edgeIdx);*/
		/*}*/

		/* 写入edgeTD */
		edgeTD[localIdx * triangleNum * 9 + triangleIdx * 3 + edgeIdx] = pointValue[threadIdx.x].x;
		edgeTD[localIdx * triangleNum * 9 + triangleNum * 3 + triangleIdx * 3 + edgeIdx] = pointValue[threadIdx.x].y;
		edgeTD[localIdx * triangleNum * 9 + triangleNum * 3 * 2 + triangleIdx * 3 + edgeIdx] = pointValue[threadIdx.x].z;
		/*__syncthreads();*/
	}
}

__host__ __device__ int factorial(int n)
{
	int result = 1;
	for (int i = 1; i <= n; ++i)
		result *= i;
	return result;
}

float B(double u, double v, double w, int n, int3 c)
{
	/*cout << "uvw = " << u << ", " << v << ", " << w << ", n = " << n << ", c = (" << c.x << ", " << c.y << ", " << c.z << ")" << endl;*/
	return factorial(n) / factorial(c.x) / factorial(c.y) / factorial(c.z) * power(u, c.x) * power(v, c.y) * power(w, c.z);
}

double B(int n, int i, double t)
{
	return factorial(n) / factorial(i) / factorial(n - i) * power(t, i) * power(1 - t, n - i);
}

__device__ float B(int n, int i, float u, float v)
{
	return factorial(n) / factorial(i) / factorial(n - i) * power(u, i) * power(v, n - i);
}

__global__ void calcBezierCtrlPointInterior(TriangleD *triangleListD, float *matrixInteriorD,
											float *interiorTD, float *triangleCtrlPointD,
											int activeThreadNum, int trianglePerBlock,
											int threadPerTriangleInterior, int matrixStartIdx,
											int ctrlPointNum, int triangleNum,
											int degree, int orderU, int orderV, int orderW,
											int ctrlPointCountU, int ctrlPointCountV, int ctrlPointCountW)
{
	/*int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;*/
	if (threadIdx.x >= activeThreadNum)
		return;
	int triangleIdx = trianglePerBlock * blockIdx.x + threadIdx.x / threadPerTriangleInterior;
	if (triangleIdx >= triangleNum)
		return;

	TriangleD triangle = triangleListD[triangleIdx];
	int boxIdx = triangle.boxIdx;
	int i = boxIdx >> 16;
	int j = boxIdx >> 8 & 0xFF;
	int k = boxIdx & 0xFF;

	float3 vec0 = triangle.v[0];
	float3 vec1 = triangle.v[1];
	float3 vec2 = triangle.v[2];

	/* 确定此 block 需要的 u、v、w 三个方向的 B 样条矩阵 */
	float *Mu = matrixCaseD(orderU, ctrlPointCountU, i + orderU - 1);
	float *Mv = matrixCaseD(orderV, ctrlPointCountV, j + orderV - 1);
	float *Mw = matrixCaseD(orderW, ctrlPointCountW, k + orderW - 1);

	int localIdx = threadIdx.x % threadPerTriangleInterior;

	float tempFloorFloat = (sqrt((float)localIdx * 8 + 9) - 3) / 2;
	int floor = rintf(tempFloorFloat);
	if ((floor * 2 + 3) * (floor * 2 + 3) != localIdx * 8 + 9)
		floor = ceilf(tempFloorFloat);
	int room = localIdx - (floor + 1) * floor / 2;
	floor += 2;
	++room;
	float3 barycentric_coord;
	barycentric_coord.x = (float)(degree - floor) / degree;
	barycentric_coord.y = (float)(floor - room) / degree;
	barycentric_coord.z = 1.0f - barycentric_coord.x - barycentric_coord.y;

	float u = vec0.x * barycentric_coord.x + vec1.x * barycentric_coord.y + vec2.x * barycentric_coord.z;
	float v = vec0.y * barycentric_coord.x + vec1.y * barycentric_coord.y + vec2.y * barycentric_coord.z;
	float w = vec0.z * barycentric_coord.x + vec1.z * barycentric_coord.y + vec2.z * barycentric_coord.z;

	float tmpKnot = knotListD[i + orderU - 1];
	float tmpKnot1 = knotListD[i + orderU];
	u = (u - tmpKnot) / (tmpKnot1 - tmpKnot);

	tmpKnot = knotListD[20 + j + orderV - 1];
	tmpKnot1 = knotListD[20 + j + orderV];
	v = (v - tmpKnot) / (tmpKnot1 - tmpKnot);

	tmpKnot = knotListD[40 + k + orderW - 1];
	tmpKnot1 = knotListD[40 + k + orderW];
	w = (w - tmpKnot) / (tmpKnot1 - tmpKnot);

	extern __shared__ float shared_array[];
	__shared__ float3 *pointValue;
	pointValue = (float3 *)shared_array;
	/* 算出该线程负责的采样点的 B 样条体值 */
	pointValue[threadIdx.x] = BSplineVolumeValueMatrixD2(Mu, Mv, Mw,
														 u, v, w, pointValue,
														 i + orderU - 1, j + orderV - 1, k + orderW - 1,
														 orderU, orderV, orderW);

	float un = power(barycentric_coord.x, degree);
	float vn = power(barycentric_coord.y, degree);
	float wn = power(barycentric_coord.z, degree);
	float3 p0, p1, p2;
	p0.x = triangleCtrlPointD[triangleIdx];
	p0.y = triangleCtrlPointD[triangleNum + triangleIdx];
	p0.z = triangleCtrlPointD[triangleNum * 2 + triangleIdx];
	p1.x = triangleCtrlPointD[triangleNum * 3 + triangleIdx];
	p1.y = triangleCtrlPointD[triangleNum * 4 + triangleIdx];
	p1.z = triangleCtrlPointD[triangleNum * 5 + triangleIdx];
	p2.x = triangleCtrlPointD[triangleNum * 6 + triangleIdx];
	p2.y = triangleCtrlPointD[triangleNum * 7 + triangleIdx];
	p2.z = triangleCtrlPointD[triangleNum * 8 + triangleIdx];
	pointValue[threadIdx.x].x -= (un * p0.x + vn * p1.x + wn * p2.x);
	pointValue[threadIdx.x].y -= (un * p0.y + vn * p1.y + wn * p2.y);
	pointValue[threadIdx.x].z -= (un * p0.z + vn * p1.z + wn * p2.z);

	for (int idx = 1; idx < degree; ++idx)
	{
		float b = B(degree, degree - idx, barycentric_coord.y, barycentric_coord.z);
		pointValue[threadIdx.x].x -= b * triangleCtrlPointD[(3 + idx - 1) * triangleNum * 3 + triangleIdx];
		pointValue[threadIdx.x].y -= b * triangleCtrlPointD[(3 + idx - 1) * triangleNum * 3 + triangleNum + triangleIdx];
		pointValue[threadIdx.x].z -= b * triangleCtrlPointD[(3 + idx - 1) * triangleNum * 3 + triangleNum * 2 + triangleIdx];
	}
	for (int idx = 1; idx < degree; ++idx)
	{
		float b = B(degree, degree - idx, barycentric_coord.z, barycentric_coord.x);
		pointValue[threadIdx.x].x -= b * triangleCtrlPointD[(3 + (degree - 1) + idx - 1) * triangleNum * 3 + triangleIdx];
		pointValue[threadIdx.x].y -= b * triangleCtrlPointD[(3 + (degree - 1) + idx - 1) * triangleNum * 3 + triangleNum + triangleIdx];
		pointValue[threadIdx.x].z -= b * triangleCtrlPointD[(3 + (degree - 1) + idx - 1) * triangleNum * 3 + triangleNum * 2 + triangleIdx];
	}
	for (int idx = 1; idx < degree; ++idx)
	{
		float b = B(degree, degree - idx, barycentric_coord.x, barycentric_coord.y);
		pointValue[threadIdx.x].x -= b * triangleCtrlPointD[(3 + (degree - 1) * 2 + idx - 1) * triangleNum * 3 + triangleIdx];
		pointValue[threadIdx.x].y -= b * triangleCtrlPointD[(3 + (degree - 1) * 2 + idx - 1) * triangleNum * 3 + triangleNum + triangleIdx];
		pointValue[threadIdx.x].z -= b * triangleCtrlPointD[(3 + (degree - 1) * 2 + idx - 1) * triangleNum * 3 + triangleNum * 2 + triangleIdx];
	}
	/*__syncthreads();*/

	interiorTD[localIdx * triangleNum * 3 + triangleIdx] = pointValue[threadIdx.x].x;
	interiorTD[localIdx * triangleNum * 3 + triangleNum + triangleIdx] = pointValue[threadIdx.x].y;
	interiorTD[localIdx * triangleNum * 3 + triangleNum * 2 + triangleIdx] = pointValue[threadIdx.x].z;
	/*printf("%d, triangle = %d, local = %d, idx = %d\n", globalIdx, triangleIdx, localIdx,*/
			/*triangleIdx * triangleNum + localIdx);*/
}

__global__ void copyEdgeToInterior(float *edgeCtrlPointD, float *triangleCtrlPointD,
								   int activeThreadNumCopyEdgeToInterior, int triangleNum, int degree_1)
{
	int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (globalIdx >= activeThreadNumCopyEdgeToInterior)
		return;

	int triangleIdx = globalIdx % triangleNum;
	int localIdx = globalIdx / triangleNum;

	int targetIdx0 = (3 + localIdx) * triangleNum * 3 + triangleIdx;
	int targetIdx1 = targetIdx0 + degree_1 * triangleNum * 3;
	int targetIdx2 = targetIdx1 + degree_1 * triangleNum * 3;

	triangleCtrlPointD[targetIdx0] = edgeCtrlPointD[localIdx * triangleNum * 9 + triangleIdx * 3];
	triangleCtrlPointD[targetIdx1] = edgeCtrlPointD[localIdx * triangleNum * 9 + triangleIdx * 3 + 1];
	triangleCtrlPointD[targetIdx2] = edgeCtrlPointD[localIdx * triangleNum * 9 + triangleIdx * 3 + 2];

	targetIdx0 += triangleNum;
	targetIdx1 += triangleNum;
	targetIdx2 += triangleNum;

	triangleCtrlPointD[targetIdx0] = edgeCtrlPointD[localIdx * triangleNum * 9 + triangleNum * 3 + triangleIdx * 3];
	triangleCtrlPointD[targetIdx1] = edgeCtrlPointD[localIdx * triangleNum * 9 + triangleNum * 3 + triangleIdx * 3 + 1];
	triangleCtrlPointD[targetIdx2] = edgeCtrlPointD[localIdx * triangleNum * 9 + triangleNum * 3 + triangleIdx * 3 + 2];

	targetIdx0 += triangleNum;
	targetIdx1 += triangleNum;
	targetIdx2 += triangleNum;

	triangleCtrlPointD[targetIdx0] = edgeCtrlPointD[localIdx * triangleNum * 9 + triangleNum * 6 + triangleIdx * 3];
	triangleCtrlPointD[targetIdx1] = edgeCtrlPointD[localIdx * triangleNum * 9 + triangleNum * 6 + triangleIdx * 3 + 1];
	triangleCtrlPointD[targetIdx2] = edgeCtrlPointD[localIdx * triangleNum * 9 + triangleNum * 6 + triangleIdx * 3 + 2];
}

/* 使用 GPU 计算每个包围盒的 Bézier 曲面控制顶点 */
void calcBezierCtrlPointD()
{
	/*float *ax = new float[3 * triangleNum * (degree + 1)];*/
	/*float *ay = new float[3 * triangleNum * (degree + 1)];*/
	/*float *az = new float[3 * triangleNum * (degree + 1)];*/
	/*fill(ax, ax + 3 * triangleNum * (degree + 1), 123);*/
	/*fill(ay, ay + 3 * triangleNum * (degree + 1), 234);*/
	/*fill(az, az + 3 * triangleNum * (degree + 1), 345);*/

	calcBezierCtrlPointCorner<<<blockNumCorner, blockSizeStep1, sizeof(float) * blockSizeStep1 * 14>>>
										(triangleListD, triangleCtrlPointD,
										 triangleCtrlPointNum, triangleNum,
										 degree, order[U], order[V], order[W],
										 ctrlPointNum[U], ctrlPointNum[V], ctrlPointNum[W]);
	/*cudaMemcpy(ax, edgeCtrlPointXD, sizeof(float) * 3 * triangleNum * (degree + 1), cudaMemcpyDeviceToHost);*/
	/*cudaMemcpy(ay, edgeCtrlPointYD, sizeof(float) * 3 * triangleNum * (degree + 1), cudaMemcpyDeviceToHost);*/
	/*cudaMemcpy(az, edgeCtrlPointZD, sizeof(float) * 3 * triangleNum * (degree + 1), cudaMemcpyDeviceToHost);*/
	/*for (int i = 0; i < 3 * triangleNum * (degree + 1); ++i)*/
		/*fout << "(" << ax[i] << ", " << ay[i] << ", " << az[i] << ")" << endl;*/
	calcBezierCtrlPointEdge<<<blockNumEdge, blockSizeStep1, sizeof(float) * blockSizeStep1 * 14>>>
										(triangleListD, matrixEdgeD,
										 edgeTD, triangleCtrlPointD,
										 activeThreadNumEdge, trianglePerBlockEdge,
										 degree - 1, matrixStartIdxEdge,
										 triangleCtrlPointNum, triangleNum,
										 degree, order[U], order[V], order[W],
										 ctrlPointNum[U], ctrlPointNum[V], ctrlPointNum[W]);
	/*cudaMemcpy(a, edgeTXD, sizeof(float) * 3 * triangleNum * (degree - 1), cudaMemcpyDeviceToHost);*/
	/*for (int i = 0; i < 3 * triangleNum * (degree - 1); ++i)*/
		/*fout << a[i] << endl;*/
	/* 计算边界控制顶点 */
	float alpha = 1.0f, beta = 0.0f;
	cublasStatus_t stat = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
									  triangleNum * 3 * 3, degree - 1, degree - 1,
									  &alpha,
									  edgeTD, triangleNum * 3 * 3,
									  matrixEdgeD + matrixStartIdxEdge, degree - 1,
									  &beta,
									  edgeCtrlPointD, triangleNum * 3 * 3);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		cout << "edgeCtrLPoint fail!!!!!!!!!!!!!" << endl;
		cout << "stat = " << stat << endl;
		cudaError_t error = cudaGetLastError();
		cout << "CUDA error: " << cudaGetErrorString(error) << endl;
		return;
	}

	/*fill(ax, ax + 3 * triangleNum * (degree + 1), 123);*/
	/*fill(ay, ay + 3 * triangleNum * (degree + 1), 234);*/
	/*fill(az, az + 3 * triangleNum * (degree + 1), 345);*/
	/*cudaMemcpy(ax, edgeCtrlPointXD, sizeof(float) * 3 * triangleNum * (degree + 1), cudaMemcpyDeviceToHost);*/
	/*cudaMemcpy(ay, edgeCtrlPointYD, sizeof(float) * 3 * triangleNum * (degree + 1), cudaMemcpyDeviceToHost);*/
	/*cudaMemcpy(az, edgeCtrlPointZD, sizeof(float) * 3 * triangleNum * (degree + 1), cudaMemcpyDeviceToHost);*/
	/*fout << "控制顶点" << endl;*/
	/*for (int i = 0; i < 3 * triangleNum; ++i)*/
	/*{*/
		/*fout << i << endl;*/
		/*for (int j = 0; j < degree + 1; ++j)*/
		/*{*/
			/*int idx = j * 3 * triangleNum + i;*/
			/*fout << "(" << ax[idx] << ", " << ay[idx] << ", " << az[idx] << ")" << endl;*/
		/*}*/
		/*fout << endl;*/
	/*}*/

	int blockSizeCopyEdgeToInterior = 256;
	int activeThreadNumCopyEdgeToInterior = triangleNum * (degree - 1);
	int blockNumCopyEdgeToInterior = ceil(static_cast<double>(activeThreadNumCopyEdgeToInterior) / blockSizeCopyEdgeToInterior);
	/*cout << "activeThreadNumCopyEdgeToInterior = " << activeThreadNumCopyEdgeToInterior << endl;*/
	/*cout << "blockNumCopyEdgeToInterior = " << blockNumCopyEdgeToInterior << endl;*/

	copyEdgeToInterior<<<blockNumCopyEdgeToInterior, blockSizeCopyEdgeToInterior>>>
						(edgeCtrlPointD, triangleCtrlPointD,
						 activeThreadNumCopyEdgeToInterior, triangleNum, degree - 1);

	/*float *ix = new float[triangleNum * threadPerTriangleInterior];*/
	/*fill(ix, ix + triangleNum * threadPerTriangleInterior, 9876);*/
	/*cudaMemcpy(interiorTXD, ix, sizeof(float) * triangleNum * threadPerTriangleInterior, cudaMemcpyHostToDevice);*/
	/*fill(ix, ix + triangleNum * threadPerTriangleInterior, 6789);*/

	/*float *iy = new float[triangleNum * threadPerTriangleInterior];*/
	/*fill(iy, iy + triangleNum * threadPerTriangleInterior, 9876);*/
	/*cudaMemcpy(interiorTYD, iy, sizeof(float) * triangleNum * threadPerTriangleInterior, cudaMemcpyHostToDevice);*/
	/*fill(iy, iy + triangleNum * threadPerTriangleInterior, 6789);*/

	/*float *iz = new float[triangleNum * threadPerTriangleInterior];*/
	/*fill(iz, iz + triangleNum * threadPerTriangleInterior, 9876);*/
	/*cudaMemcpy(interiorTZD, iz, sizeof(float) * triangleNum * threadPerTriangleInterior, cudaMemcpyHostToDevice);*/
	/*fill(iz, iz + triangleNum * threadPerTriangleInterior, 6789);*/

	calcBezierCtrlPointInterior<<<blockNumInterior, blockSizeStep1, sizeof(float) * blockSizeStep1 * 14>>>
										(triangleListD, matrixInteriorD,
										 interiorTD, triangleCtrlPointD,
										 activeThreadNumInterior, trianglePerBlockInterior,
										 threadPerTriangleInterior, matrixStartIdxInterior,
										 triangleCtrlPointNum, triangleNum,
										 degree, order[U], order[V], order[W],
										 ctrlPointNum[U], ctrlPointNum[V], ctrlPointNum[W]);
	
	/*cudaMemcpy(ix, interiorTXD, sizeof(float) * triangleNum * threadPerTriangleInterior, cudaMemcpyDeviceToHost);*/
	/*cudaMemcpy(iy, interiorTYD, sizeof(float) * triangleNum * threadPerTriangleInterior, cudaMemcpyDeviceToHost);*/
	/*cudaMemcpy(iz, interiorTZD, sizeof(float) * triangleNum * threadPerTriangleInterior, cudaMemcpyDeviceToHost);*/
	/*fout << "interiorTD:" << endl;*/
	/*for (int i = 0; i < threadPerTriangleInterior; ++i)*/
	/*{*/
		/*for (int j = 0; j < triangleNum; ++j)*/
		/*{*/
			/*fout << "(" << ix[i * triangleNum + j] << ", " << iy[i * triangleNum + j] << ", " << iz[i * triangleNum + j] << ")" << endl;;*/
		/*}*/
		/*fout << endl;*/
	/*}*/

	int delta = 3 * degree * triangleNum * 3;
	/*cout << "delta = " << delta << endl;*/
	stat = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
					   triangleNum * 3, threadPerTriangleInterior, threadPerTriangleInterior,
					   &alpha,
					   interiorTD, triangleNum * 3,
					   matrixInteriorD + matrixStartIdxInterior, threadPerTriangleInterior,
					   &beta,
					   triangleCtrlPointD + delta, triangleNum * 3);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		cout << "triangleCtrlPointD fail!!!!!!!!!!!!!" << endl;
		cout << "stat = " << stat << endl;
		cudaError_t error = cudaGetLastError();
		cout << "CUDA error: " << cudaGetErrorString(error) << endl;
		return;
	}

	/*float *ax = new float[triangleNum * triangleCtrlPointNum];*/
	/*float *ay = new float[triangleNum * triangleCtrlPointNum];*/
	/*float *az = new float[triangleNum * triangleCtrlPointNum];*/
	/*fill(ax, ax + triangleNum * triangleCtrlPointNum, 1234);*/
	/*cudaMemcpy(triangleCtrlPointXD, ax, sizeof(float) * triangleNum * triangleCtrlPointNum, cudaMemcpyHostToDevice);*/
	/*fill(ax, ax + triangleNum * triangleCtrlPointNum, 4321);*/
	/*cudaMemcpy(ax, triangleCtrlPointXD, sizeof(float) * triangleNum * triangleCtrlPointNum, cudaMemcpyDeviceToHost);*/
	/*cudaMemcpy(ay, triangleCtrlPointYD, sizeof(float) * triangleNum * triangleCtrlPointNum, cudaMemcpyDeviceToHost);*/
	/*cudaMemcpy(az, triangleCtrlPointZD, sizeof(float) * triangleNum * triangleCtrlPointNum, cudaMemcpyDeviceToHost);*/
	/*fout << "triangleCtrlPoint:" << endl;*/
	/*for (int i = 0; i < triangleNum; ++i)*/
	/*{*/
		/*fout << i << endl;*/
		/*for (int j = 0; j < triangleCtrlPointNum; ++j)*/
		/*{*/
			/*int idx = j * triangleNum + i;*/
			/*fout << idx << " = (" << ax[idx] << ", " << ay[idx] << ", " << az[idx] << ")" << endl;*/
		/*}*/
		/*fout << endl;*/
	/*}*/

	/*delete []ax;*/
	/*delete []ay;*/
	/*delete []az;*/
}

/************************************************************************************************************/

float *uvwD;
/*static __constant__ float uvwD[8000];*/
int segmentPerEdge, samplePointPerTriangle, samplePointInterior;

float *RD;
int blockSizeCopy = 256, activeThreadNumCopy, blockNumCopy;

void generateUVW(int degree, int samplePointPerEdge)
{
	segmentPerEdge = samplePointPerEdge - 1;
	samplePointPerTriangle = (samplePointPerEdge + 1) * samplePointPerEdge / 2;
	samplePointInterior = (samplePointPerEdge - 3) * (samplePointPerEdge - 2) / 2;
	if (samplePointInterior < 0)
		samplePointInterior = 0;

	activeThreadNumCopy = samplePointPerTriangle * triangleNum;
	blockNumCopy = ceil(static_cast<double>(activeThreadNumCopy) / blockSizeCopy);

	double *a = new double[samplePointPerTriangle * 3];

	int idx = 0;
	for (int i = segmentPerEdge; i >= 0; --i)
	{
		for (int j = segmentPerEdge - i; j >= 0; --j)
		{
			int k = segmentPerEdge - i - j;
			a[idx++] = (double)i / segmentPerEdge;
			a[idx++] = (double)j / segmentPerEdge;
			a[idx++] = (double)k / segmentPerEdge;
		}
	}

	int3 *c = new int3[triangleCtrlPointNum];
	c[0] = make_int3(degree, 0, 0);
	c[1] = make_int3(0, degree, 0);
	c[2] = make_int3(0, 0, degree);
	idx = 3;
	for (int i = 1; i <= degree - 1; ++i)
	{
		c[idx++] = make_int3(0, degree - i, i);
	}
	for (int i = 1; i <= degree - 1; ++i)
	{
		c[idx++] = make_int3(i, 0, degree - i);
	}
	for (int i = 1; i <= degree - 1; ++i)
	{
		c[idx++] = make_int3(degree - i, i, 0);
	}
	/*int iii = idx;*/
	for (int i = degree - 2; i >= 1; --i)
	{
		for (int j = degree - 1 - i; j >= 1; --j)
		{
			c[idx++] = make_int3(i, j, degree - i - j);
		}
	}
	/*for (int i = iii; i < triangleCtrlPointNum; ++i)*/
		/*cout << "ctrl[" << i << "] = (" << c[i].x << ", " << c[i].y << ", " << c[i].z << ")" << endl;*/

	float *b = new float[samplePointPerTriangle * triangleCtrlPointNum];
	idx = 0;
	for (int i = 0; i < samplePointPerTriangle; ++i)
	{
		for (int j = 0; j < triangleCtrlPointNum; ++j)
		{
			b[idx++] = B(a[i * 3], a[i * 3 + 1], a[i * 3 + 2], degree, c[j]);
		}
	}

	/*cudaMemcpyToSymbol(uvwD, b, sizeof(float) * triangleCtrlPointNum * samplePointPerTriangle);*/
	cudaMalloc((void**)&uvwD, sizeof(float) * triangleCtrlPointNum * samplePointPerTriangle * 3);
	cudaMemcpy(uvwD, b, sizeof(float) * triangleCtrlPointNum * samplePointPerTriangle, cudaMemcpyHostToDevice);

	/***********************************************************************************************************************************/

	idx = 0;
	for (int i = 0; i < samplePointPerTriangle; ++i)
	{
		for (int j = 0; j < triangleCtrlPointNum; ++j)
		{
			b[idx++] = factorial(degree) / factorial(c[j].x) / factorial(c[j].y) / factorial(c[j].z)
					 * (c[j].x * power(a[i * 3], c[j].x - 1) * power(a[i * 3 + 1], c[j].y) * power(a[i * 3 + 2], c[j].z)
					 - c[j].z * power(a[i * 3], c[j].x) * power(a[i * 3 + 1], c[j].y) * power(a[i * 3 + 2], c[j].z - 1));
		}
	}
	cudaMemcpy(uvwD + triangleCtrlPointNum * samplePointPerTriangle, b, sizeof(float) * triangleCtrlPointNum * samplePointPerTriangle, cudaMemcpyHostToDevice);

	/***********************************************************************************************************************************/

	idx = 0;
	for (int i = 0; i < samplePointPerTriangle; ++i)
	{
		for (int j = 0; j < triangleCtrlPointNum; ++j)
		{
			b[idx++] = factorial(degree) / factorial(c[j].x) / factorial(c[j].y) / factorial(c[j].z)
					 * (c[j].y * power(a[i * 3], c[j].x) * power(a[i * 3 + 1], c[j].y - 1) * power(a[i * 3 + 2], c[j].z)
					 - c[j].z * power(a[i * 3], c[j].x) * power(a[i * 3 + 1], c[j].y) * power(a[i * 3 + 2], c[j].z - 1));
		}
	}
	cudaMemcpy(uvwD + triangleCtrlPointNum * samplePointPerTriangle * 2, b, sizeof(float) * triangleCtrlPointNum * samplePointPerTriangle, cudaMemcpyHostToDevice);

	delete []a;
	delete []b;
	delete []c;

	cudaMalloc((void**)&RD, sizeof(float) * triangleNum * 3 * samplePointPerTriangle * 3);
}

__global__ void copyRD(float *RD,
					   float *vertexPtrVBO, float *normalPtrVBO, float *texCoordPtrVBO, float *texCoord3DPtrVBO,
					   int activeThreadNumCopy, int triangleNum, int samplePointPerTriangle,
					   TriangleD *triangleListD, bool firstLoad, float maxX, float maxY, float maxZ, int segmentPerEdge)
{
	int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (globalIdx >= activeThreadNumCopy)
		return;

	int localIdx = globalIdx % samplePointPerTriangle;
	int triangleIdx = globalIdx / samplePointPerTriangle;

	vertexPtrVBO[globalIdx * 3] = RD[triangleIdx * samplePointPerTriangle * 3 + localIdx];
	vertexPtrVBO[globalIdx * 3 + 1] = RD[(triangleIdx + triangleNum) * samplePointPerTriangle * 3 + localIdx];
	vertexPtrVBO[globalIdx * 3 + 2] = RD[(triangleIdx + triangleNum * 2) * samplePointPerTriangle * 3 + localIdx];

	// 计算法向
	float ux = RD[triangleIdx * samplePointPerTriangle * 3 + samplePointPerTriangle + localIdx];
	float uy = RD[(triangleIdx + triangleNum) * samplePointPerTriangle * 3 + samplePointPerTriangle + localIdx];
	float uz = RD[(triangleIdx + triangleNum * 2) * samplePointPerTriangle * 3 + samplePointPerTriangle + localIdx];

	float vx = RD[triangleIdx * samplePointPerTriangle * 3 + samplePointPerTriangle * 2 + localIdx];
	float vy = RD[(triangleIdx + triangleNum) * samplePointPerTriangle * 3 + samplePointPerTriangle * 2 + localIdx];
	float vz = RD[(triangleIdx + triangleNum * 2) * samplePointPerTriangle * 3 + samplePointPerTriangle * 2  + localIdx];

	float nx = uy * vz - uz * vy;
	float ny = uz * vx - ux * vz;
	float nz = ux * vy - uy * vx;
	float l = sqrtf(nx * nx + ny * ny + nz * nz);
	nx /= l;
	ny /= l;
	nz /= l;

	normalPtrVBO[globalIdx * 3] = nx;
	normalPtrVBO[globalIdx * 3 + 1] = ny;
	normalPtrVBO[globalIdx * 3 + 2] = nz;

	if (firstLoad)
	{
		int localIdx = globalIdx % samplePointPerTriangle;

		int triangleIdx = globalIdx / samplePointPerTriangle;

		// 计算纹理坐标
		float2 vt0 = triangleListD[triangleIdx].vt[0];
		float2 vt1 = triangleListD[triangleIdx].vt[1];
		float2 vt2 = triangleListD[triangleIdx].vt[2];

		float tempFloorFloat = (sqrt((float)localIdx * 8 + 9) - 3) / 2;
		int floor = rintf(tempFloorFloat);
		if ((floor * 2 + 3) * (floor * 2 + 3) != localIdx * 8 + 9)
			floor = ceilf(tempFloorFloat);
		int room = localIdx - (floor + 1) * floor / 2;

		float3 barycentric_coord;
		barycentric_coord.x = (float)(segmentPerEdge - floor) / segmentPerEdge;
		barycentric_coord.y = (float)(floor - room) / segmentPerEdge;
		barycentric_coord.z = 1.0f - barycentric_coord.x - barycentric_coord.y;

		float u = vt0.x * barycentric_coord.x + vt1.x * barycentric_coord.y + vt2.x * barycentric_coord.z;
		float v = vt0.y * barycentric_coord.x + vt1.y * barycentric_coord.y + vt2.y * barycentric_coord.z;

		texCoordPtrVBO[globalIdx * 2] = u;
		texCoordPtrVBO[globalIdx * 2 + 1] = v;

		// 计算三维纹理坐标
		texCoord3DPtrVBO[globalIdx * 3] = vertexPtrVBO[globalIdx * 3] / maxX * 3;
		texCoord3DPtrVBO[globalIdx * 3 + 1] = vertexPtrVBO[globalIdx * 3 + 1] / maxY * 3;
		texCoord3DPtrVBO[globalIdx * 3 + 2] = vertexPtrVBO[globalIdx * 3 + 2] / maxZ * 3;
	}
}

bool registered = false;
GLuint normalVBO = 0, texCoordVBO = 0, texCoord3DVBO = 0, vertexVBO = 0;
float *normalPtrVBO;							// 读写缓冲区对象所用的指针
float *texCoordPtrVBO;							// 读写缓冲区对象所用的指针
float *texCoord3DPtrVBO;						// 读写缓冲区对象所用的指针
float *vertexPtrVBO;							// 读写缓冲区对象所用的指针

struct cudaGraphicsResource* normalVBO_CUDA;
struct cudaGraphicsResource* texCoordVBO_CUDA;
struct cudaGraphicsResource* texCoord3DVBO_CUDA;
struct cudaGraphicsResource* vertexVBO_CUDA;

void tesslateD(bool firstLoad, float maxX, float maxY, float maxZ)
{
	cudaGraphicsMapResources(1, &normalVBO_CUDA, 0);
	cudaGraphicsMapResources(1, &texCoordVBO_CUDA, 0);
	cudaGraphicsMapResources(1, &texCoord3DVBO_CUDA, 0);
	cudaGraphicsMapResources(1, &vertexVBO_CUDA, 0);
	size_t size2 = sizeof(float) * samplePointPerTriangle * triangleNum * 2;
	size_t size3 = sizeof(float) * samplePointPerTriangle * triangleNum * 3;
	cudaGraphicsResourceGetMappedPointer((void**)&normalPtrVBO, &size3, normalVBO_CUDA);
	cudaGraphicsResourceGetMappedPointer((void**)&texCoordPtrVBO, &size2, texCoordVBO_CUDA);
	cudaGraphicsResourceGetMappedPointer((void**)&texCoord3DPtrVBO, &size3, texCoord3DVBO_CUDA);
	cudaGraphicsResourceGetMappedPointer((void**)&vertexPtrVBO, &size3, vertexVBO_CUDA);

	float alpha = 1.0f, beta = 0.0f;

	cublasStatus_t stat = cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
									  samplePointPerTriangle * 3, triangleNum * 3, triangleCtrlPointNum,
									  &alpha,
									  uvwD, triangleCtrlPointNum,
									  triangleCtrlPointD, triangleNum * 3,
									  &beta,
									  RD, samplePointPerTriangle * 3);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		cout << "RD fail!!!!!!!!!!!!!" << endl;
		cout << "stat = " << stat << endl;
		cudaError_t error = cudaGetLastError();
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		return;
	}
	cudaGraphicsResourceGetMappedPointer((void**)&normalPtrVBO, &size3, normalVBO_CUDA);
	cudaGraphicsResourceGetMappedPointer((void**)&texCoordPtrVBO, &size2, texCoordVBO_CUDA);
	cudaGraphicsResourceGetMappedPointer((void**)&texCoord3DPtrVBO, &size3, texCoord3DVBO_CUDA);
	cudaGraphicsResourceGetMappedPointer((void**)&vertexPtrVBO, &size3, vertexVBO_CUDA);
	copyRD<<<blockNumCopy, blockSizeCopy>>>(RD,
			vertexPtrVBO, normalPtrVBO, texCoordPtrVBO, texCoord3DPtrVBO,
			activeThreadNumCopy, triangleNum, samplePointPerTriangle,
			triangleListD, firstLoad, maxX, maxY, maxZ, segmentPerEdge);

	cudaGraphicsUnmapResources(1, &normalVBO_CUDA, 0);
	cudaGraphicsUnmapResources(1, &texCoordVBO_CUDA, 0);
	cudaGraphicsUnmapResources(1, &texCoord3DVBO_CUDA, 0);
	cudaGraphicsUnmapResources(1, &vertexVBO_CUDA, 0);

	/*delete []ax;*/
	/*delete []ay;*/
	/*delete []az;*/
}

/************************************************************************************************************/

void setGLDevice()
{
	cudaGLSetGLDevice(0);
}

/* 使用缓冲区对象进行 cuda 和 OpenGL 协同工作之前，需要进行一些初始化 */
void regGLBuffer()
{
	if (registered)
	{
		cudaGraphicsUnregisterResource(normalVBO_CUDA);
		cudaGraphicsUnregisterResource(texCoordVBO_CUDA);
		cudaGraphicsUnregisterResource(texCoord3DVBO_CUDA);
		cudaGraphicsUnregisterResource(vertexVBO_CUDA);
		registered = false;
	}
	cudaGraphicsGLRegisterBuffer(&normalVBO_CUDA, normalVBO, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&texCoordVBO_CUDA, texCoordVBO, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&texCoord3DVBO_CUDA, texCoord3DVBO, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&vertexVBO_CUDA, vertexVBO, cudaGraphicsMapFlagsWriteDiscard);
	registered = true;
	cymError = cudaGetLastError();
	cout << __LINE__ << "行，CUDA error: " << cudaGetErrorString(cymError) << endl;
}

/************************************************************************************************************/

void cudaFreeNonZero(void **ptr)
{
	if (*ptr)
	{
		cudaFree(*ptr);
		*ptr = 0;
	}
}

void freeTessMemD()
{
	cudaFreeNonZero((void**)&uvwD);
	cudaFreeNonZero((void**)&RD);
	/*cudaFreeNonZero((void**)&BqD);*/
	/*cudaFreeNonZero((void**)&BBD);*/
}

void freeModelMemD()
{
	cudaFreeNonZero((void**)&vertexParamListD);
	cudaFreeNonZero((void**)&vertexCoordListD);

	cudaFreeNonZero((void**)&triangleListD);
	/*cudaFreeNonZero((void**)&sampleValueD);*/
	/*cudaFreeNonZero((void**)&B_1D);*/
	cudaFreeNonZero((void**)&edgeTD);
	cudaFreeNonZero((void**)&edgeCtrlPointD);

	cudaFreeNonZero((void**)&interiorTD);
	cudaFreeNonZero((void**)&triangleCtrlPointD);

	freeTessMemD();
}

void freeMemD()
{
	if (registered)
	{
		cudaGraphicsUnregisterResource(normalVBO_CUDA);
		cudaGraphicsUnregisterResource(texCoordVBO_CUDA);
		cudaGraphicsUnregisterResource(texCoord3DVBO_CUDA);
		cudaGraphicsUnregisterResource(vertexVBO_CUDA);
		registered = false;
	}
	if (cublas_handle)
	{
		cublasDestroy(cublas_handle);
	}
	cudaFreeNonZero((void**)&matrixEdgeD);
	cudaFreeNonZero((void**)&matrixInteriorD);
	freeModelMemD();
}
