
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<iostream>
#define _USE_MATH_DEFINES
#include "math.h"

#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>

#define MAXITER 3000
#define centerPointX 0 // -0.761574
#define centerPointY 0 // -0.0847596
#define width 1280
#define height 640
#define aspectRatioWidth 2
#define aspectRatioHeight 1
#define waitTime 1

#define BLOCK_SIZE 32

double *zoomGpu = 0;
uchar *pixels = 0;
cudaError_t cudaStatus;
uchar *pixelsHost;

using namespace cv;
using namespace std;

__device__ inline double mapN(double x, double in_min, double in_max, double out_min, double out_max)
{
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

__device__ void setColor(uchar *pixel, int n)
{
	/*double brightness = mapN(n, 0, MAXITER, 0, 1);
	brightness = mapN(sqrt(brightness), 0, 1, 0, 255);
	if (n == MAXITER)
	{
		brightness = 0;
	}

	*(pixel) = brightness;
	*(pixel + 1) = brightness;
	*(pixel + 2) = mapN(brightness * brightness, 0, 65025, 0, 255);*/

	double t = (double)n / (double)MAXITER;

	// Use smooth polynomials for r, g, b
	int r = (int)(8.5*(1 - t)*(1 - t)*(1 - t)*t * 255); 
	int g = (int)(15 * (1 - t)*(1 - t)*t*t * 255);
	int b = (int)(9 * (1 - t)*t*t*t * 255);
	pixel[0] = b;
	pixel[1] = g;
	pixel[2] = r;

	//int N = 256; // colors per element
	//int N3 = N * N * N;
	//// map n on the 0..1 interval (real numbers)
	//double t = (double)n / (double)MAXITER;
	//// expand n on the 0 .. 256^3 interval (integers)
	//n = (int)(t * (double)N3);

	//int b = n / (N * N);
	//int nn = n - b * N * N;
	//int r = nn / N;
	//int g = nn - r * N;
	//pixel[0] = b;
	//pixel[1] = g;
	//pixel[2] = r;
}

__global__ void getPixelValue(double *zoomGpu, uchar *pixels)
{
	int blocksPerRow = width / BLOCK_SIZE;
	int row = blockIdx.x / blocksPerRow;
	int col = (blockIdx.x % blocksPerRow) * BLOCK_SIZE + threadIdx.x;

	int pos = row * width * 3 + col * 3; // pixel position for current thread in array

	double zoomLevelX = *zoomGpu;
	double zoomLevelY = *zoomGpu / aspectRatioWidth * aspectRatioHeight;
	double a = mapN(col, 0, width, centerPointX - zoomLevelX, centerPointX + zoomLevelX);
	double b = mapN(row, 0, height, centerPointY - zoomLevelY, centerPointY + zoomLevelY);

	double ca = -0.8; // a;
	double cb = 0.156; // b;

	int n = 0;

	while (n < MAXITER)
	{
		double aa = a * a - b * b;
		double bb = 2 * a * b;
		a = aa + ca;
		b = bb + cb;
		if (abs(a + b) > 2) {
			break;
		}
		n++;
	}

	setColor(&pixels[pos], n);
}

cudaError_t initCuda()
{
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error1;
	}

	cudaStatus = cudaMalloc((void**)&pixels, width * height * 3 * sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error2;
	}

	cudaStatus = cudaMalloc((void**)&zoomGpu, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error3;
	}

	pixelsHost = (uchar*)malloc(width * height * 3 * sizeof(uchar));

	return cudaStatus;

Error3:
	cudaFree(zoomGpu);
Error2:
	cudaFree(pixels);
Error1:
	cudaFree(zoomGpu);
	return cudaStatus;
}

cudaError_t getCudaImage(Mat *img, double *zoomHost)
{
	cudaStatus = cudaMemcpy(zoomGpu, zoomHost, sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	getPixelValue <<<width * height / BLOCK_SIZE, BLOCK_SIZE>>> (zoomGpu, pixels);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "getPixelValue launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getPixelValue!\n", cudaStatus);
		goto Error;
	}

	// send pixels to host
	cudaStatus = cudaMemcpy(pixelsHost, pixels, width * height * 3 * sizeof(uchar), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// convert *pixels to Mat
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			int pos = y * width * 3 + x * 3; // pixel position for current thread in array
			Vec3b &color = img->at<Vec3b>(y, x);
			color[0] = pixelsHost[pos];
			color[1] = pixelsHost[pos + 1];
			color[2] = pixelsHost[pos + 2];
		}
	}

	return cudaStatus;

Error:
	cudaFree(zoomGpu);
	return cudaStatus;
}

int main()
{
	double zoomHost = 1.8;
	double minZoomLevel = 0.00005;
	//double minZoomLevel = 0.05;
	Mat img(height, width, CV_8UC3, Scalar(51, 51, 51));

	initCuda();

	for (zoomHost = 2; zoomHost > minZoomLevel; zoomHost -= zoomHost * 0.05) {
		cudaError_t cudaStatus = getCudaImage(&img, &zoomHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "getCudaImage failed!");
			return 1;
		}
		imshow("image", img);
		waitKey(waitTime);
	}
	waitKey(0);

	cudaFree(zoomGpu);
	cudaFree(pixels);
	free(pixelsHost);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
}