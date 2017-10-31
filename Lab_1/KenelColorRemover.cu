#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#include <algorithm>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\types_c.h>
#include <opencv2\imgproc\imgproc.hpp>


__global__ void KernelColorRemover(
	unsigned char* 		outputImage,					// Return value: blurred rgba image with alpha set to 255 or opaque.
	unsigned char* 		originalImage,
	unsigned char 		minHue,
	unsigned char		maxHue,
	unsigned char       minSat,
	unsigned char       maxSat,
	unsigned char		minVal,
	unsigned char		maxVal,
	int					rows,							// image size: number of rows
	int					cols							// image size: number of columns
	)
{
	// Index of the thread
	int p = (blockIdx.x * blockDim.x + threadIdx.x) * 3;

	// Not out of bounds
	if (p > cols * rows * 3)
	{
		return;
	}

	float r, g, b;
	unsigned char ur, ug, ub;
	float h, s, v;

	b = ub = originalImage[p];
	g = ug = originalImage[p + 1];
	r = ur = originalImage[p + 2];

	// -- BEGIN RGB -> HSV -- 
	float cmax = max(ur, max(ug, ub));
	float cmin = min(ur, min(ug, ub));
	float diff = cmax - cmin;

	v = cmax;

	if (v == 0.0f) // Completely black
	{
		outputImage[p] = 0;// originalImage[p];
		outputImage[p + 1] = 0;// originalImage[p + 1];
		outputImage[p + 2] = 0;//originalImage[p + 2];
		return;
	}
	else
	{
		s = diff / v;
		if (diff < 0.255f) { // grey 
			h = 0.0f;
			//remove to allow grey filtering
			outputImage[p] = 0;//originalImage[p];
			outputImage[p + 1] = 0;//originalImage[p + 1];
			outputImage[p + 2] = 0;//originalImage[p + 2];
			return;

		}
		else
		{
			if (cmax == r)
			{
				h = (g - b) / diff;
				if (h < 0.0f)
				{
					h += 6.f;
				}
			}
			else
			{
				if (cmax == g)
				{
					h = (2.f + (b - r) / diff);
				}
				else
				{
					h = (4.f + (r - g) / diff);
				}
			}
		}
	}

	// -- END RGB -> HSV --
	
	// h = [0, 6[ -> [0, 255[
	h = h * 255.f / 6.f;

	if (h >= minHue && h <= maxHue && v >= minVal && v <= maxVal && s * 255 >= minSat && s * 255 <= maxSat) // Ignore very very white colors
	{
		// Set the color from an int!
		outputImage[p] = 0;// h;
		outputImage[p + 1] = 0;// s * 255;
		outputImage[p + 2] = 0;// v;
	}
	else
	{
		outputImage[p] = 255;// h;
		outputImage[p + 1] = 255;// s * 255;
		outputImage[p + 2] = 255;// v;
	}
}

__global__ void KernelMorph(
	unsigned char* out,
	unsigned char* in,
	unsigned char valueToFind,
	int closeSize,
	int height,
	int width
	)
{
	int p = (blockIdx.x * blockDim.x + threadIdx.x);

	// Not out of bounds
	if (p > width * height) {
		return;
	}
	int half = closeSize / 2;
	bool found = false;

	int r = p / width;
	int c = p % width;

	// Morph

	int x = -half;
	while (x <= half && !found) {
		int y = -half;
		while (y <= half && !found) {
			int		w = min(max((c + x), 0), width-1);
			int		h = min(max((r + y), 0), height-1);

			int		idx = w*3 + h*width*3;						// current pixel index
			int count = 0;
			for (int i = 0; i < 3; i++) {
				unsigned char pixel = in[idx + i];
				if (pixel == valueToFind) {
					count++;
				}
			}
			if (count == 3) {
				found = true;
			}
			y++;
		}
		x++;
	}
	if (found) {
		out[p * 3] = valueToFind;
		out[p * 3 + 1] = valueToFind;
		out[p * 3 + 2] = valueToFind;
	}
	else {
		out[p * 3] = in[p*3];// valueToFind;
		out[p * 3 + 1] = in[p*3+1];// valueToFind;
		out[p * 3 + 2] = in[p*3+2];// valueToFind;
	}
}

__global__ void KernelMorphEx(
	unsigned char* out,
	unsigned char* in,
	int valueToFind,
	unsigned char* mask,
	int maskSize,
	int height,
	int width
	)
{
	int p = (blockIdx.x * blockDim.x + threadIdx.x);

	// Not out of bounds
	if (p > width * height) {
		return;
	}
	int half = maskSize / 2;
	bool found = false;

	int r = p / width;
	int c = p % width;

	// Morph

	int x = -half;
	while (x <= half && !found) 
	{
		int y = -half;
		while (y <= half && !found) 
		{
			if (mask[x+half+(y+half)*maskSize]!=0)
			{
				int		w = min(max((c + x), 0), width - 1);
				int		h = min(max((r + y), 0), height - 1);

				int		idx = w * 3 + h*width * 3;						// current pixel index
				int count = 0;
				for (int i = 0; i < 3; i++) 
				{
					unsigned char pixel = in[idx + i];
					if (pixel == valueToFind) 
					{
						count++;
					}
				}
				if (count == 3) {
					found = true;
				}
			}			
			y += maskSize / 10;
		}
		x += maskSize / 10;
	}
	if (found) {
		out[p * 3] = valueToFind;
		out[p * 3 + 1] = valueToFind;
		out[p * 3 + 2] = valueToFind;
	}
	else {
		out[p * 3] = in[p * 3];// valueToFind;
		out[p * 3 + 1] = in[p * 3 + 1];// valueToFind;
		out[p * 3 + 2] = in[p * 3 + 2];// valueToFind;
	}

}


extern "C" cudaError_t RunDilate(
	unsigned char* imageIn, unsigned char* imageOut, int width, int height, unsigned char* mask, int maskSize,
	int minHue, int maxHue, int minSat, int maxSat, int minVal, int maxVal, int closeSize
	)
{
	cudaError hr = cudaSuccess;

	const int numPixels = width * height;
	const int size = numPixels * 3 * sizeof(unsigned char);

	const int numMaskPixels = maskSize * maskSize;
	const int maskSizeBytes = numMaskPixels * sizeof(unsigned char);

	static const int BLOCK_WIDTH = 1024;						// threads per block; because we are setting 2-dimensional block, the total number of threads is 32^2, or 1024
																// 1024 is the maximum number of threads per block for modern GPUs.

	cv::Mat imgIn = cv::Mat(cv::Size(width, height), CV_8UC3, imageIn);

	cv::Mat testIn;
	cv::Mat testOut;


	unsigned char *devImgIn, *devImgOut, *devMask;
	hr = cudaMalloc(&devImgOut, size);
	if (hr != cudaSuccess) {
		return hr;
	}
	hr = cudaMalloc(&devImgIn, size);
	if (hr != cudaSuccess) {
		return hr;
	}
	hr = cudaMalloc(&devMask, maskSizeBytes);
	if (hr != cudaSuccess) {
		return hr;
	}
	hr = cudaMemcpy(devImgIn, imageIn, size, cudaMemcpyHostToDevice);
	if (hr != cudaSuccess) {
		return hr;
	}
	hr = cudaMemcpy(devMask, mask, maskSizeBytes, cudaMemcpyHostToDevice);
	if (hr != cudaSuccess) {
		return hr;
	}

	int numBlocks;
	if ((width * height) % BLOCK_WIDTH == 0) {
		numBlocks = height * width / BLOCK_WIDTH;
	}
	else {
		numBlocks = height * width / BLOCK_WIDTH - (height * width) % BLOCK_WIDTH + BLOCK_WIDTH;
	}

	KernelColorRemover <<<numBlocks, BLOCK_WIDTH >>>(
		devImgOut,
		devImgIn,
		minHue + 30,
		maxHue + 30,
		minSat,
		maxSat,
		minVal,
		maxVal, 
		height,
		width
	);

	hr = cudaGetLastError();
	if (hr != cudaSuccess) {
		fprintf(stderr, "Sync failed: %s\n", cudaGetErrorString(hr));
		return hr;
	}
	KernelMorph <<<numBlocks, BLOCK_WIDTH >>>(
		devImgIn,
		devImgOut,
		255,
		closeSize,
		height,
		width
	);

	hr = cudaGetLastError();
	if (hr != cudaSuccess) {
		fprintf(stderr, "Sync failed: %s\n", cudaGetErrorString(hr));
		return hr;
	}

	KernelMorph << <numBlocks, BLOCK_WIDTH >> >(
		devImgOut,
		devImgIn,
		0,
		closeSize,
		height,
		width
	);

	hr = cudaGetLastError();
	if (hr != cudaSuccess) {
		fprintf(stderr, "Sync failed: %s\n", cudaGetErrorString(hr));
		return hr;
	}

	KernelMorphEx << <numBlocks, BLOCK_WIDTH >> >(
		devImgIn,
		devImgOut,
		255,
		devMask,
		maskSize,
		height,
		width
	);

	hr = cudaGetLastError();
	if (hr != cudaSuccess) {
		fprintf(stderr, "Morph Dilate 2 launch failed: %s\n", cudaGetErrorString(hr));
		return hr;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	hr = cudaDeviceSynchronize();
	if (hr != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", hr);
	}

	hr = cudaGetLastError();
	if (hr != cudaSuccess) {
		fprintf(stderr, "Sync failed: %s\n", cudaGetErrorString(hr));
		return hr;
	}

	hr = cudaMemcpy(imageOut, devImgOut, size, cudaMemcpyDeviceToHost);
	if (hr != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", hr);
	}
	testOut = cv::Mat(cv::Size(width, height), CV_8UC3, imageOut);

	hr = cudaMemcpy(imageOut, devImgIn, size, cudaMemcpyDeviceToHost);
	if (hr != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", hr);
	}
	testIn = cv::Mat(cv::Size(width, height), CV_8UC3, imageIn);


	return hr;
}
