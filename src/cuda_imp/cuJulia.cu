#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

using namespace cv;
#define DIM 20000

struct cuComplex {
	double r;
	double i;
};

__device__ void julia(int x, int y, int& juliaValue) {
	const double scale = 1.5;
	double jx = scale * (double)(DIM / 2 - x) / (DIM / 2);
	double jy = scale * (double)(DIM / 2 - y) / (DIM / 2);
	cuComplex c;
	c.r = -0.8;
	c.i = 0.156;
	cuComplex a;
	cuComplex b;
	a.r = jx;
	a.i = jy;

	int i = 0;
	for (i = 0; i<200; i++) {
		b.r = a.r;
		b.i = a.i;
		a.r = b.r * b.r - b.i * b.i + c.r;
		a.i = b.i * b.r + b.r * b.i + c.i;
		if (a.r * a.r + a.i * a.i > 5000){
			juliaValue = 1;
			break;
		}
		else
			juliaValue = 0;
	}
}

__global__ void generateJuliaImage(unsigned char *dataPtr) {
	// map from threadIdx/BlockIdx to pixel position
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;
	// now calculate the value at that position
	int juliaValue;
	julia(x, y, juliaValue);
	dataPtr[offset] = 255 * juliaValue;
}

int main(void)
{
	Mat cuImage(DIM, DIM, CV_8UC1);

	unsigned char *dev_cuImage;
	dim3 grid(DIM, DIM);
	int err = cudaMalloc((void**)&dev_cuImage, DIM*DIM*sizeof(char));
	generateJuliaImage<<<grid, 1 >>>(dev_mImage);
	err = cudaMemcpy(cuImage.data, dev_cuImage, DIM*DIM*sizeof(char), cudaMemcpyDeviceToHost);
	err = cudaFree(dev_cuImage);

	printf("finished, press any key to exit ...\n");
	imwrite("C:\\Users\\huyix\\Desktop\\cudaJulia.jpg", cuImage);
	
    return 0;
}
