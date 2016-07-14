#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <CL\cl.h>

using namespace std;
using namespace cv;

//*********************** intrinic opencl segment **********************************
#define PROGRAM_FILE "cl_Julia.cl"   // input your kernel file name
#define DIM 20000
#define IMAGENAME "C:\\Users\\huyix\\Desktop\\OpenCLJulia.jpg"

/* Find a GPU and the first available platform */
cl_device_id create_device() {

	cl_platform_id platform;
	cl_device_id dev;
	int err;

	/* Identify a platform */
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err < 0) {
		perror("Couldn't identify a platform");
	}

	/* Access a device */
//	for (int i = 0; i < 3; i++){
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 2, &dev, NULL);
	if (err == CL_DEVICE_NOT_FOUND) {
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	}
	if (err < 0) {
		perror("Couldn't access any devices");
	}
//	}
	return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

	cl_program program;
	FILE *program_handle;
	char *program_buffer, *program_log;
	size_t program_size, log_size;
	int err;

	/* Read program file and place content into buffer */
	program_handle = fopen(filename, "r");
	if (program_handle == NULL) {
		printf("Couldn't find the program file");
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	/* Create program from file */
	program = clCreateProgramWithSource(ctx, 1,
		(const char**)&program_buffer, &program_size, &err);
	if (err < 0) {
		printf("Couldn't create the program");
	}

	/* Build program */
	char options[20];
	sprintf(options, "-D DIM=%d", DIM);
	err = clBuildProgram(program, 0, NULL, options, NULL, NULL);
	if (err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
	}

	free(program_buffer);
	return program;
}

int main(void) {
	Mat mImage(DIM, DIM, CV_8UC1);

	// OpenCL data structures
	cl_device_id device;
	cl_context context;
	cl_program program;
	cl_command_queue queue;
	cl_int err = 0;

	// Create devices
	device = create_device();
	if (err < 0) {
		printf("Couldn't find a device!");
	}
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);  //129ms
	if (err < 0) {
		printf("Couldn't create a context!");
	}

	// Build program from file
	program = build_program(context, device, PROGRAM_FILE);  //5ms
	if (err < 0) {
		printf("Couldn't build the program!");
		getchar();
	};
															 // Create a command queue
	queue = clCreateCommandQueue(context, device, \
		CL_QUEUE_PROFILING_ENABLE, &err);			//CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
	if (err < 0) {
		printf("Couldn't create a command queue!");
	};

	cl_mem clImage;
	clImage = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,\
		mImage.rows * mImage.cols*sizeof(uchar), mImage.data, &err);
	if (err < 0) {
		printf("Couldn't create buffer!");
	};

	cl_kernel cl_Julia;
	cl_Julia = clCreateKernel(program, "julia", &err);
	if (err < 0) {
		printf("Couldn't create kernel!");
		getchar();
	};

	uint g_col = ((mImage.cols + 12) / 13) * 13;
	uint g_row = ((mImage.rows + 12) / 13) * 13;
	size_t globalSize[] = { g_col, g_row };
	size_t localSize[] = { 13, 13 }; // GTX-970 has 13 cuda core

	int col = mImage.cols;
	int row = mImage.rows;

	err = clSetKernelArg(cl_Julia, 0, sizeof(cl_mem), &clImage);
	err = clSetKernelArg(cl_Julia, 1, sizeof(int), &col);
	err = clSetKernelArg(cl_Julia, 2, sizeof(int), &row);
	if (err < 0) {
		printf("Couldn't set arguments!");
	}

	err |= clEnqueueNDRangeKernel(queue, cl_Julia, 2, NULL, \
		globalSize, localSize, 0, NULL, NULL);       //10ms
	if (err < 0) {
		printf("Couldn't enqueue kernel cl_Julia");
	}

	err = clEnqueueReadBuffer(queue, clImage, CL_TRUE, 0,			\
			mImage.rows * mImage.cols * sizeof(uchar), mImage.data, 0, NULL, NULL);
	if (err < 0) {
		printf("Couldn't read buffer with err");
	}

	imwrite(IMAGENAME, mImage);

	clReleaseMemObject(clImage);
	clReleaseKernel(cl_Julia);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);

	return 0;
}
