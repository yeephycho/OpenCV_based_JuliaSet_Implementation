#include <stdio.h>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <omp.h>

#define DIM 2000

using namespace cv;

int main(void){
	class complex {
	public: double re;
	public: double im;
			complex(double a, double b) : re(a), im(b) {};
			double magnitude2(void) { return re*re + im*im; }
			complex operator*(const complex& a) {
				return complex(re * a.re - im * a.im, im*a.re + re * a.im);
			}
			complex operator+(const complex& a) {
				return complex(re + a.re, im + a.im);
			}
	};
	int ppercentage = 0;
	Mat mImage(DIM, DIM, CV_8UC1);
	for (register int i = 0; i < DIM; i++) {
		register int j;
#pragma omp parallel for private(j) num_threads(6)
		for (j = 0; j < DIM; j++) {
			int offset;
			int juliaValue;
			offset = j + i * DIM;
			const double scalex = 1.500;
			const double scaley = 1.500;
			double jx = scalex * (double)(DIM / 2 - i) / (DIM / 2);
			double jy = scaley * (double)(DIM / 2 - j) / (DIM / 2);
			complex c(-0.8, 0.156);
			complex a(jx, jy);
			int x = 0;
			for (x = 0; x < 200; x++) {
				a = a*a + c;
				if (a.magnitude2() > 5000) {
					juliaValue = 1;
					break;
				}
				else
					juliaValue = 0;
			}
			*(mImage.data + offset) = 255 * juliaValue;
		}
		int percentage = (i * 100 / DIM);

		if (ppercentage != percentage) {
			printf("%d%% complete!\n", percentage);
			ppercentage = percentage;
		}
	}
	printf("Complete! Press any key to exit ...\n");
	imwrite("C:\\Users\\huyix\\Desktop\\julia2.jpg", mImage);

	getchar();
	return 0;
}