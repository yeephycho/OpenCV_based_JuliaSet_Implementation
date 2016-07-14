//#define DIM 5000 
struct clComplex {
	double r;
	double i;
};
__kernel void julia(__global uchar *data, int col, int row) {
	int idx = get_global_id(0);
	int idy = get_global_id(1);
	if (idx<DIM && idx<DIM) {
		size_t offset = idy + idx * DIM;
		const double scalex = 1.500;
		const double scaley = 1.500;
		double jx = scalex * (double)(DIM / 2 - idx) / (DIM / 2);
		double jy = scaley * (double)(DIM / 2 - idy) / (DIM / 2);
		int juliaValue;
		struct clComplex c;
		c.r = -0.8;
		c.i = 0.156;
		struct clComplex a;
		struct clComplex b;
		a.r = jx;
		a.i = jy;
		int i;
		for (i = 0; i<200; i++) {
			b.r = a.r;
			b.i = a.i;
			a.r = b.r * b.r - b.i * b.i + c.r;
			a.i = b.i * b.r + b.r * b.i + c.i;
			if (a.r * a.r + a.i * a.i > 5000) {
				juliaValue = 1;
				break;
			}
			else
				juliaValue = 0;
		}
		data[offset] = 255 * juliaValue;
	}
}
// TODO: Add OpenCL kernel code here.