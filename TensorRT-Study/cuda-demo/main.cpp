#include "cuda_runtime.h"  
#include "cublas_v2.h"  

#include <time.h>  
#include <iostream>  
#include "UpsmapleKernel.h"
using namespace std;

int main()
{
	cudaStream_t stream;
	int n = 1024;
	int input_b = 1;
	int input_c = 1;
	int input_w = 4;
	int input_h = 4;
	float scale_h = 2.0;
	float scale_w = 2.0;
	float * inputs = new float[input_b*input_w*input_h];
	for (int i = 0;i < input_b*input_w*input_h;i++) {
		inputs[i] = 1.0;
	}
	int output_h = int(input_h*scale_h);
	int output_w = int(input_w*scale_w);
	float* outputs = new float[input_b*output_h*output_w];

	for (int i = 0;i < input_b*input_w*input_h;i++) {
		cout << i << ":" << inputs[i] << endl;
	}
	
	int status = UpsampleInference(stream, 8*8,input_b, input_c, input_h, input_w, scale_h, scale_w,inputs, outputs);
	for (int i = 0;i < input_b*output_w*output_h;i++) {
		cout << i << ":" << outputs[i] << endl;
	}
	return 0;
}
