#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "reduction_complete.cl"
#define KERNEL_1 "reduction_vector"
#define KERNEL_2 "reduction_complete"
#define KERNEL_3 "vec_mult"

#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include<iostream>
#include<vector>
#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

class clConv
{
public:
    clConv(long verctorSize = 131072);
    ~clConv();

   float convolution(float*v1, float*v2);
   // long convolution(const vector<double> &_v1, const vector<double> &new_v2, vector<double> & out_result);

private:
    long VECTOR_SIZE=131072;
    long ARRAY_SIZE=VECTOR_SIZE/4;
    long verctorSize;
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_kernel vector_kernel, complete_kernel,mult_kernel;
    cl_command_queue queue;

    cl_int i, err;
    size_t local_size, global_size;
    cl_event start_event, end_event;
    /* Data and buffers */
    vector<float> _data;//_v1,_v2;
    //float _data[ARRAY_SIZE];//,_v1[VECTOR_SIZE],_v2[VECTOR_SIZE];//
    float sum, actual_sum;
    cl_mem data_buffer, sum_buffer,v1_buffer,v2_buffer;
    cl_ulong total_time;


    void create_device(cl_device_id &dev,size_t &local_size);
    void build_program(cl_context &ctx, cl_device_id dev, const char* filename,cl_program &program);
    void create_buffer(cl_context &context,vector<float>&data,float*v1,float*v2,cl_mem &data_buffer,cl_mem &sum_buffer,cl_mem &v1_buffer,cl_mem &v2_buffer,cl_int &err);
    void  create_command_queue(cl_context &context,cl_device_id &device,cl_command_queue &queue,cl_int &err);
    void create_kernel(cl_program &program,cl_kernel &vector_kernel,cl_kernel &complete_kernel,cl_kernel &mult_kernel,cl_mem &data_buffer,cl_mem &sum_buffer,cl_mem &v1_buffer,cl_mem &v2_buffer,size_t &local_size,cl_int &err);
    void  enqueue_kernels(cl_command_queue &queue,cl_kernel &mult_kernel,cl_kernel &vector_kernel,cl_kernel &complete_kernel,size_t &global_size,size_t &local_size,cl_event &start_event,cl_event &end_event,cl_ulong &total_time,cl_int &err);
    void  read_buffer_enqueue(cl_command_queue &queue,cl_mem &sum_buffer,float &sum,float &actual_sum,cl_ulong &total_time,cl_int &err);
    void deallocate_resources(cl_context &context,cl_program &program,cl_command_queue &queue,cl_kernel &complete_kernel,cl_kernel &vector_kernel,cl_kernel &mult_kernel,cl_mem &v1_buffer,cl_mem &v2_buffer,cl_mem &data_buffer,cl_mem &sum_buffer,cl_event &end_event,cl_event &start_event);
};
