#ifndef FRACCIONAL
#define FRACCIONAL

#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "matvec.cl"
#define KERNEL_FUNC "matvec_mult"

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

class clConv
{
public:
    clConv(long new_verctorSize = 1000);
    ~clConv();

    long convolution(const vector<double> &new_v1, const vector<double> &new_v2, vector<double> & out_result);

private:

    long verctorSize;
    vector<double> v1;
    vector<double> v2;
    vector<double> result;

    /* Host/device data structures */
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int err;

    /* Program/kernel data structures */
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    cl_kernel kernel;

    cl_mem mat_buff, vec_buff, res_buff;
    size_t work_units_per_kernel;


    void create_context(cl_platform_id &platform,cl_device_id &device,cl_context &context,cl_int &err);
    void create_program(FILE* program_handle,size_t &program_size,char *program_buffer,cl_program &program,cl_context &context,cl_int &err);
    void create_built_program(cl_program &program,cl_device_id &device,char* program_log,size_t &log_size,cl_int &err);
    void create_kernel(cl_kernel &kernel,cl_program &program,cl_device_id &device,cl_context &context,cl_mem &mat_buff,double *mat,double *vec,cl_mem &vec_buff,cl_mem &res_buff,cl_int &err);
    void create_queue(cl_command_queue &queue,cl_context &context,cl_device_id &device,cl_kernel &kernel,size_t &work_units_per_kernel,cl_int &err);
    void read_result(double *result,cl_command_queue &queue,cl_mem &res_buff,cl_int &err);
    void test_result(float *result,float* correct);
    void deallocate_resources(cl_context &context,cl_program &program,cl_command_queue &queue,cl_kernel &kernel,cl_mem &res_buff,cl_mem &vec_buff,cl_mem &mat_buff);


};

void comprobacion(float* pmat, float* pcorr, float * pvec);
#endif
