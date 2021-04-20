#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "matvec.cl"
#define KERNEL_FUNC "matvec_mult"

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#ifdef MAC
#include <OpenCL/cl.h>
#else  
#include <CL/cl.h>
#endif

#include "Fraccional.h"
void create_context(cl_platform_id &platform,cl_device_id &device,cl_context &context,cl_int &err);
void create_program(FILE* program_handle,size_t &program_size,char *program_buffer,cl_program &program,cl_context &context,cl_int &err);
void create_built_program(cl_program &program,cl_device_id &device,char* program_log,size_t &log_size,cl_int &err);
void create_kernel(cl_kernel &kernel,cl_program &program,cl_device_id &device,cl_context &context,cl_mem &mat_buff,float *mat,float *vec,cl_mem &vec_buff,cl_mem &res_buff,cl_int &err);
void create_queue(cl_command_queue &queue,cl_context &context,cl_device_id &device,cl_kernel &kernel,size_t &work_units_per_kernel,cl_int &err);
void read_result(float *result,cl_command_queue &queue,cl_mem &res_buff,cl_int &err);
void test_result(float *result,float* correct);
void deallocate_resources(cl_context &context,cl_program &program,cl_command_queue &queue,cl_kernel &kernel,cl_mem &res_buff,cl_mem &vec_buff,cl_mem &mat_buff);
//void comprobacion(float* pmat, float* pcorr, float *pvec);

int main() {

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
   
   /* Data and buffers */
   float mat[16], vec[4], result[4];
   float correct[4] = {0.0f, 0.0f, 0.0f, 0.0f};
   cl_mem mat_buff, vec_buff, res_buff;
   size_t work_units_per_kernel;

    clConv comp1;
   comprobacion(&mat[0], &correct[0], &vec[0]);
   /* Initialize data to be processed by the kernel */

   create_context(platform,device,context,err);
   create_program(program_handle, program_size,program_buffer,program,context,err);
   create_built_program(program,device,program_log,log_size,err);
   create_kernel(kernel,program,device,context,mat_buff,mat,vec,vec_buff,res_buff,err);
   create_queue(queue,context,device,kernel,work_units_per_kernel,err);
   read_result(result,queue,res_buff,err);
   test_result(result,correct);
   deallocate_resources(context,program,queue,kernel,res_buff,vec_buff,mat_buff);
   return 0;
}


void create_context(cl_platform_id &platform,cl_device_id &device,cl_context &context,cl_int &err){
   /* Identify a platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    if(err < 0) {
       perror("Couldn't find any platforms");
       exit(1);
    }

    /* Access a device */
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if(err < 0) {
       perror("Couldn't find any devices");
       exit(1);
    }

    /* Create the context */
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if(err < 0) {
       perror("Couldn't create a context");
       exit(1);
    }
}
void create_program(FILE* program_handle,size_t &program_size,char *program_buffer,cl_program &program,cl_context &context,cl_int &err){
    /* Read program file and place content into buffer */
    program_handle = fopen(PROGRAM_FILE, "r");
    if(program_handle == NULL) {
       perror("Couldn't find the program file");
       exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    /* Create program from file */
    program = clCreateProgramWithSource(context, 1,
       (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
       perror("Couldn't create the program");
       exit(1);
    }
    free(program_buffer);
}
void create_built_program(cl_program &program,cl_device_id &device,char* program_log,size_t &log_size,cl_int &err){
    /* Build program */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {

       /* Find size of log and print to std output */
       clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
             0, NULL, &log_size);
       program_log = (char*) malloc(log_size + 1);
       program_log[log_size] = '\0';
       clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
             log_size + 1, program_log, NULL);
       printf("%s\n", program_log);
       free(program_log);
       exit(1);
    }
}
void create_kernel(cl_kernel &kernel,cl_program &program,cl_device_id &device,cl_context &context,cl_mem &mat_buff,float *mat,float *vec,cl_mem &vec_buff,cl_mem &res_buff,cl_int &err){
    /* Create kernel for the mat_vec_mult function */
    kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    if(err < 0) {
       perror("Couldn't create the kernel");
       exit(1);
    }

    /* Create CL buffers to hold input and output data */
    mat_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
       CL_MEM_COPY_HOST_PTR, sizeof(float)*16, mat, &err);
    if(err < 0) {
       perror("Couldn't create a buffer object");
       exit(1);
    }
    vec_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
       CL_MEM_COPY_HOST_PTR, sizeof(float)*4, vec, NULL);
    res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
       sizeof(float)*4, NULL, NULL);

    /* Create kernel arguments from the CL buffers */
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
    if(err < 0) {
       perror("Couldn't set the kernel argument");
       exit(1);
    }
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &vec_buff);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &res_buff);
}
void create_queue(cl_command_queue &queue,cl_context &context,cl_device_id &device,cl_kernel &kernel,size_t &work_units_per_kernel,cl_int &err){
    /* Create a CL command queue for the device*/
    queue = clCreateCommandQueue(context, device, 0, &err);
    if(err < 0) {
       perror("Couldn't create the command queue");
       exit(1);
    }

    /* Enqueue the command queue to the device */
    work_units_per_kernel = 4; /* 4 work-units per kernel */
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_units_per_kernel,
       NULL, 0, NULL, NULL);
    if(err < 0) {
       perror("Couldn't enqueue the kernel execution command");
       exit(1);
    }
}
void read_result(float *result,cl_command_queue &queue,cl_mem &res_buff,cl_int &err){
    /* Read the result */
    err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float)*4,
       result, 0, NULL, NULL);
    if(err < 0) {
       perror("Couldn't enqueue the read buffer command");
       exit(1);
    }
}
void test_result(float *result,float* correct){
    /* Test the result */
    if((result[0] == correct[0]) && (result[1] == correct[1])
       && (result[2] == correct[2]) && (result[3] == correct[3])) {
       printf("Matrix-vector multiplication successful.\n");
    }
    else {
       printf("Matrix-vector multiplication unsuccessful.\n");
    }
}
void deallocate_resources(cl_context &context,cl_program &program,cl_command_queue &queue,cl_kernel &kernel,cl_mem &res_buff,cl_mem &vec_buff,cl_mem &mat_buff){
    /* Deallocate resources */
    clReleaseMemObject(mat_buff);
    clReleaseMemObject(vec_buff);
    clReleaseMemObject(res_buff);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
}
/*void comprobacion(float* pmat, float* pcorr, float * pvec)
{
    for(int i=0; i<16; i++) {
        pmat[i] = i * 2.0f;
    }

    for(int i=0; i<4; i++) {
        pvec[i] = i * 3.0f;
        pcorr[0] += pmat[i]    * pvec[i];
        pcorr[1] += pmat[i+4]  * pvec[i];
        pcorr[2] += pmat[i+8]  * pvec[i];
        pcorr[3] += pmat[i+12] * pvec[i];
    }
}*/
