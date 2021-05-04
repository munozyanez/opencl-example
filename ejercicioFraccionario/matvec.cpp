#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "matvec.cl"
#define KERNEL_FUNC "matvec_mult"
#define PROGRAM_FILE2 "sumvec.cl"
#define KERNEL_FUNC2 "matvec_sum"
#define SIZE_V 1024
#define SIZE_R (SIZE_V/4)
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <iostream>
#include <ctime>
#include <chrono>



#include "Fraccional.h"
//void comprobacion(float* pmat, float* pcorr, float *pvec);







int main() {
            unsigned t0, t1,t3,t4;

    /* Host/device data structures */
       cl_platform_id platform;
       cl_device_id device;
       cl_context context;
       cl_command_queue queue;
       cl_int i, err;

       /* Program/kernel data structures */
       cl_program program,program2;
       FILE *program_handle,*program_handle2;
       char *program_buffer, *program_buffer2,*program_log,*program_log2;
       size_t program_size,program_size2, log_size,log_size2;
       cl_kernel kernel,kernel2;

       /* Data and buffers */
       float mat[SIZE_V], vec[SIZE_V], result[SIZE_R];
       float correct;
       cl_mem mat_buff, vec_buff, res_buff;
       size_t work_units_per_kernel;

       /* Initialize data to be processed by the kernel */
       for(i=0; i<SIZE_V; i++) {
          mat[i] = i * 3.0f;
       }
       for(i=0; i<SIZE_V; i++) {
          vec[i] = i * 1.0f;
       }

        for(i=0; i<SIZE_V; i++) {
          correct+= mat[i] * vec[i];
        }

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

       /* Create kernel for the mat_vec_mult function */
       kernel = clCreateKernel(program, KERNEL_FUNC, &err);
       if(err < 0) {
          perror("Couldn't create the kernel");
          exit(1);
       }

       /* Create CL buffers to hold input and output data */
       mat_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
          CL_MEM_COPY_HOST_PTR, sizeof(float)*SIZE_V, mat, &err);
       if(err < 0) {
          perror("Couldn't create a buffer object");
          exit(1);
       }
       vec_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
          CL_MEM_COPY_HOST_PTR, sizeof(float)*SIZE_V, vec, NULL);
       res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
          sizeof(float)*SIZE_R, NULL, NULL);

       /* Create kernel arguments from the CL buffers */
       err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
       if(err < 0) {
          perror("Couldn't set the kernel argument");
          exit(1);
       }
       clSetKernelArg(kernel, 1, sizeof(cl_mem), &vec_buff);
       clSetKernelArg(kernel, 2, sizeof(cl_mem), &res_buff);




       /* Create a CL command queue for the device*/
       queue = clCreateCommandQueue(context, device, 0, &err);
       if(err < 0) {
          perror("Couldn't create the command queue");
          exit(1);
       }


        chrono::system_clock::time_point start = std::chrono::system_clock::now();
        t0=clock();


       /* Enqueue the command queue to the device */
       work_units_per_kernel =SIZE_R; //SIZE_R; /* 4 work-units per kernel */

       err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_units_per_kernel,
          NULL, 0, NULL, NULL);
      // err = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, &work_units_per_kernel,
        //  NULL, 0, NULL, NULL);
       if(err < 0) {
          perror("Couldn't enqueue the kernel execution command");
          exit(1);
       }
       chrono::system_clock::time_point finish = std::chrono::system_clock::now();
       chrono::nanoseconds elapsedNanoseconds = finish.time_since_epoch() - start.time_since_epoch();
       double tiempo1bucle = elapsedNanoseconds.count()/1000;
       double tiempototal = elapsedNanoseconds.count();
       cout << "Tiempo 1 bucle ms: " << (tiempo1bucle/1000000) << endl;
       cout << "Tiempo total: ms " << (tiempototal/1000000) << endl;

       t1 = clock();
       double time = (double(t1-t0)/CLOCKS_PER_SEC);
       cout << "Execution Time OpenCl: " << time << endl;



       /* Read the result */
       err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float)*SIZE_R,
          result, 0, NULL, NULL);
       if(err < 0) {
          perror("Couldn't enqueue the read buffer command");
          exit(1);
       }



   /* Test the result */
   /*if((result[0] == correct[0]) && (result[1] == correct[1])
      && (result[2] == correct[2]) && (result[3] == correct[3])) {
      printf("Matrix-vector multiplication successful.\n");
   }
   else {
      printf("Matrix-vector multiplication unsuccessful.\n");
   }*/
   cout<<"vec:"<<endl;
   for(int i=SIZE_V-16;i<SIZE_V;i++){
         if(i==SIZE_V-16)cout<<"[";
   cout<<vec[i]<<" ";
    if(i==SIZE_V-1)cout<<"]"<<endl;
   }
   cout<<"mat:"<<endl<<"|";
   for(int i=SIZE_V-16;i<SIZE_V;i++){
  /* cout<<mat[i-1]<<" ";
   if(i%4==0){cout<<"|"<<endl<<"|";}*/
   if(i==SIZE_V-16)cout<<"[";
cout<<mat[i]<<" ";
if(i==SIZE_V-1)cout<<"]"<<endl;
   }
   cout<<"result:"<<endl;
   for(int i=SIZE_R-8;i<SIZE_R;i++){
       if(i==SIZE_R-8)cout<<"[";
   cout<<result[i]<<" ";
       if(i==SIZE_R-1)cout<<"]"<<endl;
   }


   /* Deallocate resources */
   clReleaseMemObject(mat_buff);
   clReleaseMemObject(vec_buff);
   clReleaseMemObject(res_buff);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

   return 0;
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
