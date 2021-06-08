#include "Fraccional.h"

clConv::clConv(long verctorSize) : VECTOR_SIZE(verctorSize)
{
    /*Initialize the data size of type float4*/
     ARRAY_SIZE=VECTOR_SIZE/4;

    /* Create device and determine local size */
    create_device(device,local_size);

    /* Build program */
    build_program(context,device,PROGRAM_FILE,program);

}

clConv::~clConv()
{
    /* Deallocate resources */
  deallocate_resources(context,program,queue,complete_kernel,vector_kernel,mult_kernel,v1_buffer,v2_buffer,data_buffer,sum_buffer,end_event,start_event);

}

float clConv::convolution(float *v1, float*v2)
{

    /***se eliminara despues****/
    for(int i=0; i<VECTOR_SIZE; i++) {
       actual_sum+=v1[i]*v2[i];
    }
    /* Create data buffer */
    create_buffer(context,_data,v1,v2,data_buffer,sum_buffer,v1_buffer,v2_buffer,err);

    /* Create a command queue */
    create_command_queue(context,device,queue,err);

    /* Create kernels */
    create_kernel(program,vector_kernel,complete_kernel,mult_kernel,data_buffer,sum_buffer,v1_buffer,v2_buffer,local_size,err);

    /* Enqueue kernels */
    enqueue_kernels(queue,mult_kernel,vector_kernel,complete_kernel,global_size,local_size,start_event,end_event,total_time,err);

    /* Read the result */
     read_buffer_enqueue(queue,sum_buffer,sum,actual_sum,total_time,err);

    return sum;

}
void clConv::create_device(cl_device_id &dev,size_t &local_size) {

   cl_platform_id platform;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   }

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);
   }
   err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE,
         sizeof(local_size), &local_size, NULL);
/*we must ensure that the number of input data is at least more than 16 times larger than the size of the workgroups*/
   if(local_size>VECTOR_SIZE/16){
       local_size=VECTOR_SIZE/16;
   }

   std::cout<<" Local_Size:"<<local_size<<std::endl;
   if(err < 0) {
      perror("Couldn't obtain device information");
      exit(1);
   }
}
void clConv::build_program(cl_context &ctx, cl_device_id dev, const char* filename,cl_program &program) {

   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;
   /* Create a context */
   ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);
   }
   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
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
   program = clCreateProgramWithSource(ctx, 1,
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
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }
}
void clConv::create_buffer(cl_context &context,vector<float> &data,float*v1,float*v2,cl_mem &data_buffer,cl_mem &sum_buffer,cl_mem &v1_buffer,cl_mem &v2_buffer,cl_int &err){
    data_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
          CL_MEM_USE_HOST_PTR, ARRAY_SIZE * sizeof(float), &data, &err);
    /*we create the buffer for sum*/
    sum_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
          sizeof(float), NULL, &err);
   /* we create the buffer for v1 and v2*/
     v1_buffer= clCreateBuffer(context, CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR, VECTOR_SIZE *sizeof(float),v1, &err);
     v2_buffer= clCreateBuffer(context, CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR,VECTOR_SIZE *sizeof(float),v2, &err);
    if(err < 0) {
       perror("Couldn't create a buffer");
       exit(1);
    };
}
void clConv::create_command_queue(cl_context &context,cl_device_id &device,cl_command_queue &queue,cl_int &err){
    queue = clCreateCommandQueue(context, device,
          CL_QUEUE_PROFILING_ENABLE, &err);
    if(err < 0) {
       perror("Couldn't create a command queue");
       exit(1);
    }
}
void clConv::create_kernel(cl_program &program,cl_kernel &vector_kernel,cl_kernel &complete_kernel,cl_kernel &mult_kernel,cl_mem &data_buffer,cl_mem &sum_buffer,cl_mem &v1_buffer,cl_mem &v2_buffer,size_t &local_size,cl_int &err){
    vector_kernel = clCreateKernel(program, KERNEL_1, &err);
    complete_kernel = clCreateKernel(program, KERNEL_2, &err);
    mult_kernel=clCreateKernel(program, KERNEL_3, &err);
    if(err < 0) {
       perror("Couldn't create a kernel");
       exit(1);
    };

    /* Set arguments for vector kernel */
    err = clSetKernelArg(vector_kernel, 0, sizeof(cl_mem), &data_buffer);
    err |= clSetKernelArg(vector_kernel, 1, local_size * 4 * sizeof(float), NULL);

    /* Set arguments for complete kernel */
    err = clSetKernelArg(complete_kernel, 0, sizeof(cl_mem), &data_buffer);
    err |= clSetKernelArg(complete_kernel, 1, local_size * 4 * sizeof(float), NULL);
    err |= clSetKernelArg(complete_kernel, 2, sizeof(cl_mem), &sum_buffer);

    /* Set arguments for multiplication vector kernel */
    err = clSetKernelArg(mult_kernel, 0, sizeof(cl_mem), &v1_buffer);
    err |= clSetKernelArg(mult_kernel, 1, sizeof(cl_mem), &v1_buffer);
    err |= clSetKernelArg(mult_kernel, 2, sizeof(cl_mem), &data_buffer);
    if(err < 0) {
       perror("Couldn't create a kernel argument");
       exit(1);
    }

}
void clConv::enqueue_kernels(cl_command_queue &queue,cl_kernel &mult_kernel,cl_kernel &vector_kernel,cl_kernel &complete_kernel,size_t &global_size,size_t &local_size,cl_event &start_event,cl_event &end_event,cl_ulong &total_time,cl_int &err){

     cl_ulong time_start, time_end;
     global_size = ARRAY_SIZE;
     err = clEnqueueNDRangeKernel(queue, mult_kernel, 1, NULL, &global_size,NULL, 0, NULL,  &start_event);
   /*now global_size is 4 time smaller than before */
     global_size = ARRAY_SIZE/4;
      do{
        err = clEnqueueNDRangeKernel(queue, vector_kernel, 1, NULL, &global_size,
              &local_size, 0, NULL, NULL);
        printf("Global size = %zu\n", global_size);
        if(err < 0) {
           perror("Couldn't enqueue the kernel");
           exit(1);
        }
        global_size = global_size/local_size;
     }while(global_size/local_size > local_size);

     err = clEnqueueNDRangeKernel(queue, complete_kernel, 1, NULL, &global_size,
           NULL, 0, NULL, &end_event);
     printf("Global size = %zu\n", global_size);

     /* Finish processing the queue and get profiling information */
     clFinish(queue);
     clGetEventProfilingInfo(start_event, CL_PROFILING_COMMAND_START,
           sizeof(time_start), &time_start, NULL);
     clGetEventProfilingInfo(end_event, CL_PROFILING_COMMAND_END,
           sizeof(time_end), &time_end, NULL);
     total_time = time_end - time_start;

}
void clConv::read_buffer_enqueue(cl_command_queue &queue,cl_mem &sum_buffer,float &sum,float &actual_sum,cl_ulong &total_time,cl_int &err){

    err = clEnqueueReadBuffer(queue, sum_buffer, CL_TRUE, 0,
       sizeof(float), &sum, 0, NULL, NULL);
    if(err < 0) {
       perror("Couldn't read the buffer");
       exit(1);
    }
     std::cout<<"Suma real: "<<sum<<std::endl;
     /*se borrara******/
    /* Check result */
     std::cout<<"Suma esperada: "<<actual_sum<<std::endl;
    if(fabs(sum - actual_sum) > 0.01*fabs(sum))
       printf("Check failed.\n");
    else
       printf("Check passed.\n");
    printf("Total time = %lu\n", total_time);

}
void clConv::deallocate_resources(cl_context &context,cl_program &program,cl_command_queue &queue,cl_kernel &complete_kernel,cl_kernel &vector_kernel,cl_kernel &mult_kernel,cl_mem &v1_buffer,cl_mem &v2_buffer,cl_mem &data_buffer,cl_mem &sum_buffer,cl_event &end_event,cl_event &start_event){
    clReleaseEvent(start_event);
    clReleaseEvent(end_event);
    clReleaseMemObject(sum_buffer);
    clReleaseMemObject(data_buffer);
    clReleaseMemObject(v1_buffer);
    clReleaseMemObject(v2_buffer);
     clReleaseKernel(mult_kernel);
    clReleaseKernel(vector_kernel);
    clReleaseKernel(complete_kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
}
