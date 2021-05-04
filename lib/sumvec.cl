
__kernel void matvec_sum(__global float* result) {
   int i = get_global_id(0);
    result[0]=result[i]+result[i+1]+result[i+2]+result[i+3];
}
