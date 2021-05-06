__kernel void matvec_mult(__global float* v1,
                          __global float* v2,
                          __global float* result){


    int i = get_global_id(0);
    result[i] = dot(v1[i],v2[i]);
}