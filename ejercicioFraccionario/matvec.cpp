#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "matvec.cl"
#define KERNEL_FUNC "matvec_mult"

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>


#include "Fraccional.h"
//void comprobacion(float* pmat, float* pcorr, float *pvec);

int main() {

   
   /* Data and buffers */
   float mat[16], vec[4], result[4];
   vector<double> v1, v2, rv1v2;
   float correct[4] = {0.0f, 0.0f, 0.0f, 0.0f};


    clConv conv1;
    conv1.convolution(v1,v2,rv1v2);
   comprobacion(&mat[0], &correct[0], &vec[0]);
   /* Initialize data to be processed by the kernel */


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
