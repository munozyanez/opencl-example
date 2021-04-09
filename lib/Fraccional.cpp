#include "Fraccional.h"
#ifdef MAC
#include <OpenCL/cl.h>
#else
//#include <CL/cl.h>
#endif


void comprobacion(float* pmat, float* pcorr, float * pvec)
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
}
