#define VECTOR_SIZE_1 1024
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <iostream>
#include <ctime>
#include <chrono>

#include "Fraccional.h"
int main() {

    /* Data and buffers */
    float v1[VECTOR_SIZE_1],v2[VECTOR_SIZE_1];
    //    float a;
    //     vector<float>v1,v2;
    //     v1.resize(VECTOR_SIZE_1);v2.resize(VECTOR_SIZE_1);
    for(int i=0; i<VECTOR_SIZE_1; i++){
        v1[i] = 1.0f*i;
        v2[i] = 1.0f*i;
        //       a+=v1[i]*v2[i];
    }
    clConv convolucion(VECTOR_SIZE_1);
    float a=convolucion.convolution(v1,v2);
    std::cout<<"La respuesta es: "<<a<<endl;

    chrono::system_clock::time_point start = std::chrono::system_clock::now();

    bool cl=0;
    for (int i=0; i<1000; i++)
    {
        if (cl)
        {
            convolucion.convolution(v1,v2);

        }
        else
        {
            double actual_sum=0;
            for(int i=0; i<VECTOR_SIZE_1; i++) {
                actual_sum+=v1[i]*v2[VECTOR_SIZE_1-1-i];
            }

        }
    }

    chrono::system_clock::time_point finish = std::chrono::system_clock::now();

    //    cout<<v2[131071]<<" "<<"respuesta"<<a<<endl;





    chrono::nanoseconds elapsedNanoseconds = finish.time_since_epoch() - start.time_since_epoch();
    double tiempototal = elapsedNanoseconds.count();
    cout << "Tiempo total: ms " << (tiempototal/1000000) << endl;

    return 0;
}


