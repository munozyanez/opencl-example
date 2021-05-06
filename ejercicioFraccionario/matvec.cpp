#define VECTOR_SIZE_1 131072
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
    // vector<float>v1,v2;
     //v1.resize(VECTOR_SIZE_1);v2.resize(VECTOR_SIZE_1);
    for(int i=0; i<VECTOR_SIZE_1; i++){
       v1[i] = 1.0f*i;
       v2[i] = 1.0f*i;
    }

          clConv convolucion(1024);
          float a=convolucion.convolution(v1,v2);
          std::cout<<" La respuesta es: "<<a<<endl;



//        chrono::system_clock::time_point start = std::chrono::system_clock::now();

//       chrono::system_clock::time_point finish = std::chrono::system_clock::now();
//       chrono::nanoseconds elapsedNanoseconds = finish.time_since_epoch() - start.time_since_epoch();
//       double tiempo1bucle = elapsedNanoseconds.count()/1000;
//       double tiempototal = elapsedNanoseconds.count();
//       cout << "Tiempo 1 bucle ms: " << (tiempo1bucle/1000000) << endl;
//       cout << "Tiempo total: ms " << (tiempototal/1000000) << endl;


   return 0;
}


