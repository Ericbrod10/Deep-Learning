#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dot.h"

void kernel(int rows, int cols , float* ddata,float* vdata ,float *results, int jobs){
	
	int i, j, stop;
        float dp;
        int tid = omp_get_thread_num();
//	printf("\n tid = %d \n" , tid);
        if ((tid +1)*jobs < rows) {
	stop = rows;
}
        else {
	stop = (tid +1)*jobs;
}
        printf("thread id= %d, start = %d, stop =%d \n",tid,tid*jobs, stop);

	for (j = tid*jobs; j<stop; j++)
        {
	    dp = 0;
            for(i =0; i<cols ;i++ )
            {
                    dp+= ddata[ i * rows + j] * vdata[i];
		//printf("\n cols = %d, i = %d", cols, i);
            }
            results[j] = dp;
		//printf("%f",dp);
        }
}

