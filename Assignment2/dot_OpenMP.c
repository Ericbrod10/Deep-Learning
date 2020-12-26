#include<stdio.h>
#include<omp.h>
#include<stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>


int main(int argc ,char* argv[]) {
	//printf("hello = %ld" , CLOCKS_PER_SEC);
	clock_t start, end;
	float seconds = 0;
	start = clock();
	

	FILE *data_file;
	FILE *vector_file;
	size_t size;
	
	/* Initialize rows, cols, CUDA devices and threads from the user */
	unsigned int rows=atoi(argv[3]);
	unsigned int cols=atoi(argv[4]);
	int nprocs = atoi(argv[5]);
	
	if (nprocs == 0){
	printf("\n**** ERROR: No Processes Found ****\n\n");
}	
	printf("**** Input Values ****\n");
	printf("Rows = %d\nCols = %d\nProcesses = %d \n",rows,cols,nprocs);

	/*Host variable declaration */

	float* host_results = (float*) malloc(rows * sizeof(float)); 
	struct timeval starttime,starttimeorg, endtime;
	gettimeofday(&starttimeorg, NULL);

	unsigned int jobs; 
	unsigned long i;

	/*Kernel variable declaration */
	
	float arr[rows][cols];
	float var ;
	int vrow =1;

	start = clock();

	/* Validation to check if the data file is readable */
	
	data_file = fopen(argv[1], "r");
	vector_file = fopen(argv[2],"r");
	
	if (data_file == NULL) {
    		printf("Cannot Open the Data File");
		return 0;
	}
	if (vector_file == NULL){
		printf("cannot open the vector file");
	}
	size = (size_t)((size_t)rows * (size_t)cols);
	size_t sizeV = 0;
	sizeV = (size_t)((size_t)vrow*(size_t)cols);

	fflush(stdout);
	
	float *dataT = (float*)malloc((size)*sizeof(float));
	float *dataV = (float*)malloc((sizeV) * sizeof(float));

	if(dataT == NULL) {
	        printf("ERROR: Memory for data not allocated.\n");
	}
	
	if(dataV == NULL){
		printf("ERROR: Memory for vector not allocated. \n");
	}
        gettimeofday(&starttime, NULL);
	int j = 0;

    /* Transfer the Data from the file to CPU Memory */
	
	printf("\n**** Transfer Data from File to CPU Memory ****\n");
	
        for (i =0; i< rows;i++){
		for(j=0; j<cols ; j++){
			fscanf(data_file,"%f",&var);
                        arr[i][j]=var;
			//printf("%f\n",var);
		}
	}
	for (i =0;i<cols;i++){
		for(j= 0; j<rows; j++){
			dataT[rows*i+j]= arr[j][i];
	}
	}		

	for ( i = 0; i <rows; i++){
		
		for (j=0;j<cols;j++){
			fscanf(vector_file,"%f",&dataV[j]);
		}
	}
//   	printf("Read Data");
	fclose(data_file);
	fclose(vector_file);
        fflush(stdout);

        gettimeofday(&endtime, NULL);
        seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/CLOCKS_PER_SEC)-((double)starttime.tv_sec+(double)starttime.tv_usec/CLOCKS_PER_SEC);

        printf("time to read data = %f\n", seconds);

	jobs = (unsigned int) ((rows + nprocs - 1) / nprocs);

        gettimeofday(&starttime, NULL);

	/* Calling the kernel function */
	
	printf("jobs=%d\n", jobs);
	
	kernel(rows,cols,dataT,	dataV, host_results, jobs);
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/CLOCKS_PER_SEC)-((double)starttime.tv_sec+(double)starttime.tv_usec/CLOCKS_PER_SEC);
	printf("time for kernel=%f\n", seconds);
			
	printf("\n**** Output ****\n");
	
	int m;
	
	for(m = 0; m < rows; m++) {
		printf("%f ", host_results[m]);
		printf("\n");
	}
	printf("\n");
	gettimeofday(&endtime, NULL);
        seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/CLOCKS_PER_SEC)-((double)starttimeorg.tv_sec+(double)starttimeorg.tv_usec/CLOCKS_PER_SEC);
	printf("Program Execution Time = %f\n", seconds);

	return 0;

}
