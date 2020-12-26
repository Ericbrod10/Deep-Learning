
// Main function
int main(int argc ,char* argv[]) {


FILE *data;
FILE *weights;
size_t size;
	

// Declaring the rows and columns and CUDA device and number of threads 
unsigned int rows=atoi(argv[3]);
unsigned int cols=atoi(argv[4]);
int CUDA_DEVICE = atoi(argv[5]);
int THREADS = atoi(argv[6]);

printf("Rows = %d, Cols = %d, CUDA_DEVICE = %d, THREADS = %d \n",rows,cols,CUDA_DEVICE,THREADS);

cudaError err = cudaSetDevice(CUDA_DEVICE);
if(err != cudaSuccess) { printf("Error in setting the CUDA device\n"); exit(EXIT_FAILURE); }



// Declaring the variable for the host
int BLOCKS;
float* host_results = (float*) malloc(rows * sizeof(float)); 
struct timeval starttime, endtime;
clock_t start, end;
float seconds = 0;
unsigned int jobs; 
unsigned long i;



// Declaring the variable for the devices
float  *dev_dataT;
float *dev_dataV;
float *results;
float arr[rows][cols];
float var ;
int vrow =1;
start = clock();



// Validate if the file is readable
data = fopen(argv[1], "r");
weights = fopen(argv[2],"r");
if (data == NULL) {
  printf("Error in reading in the data\n");
	return 0;
}
if (weights == NULL){
	printf("Error in reading in the weights\n");
}
size = (size_t)((size_t)rows*(size_t)cols);
size_t sizeV = 0;
sizeV = (size_t)((size_t)vrow*(size_t)cols);
fflush(stdout);


// Memory allocation for the dat files
float *dataT = (float*)malloc((size)*sizeof(float));
float *dataV = (float*)malloc((sizeV)*sizeof(float));
if(dataT == NULL) {
	printf("Error in allocating memory for the data file.\n");
}
if(dataV == NULL){
	printf("Error in allocating memory for the weights file. \n");
}  
gettimeofday(&starttime, NULL);

int j = 0;
// Moving the data from the file to the allocated memory
for (i =0; i< rows;i++){
	for(j=0; j<cols ; j++){
		fscanf(data,"%f",&var);
      arr[i][j]=var;
}
}
for (i =0;i<cols;i++){
  for(j= 0; j<rows; j++){
		dataT[rows*i+j]= arr[j][i];
}
}		
for (j=0;j<cols;j++){
	fscanf(weights,"%f",&dataV[j]);
}   





fclose(data);
fclose(weights);
fflush(stdout);
gettimeofday(&endtime, NULL);
seconds+=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);


// Memory allocation on the GPU for the data
gettimeofday(&starttime, NULL);
err = cudaMalloc((float**) &dev_dataT, (size_t) size * (size_t) sizeof(float));

if(err != cudaSuccess) { printf("Error in allocating memory on the GPU\n"); }
gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
gettimeofday(&starttime, NULL);


// test vector
err = cudaMalloc((float**) &dev_dataV, sizeV * sizeof(float));
if(err != cudaSuccess) { printf("Error in allocating memory on GPU\n"); }
gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
gettimeofday(&starttime, NULL);
	
// Memory allocation on GPU
err = cudaMalloc((float**) &results, rows * sizeof(float) );
if(err != cudaSuccess) { printf("Error in allocating memory on the GPU for the results\n"); }
gettimeofday(&endtime, NULL); 
seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);

// Copying the data to the GPU
gettimeofday(&starttime, NULL);
err = cudaMemcpy(dev_dataT, dataT, (size_t)size *sizeof(float), cudaMemcpyHostToDevice);
if(err != cudaSuccess) { printf("Error in copying data to the GPU\n"); }
gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);

// Copying weights to the allocated memory on the GPU
gettimeofday(&starttime, NULL);
err = cudaMemcpy(dev_dataV, dataV, sizeV*sizeof(float), cudaMemcpyHostToDevice);
if(err != cudaSuccess) { printf("Error in copying the weights to the GPU\n"); }
gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
jobs = rows;
BLOCKS = (jobs + THREADS - 1)/THREADS;
gettimeofday(&starttime, NULL);

// calling  the kernel function
kernel<<<BLOCKS,THREADS>>>(rows,cols,dev_dataT,	dev_dataV, results);
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
		
// copy the results back to the CPU
cudaMemcpy(host_results,results,rows * sizeof(float),cudaMemcpyDeviceToHost);
printf("Output of the dot product: \n");
printf("\n");

for(int k = 0; k < jobs; k++) {
	printf("%f ", host_results[k]);
	printf("\n");
}


printf("\n");
cudaFree( dev_dataT );
cudaFree( results );
end = clock();
seconds = (float)(end - start) / CLOCKS_PER_SEC;
printf("Execution time: %f\n", seconds);
return 0;
}
