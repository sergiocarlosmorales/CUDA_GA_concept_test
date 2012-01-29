#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <cuda.h>
using namespace std;

//Kernel to get exponent, only positives
__device__ void calculate_exponent(int base,int exponent,long &result){
	result = 1;
	if(exponent==0){
		return;
	}
	for(int counter=1;counter<=exponent;counter++){
		result *= base;
	}
}

// Kernel to fill the array with somethin', in this case its just the position but works
__global__ void fill_array(int *input, int totalSizeOfArray, int individualsPerThread, int number_genes, int *randomNumbers)
{
	int startingPosition = threadIdx.y * (number_genes*individualsPerThread);
	for(int counter=0;counter<(individualsPerThread*number_genes);counter++){
		if(counter+startingPosition>=totalSizeOfArray){
			return;
		}

		input[counter+startingPosition] = randomNumbers[counter+startingPosition];
	}

}
// Kernel to evaluate an individual
__global__ void evaluate(int *input, int totalSizeOfArray, int number_genes, int individualsPerThread, long *scores){

	int startingPosition = threadIdx.y * (number_genes*individualsPerThread);
	int startingPosition_scores = threadIdx.y * individualsPerThread;
	long acumulated = 0;
	long temp = 0;
	for(int counter_individuals=0;counter_individuals<individualsPerThread;counter_individuals++){
		if(startingPosition + (counter_individuals*number_genes) >= totalSizeOfArray){
			return;
		}
		for(int counter_gene=0;counter_gene<number_genes;counter_gene++){
			int base = startingPosition + (counter_individuals*number_genes) + counter_gene;
			calculate_exponent(input[base],(number_genes-1)-counter_gene,temp);
			acumulated += temp;
		}
		scores[(threadIdx.y*individualsPerThread)+counter_individuals] = acumulated;
		
		acumulated=0;
	}

}





// main routine that executes on the host
int main(void)
{

	const int number_genes = 10;
	const int number_individuals = 1000000;

	int *population_array_host = new int[number_genes*number_individuals];
	int *population_array_device;

	long *score_array_host = new long[number_individuals];
	long *score_array_device;

	int *random_numbers_host = new int[number_genes*number_individuals];
	int *random_numbers_device;
	
	//we need to initialize the population array
	//must be done randomly
	//we calculate the number of threads required to fill the array in parallel
	int individuals_per_thread = 2000;
	int number_of_threads = number_individuals/individuals_per_thread + (number_individuals%individuals_per_thread == 0 ? 0:1);
	//we now randomly fill the random numbers array
	srand ( time(NULL));
	for(int contador=0;contador<number_genes*number_individuals;contador++){
		random_numbers_host[contador] = ( rand()  % 10 );
	}
	//we move the random numbers array to device
	size_t memory_for_random_numbers = number_genes*number_individuals*sizeof(int);
	cudaMalloc((void **) &random_numbers_device, memory_for_random_numbers);
	cudaMemcpy(random_numbers_device, random_numbers_host, memory_for_random_numbers, cudaMemcpyHostToDevice);

	//we zero-ise the scores
	for(int contador=0;contador<number_individuals;contador++){
		score_array_host[contador] = 0;
	}
	//we move the scores array to device
	size_t memory_for_scores = number_individuals*sizeof(long);
	cudaMalloc((void **) &score_array_device, memory_for_scores);
	cudaMemcpy(score_array_device, score_array_host, memory_for_scores, cudaMemcpyHostToDevice);

	//now we must launch 1 block with dimensions: x=1,y=number_of_threads, we define them
	dim3 grid_fill(1,1);
	dim3 block_fill(1,number_of_threads);
	//we now allocate memory in device
	size_t memory_for_population = number_genes*number_individuals*sizeof(int);
	cudaMalloc((void **) &population_array_device, memory_for_population);
	//we now launch the kernel for populating
	fill_array <<< grid_fill, block_fill >>> (population_array_device, number_genes * number_individuals, individuals_per_thread, number_genes,random_numbers_device);

	//we now launch the kernel for evaluating
	evaluate <<< grid_fill, block_fill >>> (population_array_device, number_genes * number_individuals, number_genes, individuals_per_thread,score_array_device);

	cudaMemcpy(population_array_host, population_array_device, memory_for_population, cudaMemcpyDeviceToHost);
	cudaMemcpy(score_array_host, score_array_device, memory_for_scores, cudaMemcpyDeviceToHost);

	

	
	///END, move back to host, print PopulationArray

	for(int contador=0;contador<number_genes*number_individuals;contador++){
		if(contador%number_genes==0 && contador > 0){
			cout << endl;
		}
		cout << population_array_host[contador] << "-";

	}
	cout << endl;
	cout << "----";
	cout << endl;
	
	for(int contador=0;contador<number_individuals;contador++){
		cout << score_array_host[contador] << endl;
	}
	




}
