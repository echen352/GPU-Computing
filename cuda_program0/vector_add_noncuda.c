#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void vector_add(int *out, int *a, int *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv){
    int *a, *b, *out;
    int N = atoi(argv[1]);
    clock_t t;
    
    // Allocate memory
    a   = (int*)malloc(sizeof(int) * N);
    b   = (int*)malloc(sizeof(int) * N);
    out = (int*)malloc(sizeof(int) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i + 1;
    }

    t = clock();

    // Main function
    vector_add(out, a, b, N);
    
    t = clock() - t;
    
    // Print results
    /*for (int i = 0; i < N; i++) {
    	printf("\n C[%d]=%d", i, out[i]);
    }*/
    
    printf("\nTime Taken: %f sec\n", ((double)t)/CLOCKS_PER_SEC);
    
}
