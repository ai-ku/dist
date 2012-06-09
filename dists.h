#ifndef __DISTS_H__
#define __DISTS_H__
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
/*My headers*/
#include "foreach.h"
#include "array.h"
#include "heap.h"

/*typedefs*/
typedef struct _Row{
     int nnz;
     double norm;
     float *val;
     float *lval;
     unsigned int * ids;
} *Row;

typedef struct WorkS{
     int id;     
     int str_r;
     int end_r;
} * Work;

/*Macros*/
#define id(w) ((w)->id)
#define str(w) ((w)->str_r)
#define end(w) ((w)->end_r)
#define msg if(VERBOSE) g_message
#define info(w) g_assert((w)!= NULL);if(VERBOSE)fprintf(stderr,"[Work[%d] str:%d end:%d]\n", (w)->id, (w)->str_r, (w)->end_r)
/*Constants*/
#define EPS 0

/*Function Declerations*/
void init_dist();
void read_sparse_binary_data_to_array(char * fname);
void read_data_stdin();
void split_thread_work();
void * dist_sparse_thread(void * ptr);
float dist_euclid_sparse(Row p, Row q);
float dist_cosine_sparse(Row p, Row q);
float dist_manhattan_sparse(Row p, Row q);
float dist_maximum_sparse(Row p, Row q);
float dist_jensen_sparse(Row p, Row q);

#endif
