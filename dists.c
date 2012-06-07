#include "dists.h"
char * version = "v0.3";
char * usage =  "\nUsage: dists [options] < input-stream\n"
     "input-stream:<n:number of elements in the row> <c_i:column id> <c_i_v:column val> ... <c_n:column id> <c_n_v:column val>\n"
     "Options:\n"
     "\t-h \t\t\t\t Display this information\n"
     "\t-d <distance-type>\t\t Set <distance-type> 0 for Euclid, 1 for Cosine(default Cosine)\n"
     "\t-u <upper-bound>\t\t Calculate 1000NN of the rows up to the <upper-bound>(default number of rows)\n"
     "\t-l <lower-bound>\t\t Calculate 1000NN of the rows starting from <lower-bound>(default 0)\n"
     "\t-p <arg>\t\t\t Run <arg> parallel jobs to calculate kNN(default 1)\n"
     "\t-k <arg>\t\t\t Calculate <arg>NN of the data(default 1000)\n"
     "\t-v \t\t\t\t Verbose\n";

int Up;
int Low;
int ThreadCount;
int SubstituteTypes;
int K;
float (* DistFunc)(Row, Row);
Array Data;
Hnode ** Dist;
int VERBOSE;

void init_dist(){
     Data = NULL;
     Dist = NULL;
     DistFunc = dist_cosine_sparse;
     ThreadCount = 1;
     Up = Low = -1;
     SubstituteTypes = 0;
     K = 1000;
     VERBOSE = 0;
}
                             
void read_sparse_binary_data_to_array(char * fname, int type){
     FILE * infile;     
     unsigned lr,c, totalByte = 0;
     if(!(infile = fopen(fname, "r"))){
          g_error("file:%s not found\n",fname);
     }
     fread(&lr, sizeof(char) * 4,1,infile);
     totalByte += sizeof(char) * 4;
     Data = sized_array(lr);
     foreach_int(i, 0, lr){          
          if(i%10000 == 0)fprintf(stderr,".");
          fread(&c, sizeof(char) * 4,1,infile);
          Row rr = g_new0(struct _Row,1);         
          rr->nnz = c;
          rr->norm = 0;
          totalByte += sizeof(char) * 4 * 2 * c;
          rr->ids = (unsigned*) malloc(4 *c * sizeof(char));
          rr->val = (float *) malloc(4 *c * sizeof(char));
          if(type != 1)/*Only allocate if the distance metric is euclid*/               
               rr->lval = (float *) malloc(4 *c * sizeof(char));
          fread(rr->ids, 4 * sizeof(char), c, infile);
          fread(rr->val, 4 * sizeof(char), c, infile);
          push(Data,rr);
          foreach_int(jj,0,rr->nnz){
               /* This part improves the performance of by pre-calculation */
               if (rr->ids[jj] > SubstituteTypes) SubstituteTypes = rr->ids[jj];
               switch(type){
                    case 0://euclid-precalculation                         
                         rr->lval[jj] = rr->val[jj] * rr->val[jj];
                         break;
                    case 1://cosine-precalculation
                         rr->norm += rr->val[jj] * rr->val[jj];
                         break;
                    default:
                         break;
               }
          }
          rr->norm = sqrt(rr->norm);
          if (type == 1){
               /*Cosine L2Normalizing the vector*/
               foreach_int(jj,0,rr->nnz){
                    rr->val[jj] = rr->val[jj] / rr->norm;
               }
          }
     }
     /*Number of unique substitutes in all data*/
     SubstituteTypes += 1;
     fclose(infile);
     msg("Data[%d x %d = %dBytes]\n",lr,SubstituteTypes, totalByte);
}

void read_data_stdin(){
     Data = empty_array();
     int licnt = 0;
     foreach_line(str,"") {
          int nnz = -1, iter = 0, pos = 0;
          Row rr = g_new0(struct _Row,1);
          if (licnt++ % 10000 == 0) fprintf(stderr,".");
          foreach_token(word, str) {
               if (nnz == -1){
                    nnz = atoi(word);
                    iter = 0;
                    rr->nnz = nnz;
                    rr->norm = 0;
                    rr->ids = (unsigned*) calloc(nnz, sizeof(float));
                    rr->val = (float *) calloc(nnz, sizeof(float));
                    rr->lval = (float *) calloc(nnz, sizeof(float));
                    push(Data,rr);
                    continue;
               }
               if (iter % 2 == 0){
                    rr->ids[pos] = atoi(word);
                    if (rr->ids[pos] > SubstituteTypes) SubstituteTypes = rr->ids[pos];
               }
               else if(iter %2 == 1){
                    rr->val[pos] = atof(word);
                    rr->lval[pos] = rr->val[pos] * rr->val[pos];
                    rr->norm += rr->val[pos] * rr->val[pos];
                    pos++;
               }
               iter++;
          }
          rr->norm = sqrt(rr->norm);
          foreach_int(i,0,nnz){
               rr->val[i] /= rr->norm;
          }
     }
     SubstituteTypes += 1;
}

void split_thread_work(){
     unsigned r = Up;
     int current_r = Low;
     if(r - current_r < ThreadCount)
          ThreadCount = r - current_r;     
     pthread_t * threads = (pthread_t *) calloc(ThreadCount, sizeof(pthread_t));
     Work * w = (Work *) calloc(ThreadCount, sizeof(Work));       
     int interval = (r - current_r) / ThreadCount;
     int last = r - current_r - interval * (ThreadCount - 1);
     msg("NumberOfThreads:%d start:%d end:%d",ThreadCount, current_r, r);
     msg("First n-1 threads will do interval:%d last thread wiil do:%d",interval,last);
     foreach_int(i,0,ThreadCount){
          if(w[i] == NULL){
               w[i] = (Work)calloc(1,sizeof(struct WorkS));
               id(w[i]) = i;
               str(w[i]) = end(w[i]) = -1;
          }
          str(w[i]) = current_r;    
          end(w[i]) = i < ThreadCount - 1 ? current_r + interval : current_r + last;
          current_r += i < ThreadCount - 1 ? interval : last;
          msg("Thread:%d start:%d ends:%d",i,str(w[i]), end(w[i]));
          pthread_create(&(threads[i]), NULL, dist_sparse_thread, (void *)w[i]);  
          g_assert(current_r <= r);
     }
     foreach_int(i,0,ThreadCount){
          void *end = NULL;
          pthread_join(threads[i], end);
          free(w[i]);
     }
     free(threads);
     free(w);
}

void * dist_sparse_thread(void * ptr){    
     Work w = (Work) ptr;
     info(w);
     Hnode n;
     float best = INT_MAX;
     clock_t start = clock();
     foreach_int(i, str(w), end(w)){
          Row r_i = aref(Data,i);
          if(i % 100 == 0 && VERBOSE) {
               fprintf(stderr,".");
          }               
          foreach_int(j, 0, length(Data)){               
               if (i == j) continue;
               Row r_j = aref(Data,j);
               n.val = DistFunc(r_i, r_j);
               n.id = j;
               if (heap_size(Dist[i]) < K){
                    max_heap_insert(Dist[i],n);
               }
               else if (heap_size(Dist[i]) > 0 && heap_best(Dist[i]) > n.val){
                    Dist[i][1] = n;
                    max_heapify(Dist[i],1);
               }
               best = heap_best(Dist[i]);
          }
          heap_sort(Dist[i]);
     }      
     if(VERBOSE)
          fprintf(stderr, "[node:%d %f]",id(w), ((float)clock() - start) / CLOCKS_PER_SEC);
     info(w);
     return NULL;
}

float dist_euclid_sparse(Row p, Row q){
     double sum = 0; 
     int p_i = 0, q_i = 0, pnnz = p->nnz, qnnz = q->nnz;
     unsigned * pids = p->ids, *qids = q->ids;
     while(p_i < pnnz && q_i < qnnz){
          if (pids[p_i] == qids[q_i]){
               float d = p->val[p_i] - q->val[q_i];
               sum += d * d;
               p_i++;
               q_i++;            
          }
          else if(pids[p_i] < qids[q_i]){
               sum += p->lval[p_i];
               p_i++;
          }
          else{
               sum += q->lval[q_i];
               q_i++;
          }
     }
     while(p_i < pnnz){
          sum += p->lval[p_i];
          p_i++;
     }
     while(q_i < qnnz){
          sum += q->lval[q_i];
          q_i++;
     }
     return sqrt(sum);
}

float dist_cosine_sparse(Row p, Row q){
     double sum = 0;
     int p_i = 0, q_i = 0, pnnz = p->nnz, qnnz = q->nnz;
     unsigned * pids = p->ids, *qids = q->ids;
     float * pval = p->val, *qval = q->val;
     while(p_i < pnnz && q_i < qnnz){
          if (pids[p_i] == qids[q_i]){
               sum += pval[p_i] * qval[q_i];
               p_i++;
               q_i++;
          }
          else if(pids[p_i] < qids[q_i]){
               p_i++;
          }
          else{
               q_i++;
          }
     }
     if (sum > 1) sum = 1;
     else if (sum < -1) sum = -1;
     return acos(sum) * M_1_PI;
}

int main (int argc, char * argv[]){
     init_dist();
     clock_t start;
     int r = -1;
     /*Options of program*/
     int opt;
     int opt_dist = 0;
     char *opt_file = NULL;
     while ((opt = getopt(argc, argv, "d:f:u:l:p:k:hv")) != -1) {
          switch(opt) {
               case 'd':
                    opt_dist = atoi(optarg);
                    switch(opt_dist){
                         case 0://euclidian
                              msg("Sparse Euclid distance\n");
                              DistFunc = dist_euclid_sparse;
                              break;
                         case 1://Cosine distance
                              msg("Sparse Cosine distance\n");
                              DistFunc = dist_cosine_sparse;
                              break;
                         default:
                              g_error("Invalid Distance Option\n");
                    }
                    break;
               case 'k':
                    K = atoi(optarg);
                    break;
               case 'f':
                    opt_file = optarg;
                    break;
               case 'u':
                    Up= atoi(optarg);
                    break;
               case 'l':
                    Low = atoi(optarg);
                    break;
               case 'p':
                    ThreadCount = atoi(optarg) >= 1 ? atoi(optarg) : 1;
                    break;
               case 'h':
                    fprintf(stderr,usage);
                    exit(-1);
               case 'v':
                    VERBOSE = 1;
                    break;
               default:
                    g_warning("Invalid option:%s\n",optarg);
                    g_error(usage);
          }
     }
     msg("File:%s Distance:%d Threads:%d Up:%d Low:%d KNN:%d\n", opt_file, opt_dist, ThreadCount, Up, Low, K);
     if (argc <= 1 || DistFunc == NULL)
          g_error(usage);
     start = clock();
     fprintf(stderr,"read sparse_binary\n");
     if (opt_file != NULL)
          read_sparse_binary_data_to_array(opt_file, opt_dist);          
     else
          read_data_stdin();
     r = length(Data);
     if(r == 0 || Data == NULL){
          g_error("Invalid input file\n");
     }
     if (Up == Low){
          Up = r;
          Low = 0;
     }
     if(Up > -1){
          Up = Up > r ? r : Up;
     }
     if(Low < -1 || Low >= r){
          Low = 0;
     }
     g_assert(Up > Low);
     msg("time:%f\n",((float)clock() - start) / CLOCKS_PER_SEC);
     msg("Allocating Heaps [%d * %dNN]\n",r , K);
     Dist = (Hnode**)malloc(r * sizeof(Hnode*));
     for(int i = Low; i < Up; i++){
          if(i % 10000 == 0 && VERBOSE) fprintf(stderr,".");
          Dist[i] = new_heap(K);
     }
     msg("\nCalculate distances\n");
     split_thread_work();
     foreach_int(i,Low,Up){
          printf("%d ",i);
          heap_print(((Hnode**)Dist)[i]);
     }
     foreach_int(i,0,r){
          g_free(Dist[i]);
          Row rr = aref(Data, i);               
          g_free(rr->val);
          g_free(rr->lval);
          g_free(rr->ids);
     }    
     g_free(Dist);
     free_array(Data);
     return 0;
}
