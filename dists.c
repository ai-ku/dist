#include "dists.h"
char * version = "v0.9";
char * usage =  "\nUsage: dists [options] < input-stream\n"
     "input-stream format sorted according to column ids (c_i)"
     "input-stream:<n:number of elements in the row> <c_i:column id> <c_i_v:column val> ... <c_n:column id> <c_n_v:column val>\n"
     "Options:\n"
     "\t-h \t\t\tDisplay this information\n"
     "\t-d <distance-type>\t. Set <distance-type> \n"
"\t\t0 for Euclid\n"
"\t\t1 for Cosine(default)\n" 
"\t\t2 for Manhattan\n"
"\t\t3 for Maximum\n"
"\t\t4 for Jensen\n"
"\t\t5 for KL2 distance (KL(p,q) + KL(q,p))\n"
"\t\t6 for Zero-mean covariance\n"
     "\t-u <upper-bound>\tCalculate 1000NN of the rows up to the <upper-bound>(default number of rows)\n"
     "\t-l <lower-bound>\tCalculate 1000NN of the rows starting from <lower-bound>(default 0)\n"
     "\t-p <arg>\t\tRun <arg> parallel jobs to calculate kNN(default 1)\n"
     "\t-k <arg>\t\tCalculate <arg>NN of the data, set 0 to get all distances.(default 1000)\n"
     "\t-t Transpose input data\n"
     "\t-g Take log2 of the input (works with Euclid, Cosine, Manhattan, Maximum)\n"
     "\t-v Verbose\n";

/* NOT TESTED FUNCTIONS
   
*/
int LogScale;
int Up;
int Low;
int ThreadCount;
int Ncol;
int K;
int Transpose;
float (* DistFunc)(Row, Row);
Array Data;
Hnode ** Dist;
int VERBOSE;
enum DTYPE {
     euclid = 0,
     cosine = 1,
     manhattan = 2,
     maximum = 3,
     jensen = 4,
     kl2 = 5,
     zerocov = 6
} Dtype;

void init_dist(){
     Data = NULL;
     Dist = NULL;
     DistFunc = dist_cosine_sparse;
     ThreadCount = 1;
     LogScale = 0;
     Up = Low = -1;
     Ncol = 0;
     K = 1000;
     Transpose = 0;
     VERBOSE = 0;
}

void free_data(Array data){
     foreach(Row,rr,Data){
          g_free(rr->val);
          g_free(rr->lval);
          g_free(rr->ids);
          g_free(rr);
     }      
     free_array(Data);    
}

void read_sparse_binary_data_to_array(char * fname){
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
          if(Dtype == euclid)/*Only allocate if the distance metric is euclid*/               
               rr->lval = (float *) malloc(4 *c * sizeof(char));
          fread(rr->ids, 4 * sizeof(char), c, infile);
          fread(rr->val, 4 * sizeof(char), c, infile);
          push(Data,rr);
          foreach_int(jj,0,rr->nnz){
               /* This part improves the performance of by pre-calculation */
               if (rr->ids[jj] > Ncol) Ncol = rr->ids[jj];
               switch(Dtype){
                    case cosine://euclid-precalculation                         
                         rr->lval[jj] = rr->val[jj] * rr->val[jj];
                         break;
                    case euclid://cosine-precalculation
                         rr->norm += rr->val[jj] * rr->val[jj];
                         break;
                    default:
                         break;
               }
          }
          rr->norm = sqrt(rr->norm);
          if (Dtype == cosine){
               /*Cosine L2Normalizing the vector*/
               foreach_int(jj,0,rr->nnz){
                    rr->val[jj] = rr->val[jj] / rr->norm;
               }
          }
     }
     /*Number of unique substitutes in all data*/
     Ncol += 1;
     fclose(infile);
     msg("Data[%d x %d = %dBytes]",lr,Ncol, totalByte);
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
                    rr->ids = (unsigned*) calloc(nnz, sizeof(unsigned int));
                    rr->val = (float *) calloc(nnz, sizeof(float));
                    if(Dtype == euclid)
                         rr->lval = (float *) calloc(nnz, sizeof(float));
                    else if(Dtype == jensen)
                         rr->lval = (float *) calloc(nnz, sizeof(float));
                    else if(Dtype == kl2)
                         rr->lval = (float *) calloc(nnz, sizeof(float));
                    else rr->lval = NULL;
                    push(Data,rr);
                    continue;
               }
               if (iter % 2 == 0){
                    rr->ids[pos] = atoi(word);
                    if (rr->ids[pos] > Ncol) Ncol = rr->ids[pos];
               }
               else if(iter %2 == 1){
                    rr->val[pos] = atof(word);
                    if (LogScale) rr->val[pos] = rr->val[pos] == 0 ? LOG_ZERO : log2(rr->val[pos]);
                    if(Dtype == euclid)
                         rr->lval[pos] = rr->val[pos] * rr->val[pos];
                    else if(Dtype == cosine){
                         if(rr->val[pos] == LOG_ZERO) rr->val[pos] = 0;
                         rr->norm += rr->val[pos] * rr->val[pos];
                    }
                    else if(Dtype == jensen)
                         rr->lval[pos] = rr->val[pos] == 0 ? 0 : rr->val[pos] * log2(rr->val[pos]);
                    else if(Dtype == kl2)
                         rr->lval[pos] = rr->val[pos] == 0 ? 0 : log2(rr->val[pos]);
                    else if(Dtype == zerocov)
                         rr->norm += rr->val[pos];
                    pos++;
               }
               iter++;
          }
          if(Dtype == cosine){
               rr->norm = sqrt(rr->norm);
               foreach_int(i,0,nnz){
                    rr->val[i] /= rr->norm;
               }
          } 
     }
     Ncol += 1;
     if(Dtype == zerocov){
          foreach(Row,rr,Data){
               rr->norm /= Ncol;
          }
     }
     msg("Reading is done\n");
}

void data_transpose(){
     /*Transposes the input data*/
     int nrow = length(Data);
     Array tdata = sized_array(Ncol);
     int *getnnz = (int*)malloc(Ncol * sizeof(int));
     msg("data_transpose:[%d %d]\n", nrow, Ncol);
     foreach_int(ii,0,Ncol)
          getnnz[ii] = 0;
     /*Get nonzero values*/
     msg("Get columnwise element counts");
     foreach(Row, rr, Data){
          foreach_int(ii, 0, rr->nnz){
               getnnz[rr->ids[ii]]++;
          }
     }
     msg("Allocate transpose data");
     foreach_int(ii,0,Ncol){
          Row nr = g_new0(struct _Row,1);
          nr->nnz = 0;
          nr->norm = 0;
          nr->ids = (unsigned*) malloc(getnnz[ii] * sizeof(unsigned int));
          nr->val = (float *) malloc(getnnz[ii] * sizeof(float));
          push(tdata,nr);
     }
     msg("Construct transpose");
     int ri = 0;
     foreach(Row, rr, Data){
          foreach_int(ii, 0, rr->nnz){
               int cid = rr->ids[ii];
               Row nr = aref(tdata,cid);
               int idx = nr->nnz++;
               nr->ids[idx] = ri;
               nr->val[idx] = rr->val[ii];
               nr->norm += rr->val[ii];
          }
          ri++;
     }
     Ncol = nrow;
     if(Dtype == zerocov){
          foreach(Row, nr, tdata){
               nr->norm /= Ncol;
          }
     }
     else
          g_error("Transpose currently supported only by zerocov distance metric");
     free_data(Data);
     free(getnnz);
     Data = tdata;
     /* msg("Print transpose"); */
     /* ri = 0; */
     /* foreach(Row, nr, tdata){ */
     /*      printf("%d", nr->nnz);  */
     /*      foreach_int(ii, 0, nr->nnz){ */
     /*           printf(" %d %f", nr->ids[ii], nr->val[ii]); */
     /*      }  */
     /*      printf(" norm:%f\n",nr->norm); */
     /*      if(ri++ > 20) exit(1); */
     /* }  */

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
     msg("First n-1 threads will do interval:%d last thread will do:%d",interval,last);
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
     clock_t start = clock();
     foreach_int(i, str(w), end(w)){
          Row r_i = aref(Data,i);
          if(i % 100 == 0 && VERBOSE) {
               fprintf(stderr,".");
          }               
          foreach_int(j, 0, length(Data)){               
               if (i == j && Dtype != zerocov) continue;
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

float dist_zeromeancov_sparse(Row p, Row q){
     double sum = 0;
     int p_i = 0, q_i = 0, pnnz = p->nnz, qnnz = q->nnz, done = 0;
     unsigned * pids = p->ids, *qids = q->ids;
     float * pval = p->val, *qval = q->val;
     while(p_i < pnnz && q_i < qnnz){
          if (pids[p_i] == qids[q_i]){
               sum += (pval[p_i] - p->norm) * (qval[q_i] - q->norm);
               p_i++;
               q_i++;
               done++;
          }
          else if(pids[p_i] < qids[q_i]){
               sum += (pval[p_i] - p->norm) * -q->norm;
               p_i++;
               done++;
          }
          else{
               sum +=  -p->norm * (qval[q_i] - q->norm);
               q_i++;
               done++;
          }
     }
     while(p_i < pnnz){
          sum += (pval[p_i] - p->norm) * -q->norm;
          p_i++;
          done++;
     }
     while(q_i < qnnz){
          sum +=  -p->norm * (qval[q_i] - q->norm);
          q_i++;
          done++;
     }
     sum += (Ncol - done) * (p->norm * q->norm);     
     return sum;
}

float dist_manhattan_sparse(Row p, Row q){
     double sum = 0;
     int p_i = 0, q_i = 0, pnnz = p->nnz, qnnz = q->nnz;
     unsigned * pids = p->ids, *qids = q->ids;
     float * pval = p->val, *qval = q->val;
     while(p_i < pnnz && q_i < qnnz){
          if(pids[p_i] == qids[q_i]){
               sum += fabs(pval[p_i] - qval[q_i]);
               p_i++;
               q_i++;
          }
          else if(pids[p_i] < qids[q_i]){
               sum += fabs(pval[p_i]);
               p_i++;
          }
          else{
               sum += fabs(qval[q_i]);
               q_i++;
          }
     }
    while(p_i < pnnz){
         sum += fabs(p->val[p_i]);
         p_i++;
    }
    while(q_i < qnnz){
         sum += fabs(q->val[q_i]);
         q_i++;
    }
    return sum;
}

float dist_maximum_sparse(Row p, Row q){
     double max_diff = 0;
     int p_i = 0, q_i = 0, pnnz = p->nnz, qnnz = q->nnz;
     unsigned * pids = p->ids, *qids = q->ids;
     float * pval = p->val, *qval = q->val;
     while(p_i < pnnz && q_i < qnnz){
          if(pids[p_i] == qids[q_i]){
               double d = fabs(pval[p_i] - qval[q_i]);
               max_diff = d > max_diff ? d : max_diff;
               p_i++;
               q_i++;
          }
          else if(pids[p_i] < qids[q_i]){
               max_diff = fabs(pval[p_i]) > max_diff ? fabs(pval[p_i]) : max_diff;
               p_i++;
          }
          else{
               max_diff = fabs(qval[q_i]) > max_diff ? fabs(qval[q_i]) : max_diff;
               q_i++;
          }
     }
    while(p_i < pnnz){
         max_diff = fabs(pval[p_i]) > max_diff ? fabs(pval[p_i]) : max_diff;
         p_i++;
    }
    while(q_i < qnnz){
         max_diff = fabs(qval[q_i]) > max_diff ? fabs(qval[q_i]) : max_diff;
         q_i++;
    }
    return max_diff;
}

float dist_jensen_sparse(Row p, Row q){
     double sum = 0; 
     int p_i = 0, q_i = 0, pnnz = p->nnz, qnnz = q->nnz;
     unsigned * pids = p->ids, *qids = q->ids;
     while(p_i < pnnz && q_i < qnnz){
          if (pids[p_i] == qids[q_i]){
               float m = (p->val[p_i] + q->val[q_i]);
               float term = m == 0 ? 0 : m * log2(0.5*m);               
               sum += p->lval[p_i] + q->lval[q_i] - term;
               p_i++;
               q_i++;            
          }
          else if(pids[p_i] < qids[q_i]){
               sum += p->val[p_i];
               p_i++;
          }
          else{
               sum += q->val[q_i];
               q_i++;
          }
     }
     while(p_i < pnnz){
          sum += p->val[p_i];
          p_i++;
     }
     while(q_i < qnnz){
          sum += q->val[q_i];
          q_i++;
     }
     if(sum < 0) sum = 0;
     return sqrt(0.5 * sum);     
}

float dist_kl2_sparse(Row p, Row q){
     double sum = 0; 
     int p_i = 0, q_i = 0, pnnz = p->nnz, qnnz = q->nnz;
     unsigned * pids = p->ids, *qids = q->ids;
     if(pnnz != qnnz)  g_error("KL2 only accepts distributions from same domain [%d %d]\n", pnnz, qnnz);
     while(p_i < pnnz && q_i < qnnz){
          if (pids[p_i] != qids[q_i]){
               g_error("KL2 only accepts distributions from same domain\n");
          }
          sum += p->val[p_i] * (p->lval[p_i] - q->lval[p_i]) + q->val[q_i] * (q->lval[q_i] - p->lval[p_i]);
          p_i++;
          q_i++;            
     }
     if(sum < 0) sum = 0;
     return sum;
}

int main (int argc, char * argv[]){
     init_dist();
     clock_t start;
     int r = -1;
     /*Options of program*/
     int opt;
     enum DTYPE opt_dist = 1;
     char *opt_file = NULL;
     while ((opt = getopt(argc, argv, "d:f:u:l:p:k:htvg")) != -1) {
          switch(opt) {
               case 'd':
                    opt_dist = atoi(optarg);
                    switch(opt_dist){
                         case euclid://euclidian
                              msg("Sparse Euclid distance");
                              Dtype = euclid;
                              DistFunc = dist_euclid_sparse;
                              break;
                         case cosine://Cosine distance
                              msg("Sparse Cosine distance");
                              Dtype = cosine;
                              DistFunc = dist_cosine_sparse;
                              break;
                         case manhattan://Manhattan distance
                              msg("Sparse Manhattan distance");
                              Dtype = manhattan;
                              DistFunc = dist_manhattan_sparse;
                              break;
                         case maximum://Maximum distance
                              msg("Sparse Maximum  distance");
                              Dtype = maximum;
                              DistFunc = dist_maximum_sparse;
                              break;
                         case jensen:
                              msg("Sparse Jensen  distance");
                              Dtype = jensen;
                              DistFunc = dist_jensen_sparse;
                              break;
                         case kl2:
                              msg("Sparse KL2 distance");
                              Dtype = kl2;
                              DistFunc = dist_kl2_sparse;
                              break;
                         case zerocov:
                              msg("Sparse zero-mean covariance");
                              Dtype = zerocov;
                              DistFunc = dist_zeromeancov_sparse;
                              break;
                         default:
                              g_error("Invalid Distance Option");
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
                    fprintf(stderr,"%s",usage);
                    exit(0);
               case 't':
                    Transpose = 1;
                    break;        
               case 'v':
                    VERBOSE = 1;
                    break;
               case 'g':
                    LogScale = 1;
                    break;
               default:
                    g_warning("Invalid option:%s\n",optarg);
                    g_error("%s",usage);
          }
     }     
     if(LogScale && (opt_dist == 4 || opt_dist == 5 || opt_dist == 6))
          g_error("Log Scale is only applicable to Euclid, Cosine, Manhattan and Maximum\n%s\n", usage);
     else if (LogScale) msg("Log Scale is enabled\n");

     msg("File:%s Distance:%d Threads:%d Up:%d Low:%d KNN:%d", opt_file, opt_dist, ThreadCount, Up, Low, K);
     /* if (argc <= 1 || DistFunc == NULL){ */
     /*      g_error("%s",usage); */
     /* } */
     start = clock();
     if (opt_file != NULL){
          msg("read sparse_binary");
          read_sparse_binary_data_to_array(opt_file);          
     }
     else{
          msg("read sparse_stdin");
          read_data_stdin();
     }
     if(Transpose)
          data_transpose();    
     r = length(Data);     
     if(r == 0 || Data == NULL)
          g_error("Invalid input file");
     if (Up == Low){
          Up = r;
          Low = 0;
     }
     if(Up > -1)
          Up = Up > r ? r : Up;
     else if(Up == -1)
          Up = r;     
     if(Low <= -1 || Low >= r)
          Low = 0;
     g_assert(Up > Low);
     K = K == 0 ? r : K > r ? r : K;
     msg("Data[%d %d] read time:%f",length(Data), Ncol, ((float)clock() - start) / CLOCKS_PER_SEC);     
     msg("Allocating Heaps [%d * %dNN]",r , K);
     Dist = (Hnode**)malloc(r * sizeof(Hnode*));
     for(int i = Low; i < Up; i++){
          if(i % 10000 == 0 && VERBOSE) fprintf(stderr,".");
          Dist[i] = new_heap(K);
     }
     msg("Calculate distances");
     split_thread_work();
     foreach_int(i,Low,Up){
          printf("%d ",i);
          heap_print(((Hnode**)Dist)[i]);
     }
     free_data(Data);
     foreach_int(i,Low,Up){
          g_free(Dist[i]);
     }    
     g_free(Dist);     
     return 0;
}
