#include "heap.h"
/*
  Based on the pseudo-code in CLRS Introduction to Algorithms book.
*/
Heap new_heap(unsigned int size){
     // creates heap[size+1] first node keeps heap information
     Heap h = calloc(size + 1, sizeof(Hnode));
     h[0].id = size;
     h[0].val = 0;
     return h;
}
void free_heap(Heap h){
     free(h);
}

void max_heapify(Heap h, unsigned int idx){
     unsigned int l = 0 ,r = 0, sz = 0, largest = 0;
     l = left_node(idx);
     r = right_node(idx);
     sz = heap_size(h);
     largest = idx;
     if (l <= sz  && h[l].val > h[idx].val){
          largest = l;
     }
     if (r <= sz  && h[r].val > h[largest].val){
          largest = r;
     }
     if (largest != idx){
          Hnode temp;
          temp = h[idx];
          h[idx] = h[largest];
          h[largest] = temp;
          max_heapify(h, largest);
     }
}

void heap_increase_key(Heap h, unsigned int idx, Hnode value){
     g_assert(value.val  >= h[idx].val);
     h[idx] = value;
     while(idx > 1 && h[parent_node(idx)].val < h[idx].val){
          Hnode temp;
          temp = h[idx];
          h[idx] = h[parent_node(idx)];
          h[parent_node(idx)] = temp;
          idx = parent_node(idx);
     }
}

void max_heap_insert(Heap h, Hnode value){
     h[0].val += 1;
     h[heap_size(h)].val = INT_MIN;    
     heap_increase_key(h, heap_size(h), value);
}

Hnode heap_max(Heap h){
     return h[1];
}

Hnode heap_extract_max(Heap h){
     g_assert(heap_size(h) >= 1);
     Hnode max = h[1];
     h[1] = h[heap_size(h)];
     h[0].val -= 1;
     max_heapify(h,1);
     return max;
}

void heap_sort(Heap h){
     unsigned int sz = h[0].val;
     for(int i = heap_size(h) ; i > 1; i--){
          Hnode temp = h[i];
          h[i] = h[1];
          h[1] = temp;
          h[0].val -= 1;
          max_heapify(h,1);
     }
     h[0].val = sz;
}

void heap_print(Heap h){
     for(int i = 1 ; i <= heap_size(h); i++){
          printf("%d %g ",h[i].id, h[i].val);
     }
     printf("\n");
}


