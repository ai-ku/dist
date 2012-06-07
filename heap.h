#ifndef __HEAP_H__
#define __HEAP_H__
#include <foreach.h>

typedef struct _Hnode{
     unsigned int id;
     float val;
} Hnode, *Heap;
/*
  Initial node of the heap: 
  id = capacity of the heap
  val = current number of elements in heap

  For the rest of the nodes:
  id = instance id
  val = instance dist
*/

#define heap_cap(h) ((h)[0].id)
#define heap_size(h) ((unsigned int)(h)[0].val)
#define heap_best(h) ((h)[1].val)

#define parent_node(n) (n >> 1)
#define left_node(n) (n << 1)
#define right_node(n) ((n << 1) + 1)


Heap new_heap(unsigned int size);
void free_heap(Heap h);
void max_heapify(Heap h, unsigned int idx);
void heap_increase_key(Heap h, unsigned int idx, Hnode value);
void max_heap_insert(Heap h, Hnode value);
Hnode heap_max(Heap h);
Hnode heap_extract_max(Heap h);
void heap_sort(Heap h);
void heap_print(Heap h);

#endif
