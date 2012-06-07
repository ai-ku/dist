#ifndef __ARRAY_H__
#define __ARRAY_H__
#include <glib.h>
#define length(a) ((a)->len)
#define aref(a,i) g_ptr_array_index(a,i)
#define empty_array() g_ptr_array_new()
#define sized_array(s) g_ptr_array_sized_new(s)
#define push(a,p) g_ptr_array_add(a,p)
#define free_array(a) g_ptr_array_free(a, TRUE)
#define clear_array(a) g_ptr_array_set_size((a), 0);
typedef GPtrArray *Array;
#endif
