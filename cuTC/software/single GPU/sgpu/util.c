//includes
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "base.h"
#include "csf.h"



//functions
/*idx_t inline cal_modereferecesize(idx_t rank)
{ idx_t Modereference;
  Modereference = RESULT_MEM_REF / rank;
  return Modereference;
}*/

/*void cal_blocksize(idx_t * blocksize, csf_sptensor* reference)
{  idx_t i_number,j_number ;
   idx_t totalnumber;
   
   i_number = reference[0] / (idx_t)PHISICAL_CORE_ROW_NUM;
   j_number = reference[1] / (idx_t)CALCULATE_CORE_NUM;
   
}*/

static inline void p_swap_entries(idx_t* sortarray, idx_t* noarray0,
                                  idx_t* noarray1, double* value,
                                  idx_t a, idx_t b)
{
  if(a == b) {
    return;
  }

  idx_t itemp = sortarray[a];
  sortarray[a] = sortarray[b];
  sortarray[b] = itemp;

  itemp = noarray0[a];
  noarray0[a] = noarray0[b];
  noarray0[b] = itemp;

  itemp = noarray1[a];
  noarray1[a] = noarray1[b];
  noarray1[b] = itemp;

  double ftemp = value[a];
  value[a] = value[b];
  value[b] = ftemp;
}

void quicksort(idx_t* sortarray, idx_t* noarray0, idx_t* noarray1,
               double* value, idx_t begin, idx_t end)
{
  if(begin >= end) {
    return;
  }

  idx_t pivot = sortarray[begin + ((end - begin) / 2)];
  idx_t lt = begin;
  idx_t i = begin;
  idx_t gt = end;

  while(i <= gt) {
    if(sortarray[i] < pivot) {
      p_swap_entries(sortarray, noarray0, noarray1, value, lt, i);
      lt++;
      i++;
    } else if(sortarray[i] > pivot) {
      p_swap_entries(sortarray, noarray0, noarray1, value, i, gt);
      if(gt == 0) {
        break;
      }
      gt--;
    } else {
      i++;
    }
  }

  if(lt > begin) {
    quicksort(sortarray, noarray0, noarray1, value, begin, lt - 1);
  }
  if(gt < end) {
    quicksort(sortarray, noarray0, noarray1, value, gt + 1, end);
  }
}

