/*------------------------------------------------------------------------------
  
  Crystal Router
  
  Accomplishes all-to-all communication in log P msgs per proc
  The routine is low-level; the format of the input/output is an
  array of integers, consisting of a sequence of messages with format:
  
      target proc
      source proc
      m
      integer
      integer
      ...
      integer  (m integers in total)

  Before crystal_router is called, the source of each message should be
  set to this proc id; upon return from crystal_router, the target of each
  message will be this proc id.
  
  Example Usage:
  
    struct crystal cr;
    
    crystal_init(&cr, &comm);  // makes an internal copy of comm
    
    crystal.data.n = ... ;  // total number of integers (not bytes!)
    buffer_reserve(&cr.data, crystal.n * sizeof(uint));
    ... // fill cr.data.ptr with messages
    crystal_router(&cr);
    
    crystal_free(&cr);
    
  ----------------------------------------------------------------------------*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "types.h"
#include "comm.h"
#include "mem.h"
#include "crystal.h"
#include "sort.h"
#include "sarray_sort.h"
#include "sarray_transfer.h"

void crystal_init(struct crystal *cr, const comm_ptr comm)
{
  comm_dup(&(cr->comm), comm);
  buffer_init(&cr->data,1000);
  buffer_init(&cr->work,1000);

  comm_alloc(cr->comm, 1000 * sizeof(uint));
  cr->size = upc_all_alloc(THREADS, sizeof(uint));
}

void crystal_free(struct crystal *cr)
{
  upc_barrier;
  comm_free(&(cr->comm));
  buffer_free(&cr->data);
  buffer_free(&cr->work);

  if (MYTHREAD == 0)
    upc_free((shared void*)cr->size);
  upc_barrier;
  cr->size = NULL;
}

static void uintcpy(uint *dst, const uint *src, uint n)
{
  if(dst+n<=src)    memcpy (dst,src,n*sizeof(uint));
  else if(dst!=src) memmove(dst,src,n*sizeof(uint));
}

static uint crystal_move(struct crystal *cr, uint cutoff, int send_hi)
{
  uint len, *src, *end;
  uint *keep = cr->data.ptr, *send;
  uint n = cr->data.n;

  comm_alloc(cr->comm, n * sizeof(uint));
  send = (unsigned int*) cr->comm->buf;

  while(cr->size[MYTHREAD] > 0) UPC_POLL;

  if(send_hi) { /* send hi, keep lo */
    for(src=keep,end=keep+n; src<end; src+=len) {
      len = 3 + src[2];
      if(src[0]>=cutoff) memcpy (send,src,len*sizeof(uint)), send+=len;
      else               uintcpy(keep,src,len),              keep+=len;
    }
  } else      { /* send lo, keep hi */
    for(src=keep,end=keep+n; src<end; src+=len) {
      len = 3 + src[2];
      if(src[0]< cutoff) memcpy (send,src,len*sizeof(uint)), send+=len;
      else               uintcpy(keep,src,len),              keep+=len;
    }
  }
  cr->data.n = keep - (uint*)cr->data.ptr;
  cr->size[MYTHREAD] = send - (uint*)cr->comm->buf;
  return      send - (uint*)cr->comm->buf;

}

static void crystal_exchange(struct crystal *cr, uint send_n, uint targ,
                             int recvn, int tag)
{

  comm_req req[3];
  uint count[2] = {0,0}, sum, *recv[2];

  if (recvn) count[0] = cr->size[targ];

  if (recvn == 2) count[1] = cr->size[cr->comm->id - 1];

  sum = cr->data.n + count[0] + count[1];
  buffer_reserve(&cr->data,sum*sizeof(uint));
  recv[0] = (uint*)cr->data.ptr + cr->data.n, recv[1] = recv[0] + count[0];
  cr->data.n = sum;

  if (recvn)
    upc_memget(recv[0], cr->comm->buf_dir[targ], count[0] * sizeof(uint));
  if (recvn == 2)
    upc_memget(recv[1], cr->comm->buf_dir[cr->comm->id-1], count[1] * sizeof(uint));
  if (recvn) cr->size[targ] = 0;
  if (recvn==2) cr->size[cr->comm->id - 1] = 0;

}

void crystal_router(struct crystal *cr)
{
  uint bl=0, bh, nl;
  uint id = cr->comm->id, n=cr->comm->np;
  uint send_n, targ, tag = 0;
  int send_hi, recvn;
  const int st[2] = {-2, -3};

#if defined( __UPC_ATOMIC__) && defined(USE_ATOMIC)
  upc_atomic_relaxed(cr->comm->upc_domain, NULL, UPC_SET, 
		     (shared void *) &cr->comm->flgs[MYTHREAD], 
		     &st[0], 0);
#else
  cr->comm->flgs[MYTHREAD] = -2;
#endif
  cr->size[MYTHREAD] = 0;
  upc_barrier;
  
  while(n>1) {
    nl = (n+1)/2, bh = bl+nl;
    send_hi = id<bh;
    recvn = 1, targ = n-1-(id-bl)+bl;
    while(cr->comm->flgs[targ] != (tag - 2)) UPC_POLL;
    send_n = crystal_move(cr,bh,send_hi);
    
    if(id==targ) targ=bh, recvn=0;
    if(n&1 && id==bh) recvn=2;
#if defined( __UPC_ATOMIC__) && defined(USE_ATOMIC)
    upc_atomic_relaxed(cr->comm->upc_domain, NULL, UPC_SET, 
		       (shared void *) &cr->comm->flgs[targ], 
		       &st[1], 0);
#else
    cr->comm->flgs[targ] = -3;
#endif
    while(cr->comm->flgs[MYTHREAD] != -3) UPC_POLL;
    crystal_exchange(cr,send_n,targ,recvn,tag);

    if(id<bh) n=nl; else n-=nl,bl=bh;
#if defined( __UPC_ATOMIC__) && defined(USE_ATOMIC)
    upc_atomic_relaxed(cr->comm->upc_domain, NULL, UPC_SET, 
		       (shared void *) &cr->comm->flgs[MYTHREAD], 
		       &tag, 0);
#else
    cr->comm->flgs[MYTHREAD] = tag;
#endif
    tag += 2;
  }
}



