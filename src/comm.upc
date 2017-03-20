#include <stddef.h> /* for size_t */
#include <stdio.h>
#include <stdlib.h> /* for exit */
#include <string.h> /* memcpy */
#include <time.h>
#include <math.h>
#include <limits.h> /* for gs identities */
#include <float.h>  /* for gs identities */
#include "name.h"
#include "fail.h"
#include "types.h"
#include "tensor.h"
#include "gs_defs.h"
#include "gs_local.h"
#include "comm.h"

// Global declarations
// Each element in msg_queue is a pointer to a struct message
// msg_queue[i]->next points to the next msg in a linked list.
static struct message *shared msg_queue[THREADS];

// An array of locks to allow for atomic updates of the message queues
static upc_lock_t *shared msg_queue_lock[THREADS];

// Initialise Comms methods
void comm_init()
{
#ifdef MPI
  MPI_Init(0,0);
#elifdef __UPC__

  // Initialise the message queue and lock per thread.
  // OPT: msg_queue[MYTHREAD] = blah ... upc_barrier() would do the same sans loop.
  upc_forall (int i=0; i<THREADS; i++; i) {
    msg_queue[i] = NULL;
    msg_queue_lock[i] = NULL;
    msg_queue_lock[i] = upc_global_lock_alloc();
  }
  
#endif
}

void comm_init_check_(comm_ptr cp, comm_ptr ce, uint np, const char *file, unsigned line)
{
#ifdef HAVE_MPI
  /* comm_init(c,MPI_Comm_f2c(ce)); */
  comm_world(cp);
  if(cp->np != np)
    fail(1,file,line,"comm_init_check: passed P=%u, "
                     "but MPI_Comm_size gives P=%u",
                     (unsigned)np,(unsigned)c->np);
#else
  comm_world(&cp);
  if(np != 1)
    fail(1,file,line,"comm_init_check: passed P=%u, but not compiled with -DMPI",(unsigned)np);
#endif
}
#define comm_init_check(c,ce,np) comm_init_check_(c,ce,np,__FILE__,__LINE__)


// Tear down the msg queues and associated locks.
// Note, does not iterate through the msg_queue to verify that *msg_queue[i]->next is NULL
void comm_finalize()
{
#ifdef MPI
  MPI_Finalize();
#elifdef __UPC__

  upc_forall (int i=0; i<THREADS; i++; i) {

    upc_lock(msg_queue_lock[i]);
    //todo: msg_queue[i];
    msg_queue[i] = NULL;
    upc_unlock(msg_queue_lock[i]);
    
    upc_global_lock_free(msg_queue_lock[i]);
    msg_queue_lock[i] = NULL;
  }
  
#endif
}


// Allocate memory for the comms
// Take in a pointer to a communicator (comm_ptr is: struct comm*)
#ifdef __UPC__
int comm_alloc(comm_ptr cp, size_t n)
{
  int id = cp->id;
  int np = cp->np;
  shared[] char *tmp;


  // Sanitise inputs based on what's in cp's members
  if (n <= 0) {
    return 0;
  }
  
  if (cp->buf_len > 0 && cp->buf_len >= n) {
    return 0;
  }

  // If flgs is empty, allocate np int's in the shared memory region
  if (cp->flgs == NULL) {
    cp->flgs = upc_all_alloc(np, sizeof(int));
  }

  // If buf_dir is empty, allocate space for np pointers-to-shared-chars
  if (cp->buf_dir == NULL) {
    cp->buf_dir = upc_all_alloc(np, sizeof(shared[] char *shared));
  }

  // If buf_dir is NOT empty, tmp (a pointer to a shared char) is pointed at the id'th element in buf_dir
  if (cp->buf != NULL) {
    tmp = (shared[] char *) cp->buf_dir[id];
  }

  // Alloc up n BYTES for a buffer and stash the pointer to it at buf_dir[id]
#if defined(__GUPC__) || defined(__clang_upc__)
  cp->buf_dir[id] = upc_alloc(n);
#else
  cp->buf_dir[id] = (shared[] char *shared) upc_alloc(n);
#endif
  upc_barrier;

  // If buf_len isn't 0, ie, something was already present in buf_dir[id], then
  // copy from tmp to buf_dir[id] the bytes.
  // This is actually totally wrong as we never copied the CONTENTS to tmp, or ever allocated it space
  // That free looks like a bad idea too, as it doesn't have a matching alloc
  // It needs a trap.
  if (cp->buf_len != 0) {
    upc_memcpy(cp->buf_dir[id], tmp, cp->buf_len);
    upc_free(tmp);
  }

  // Make buf_len store the size of buffer allocated at each element of buf_dir
  cp->buf_len = n;

  // Pretend that the locally affine buffers are just chars...
  // This is a shared->private cast which loses the affinity and phase.
#if defined(__UPC_CASTABLE__)
  cp->buf = (char *) upc_cast(&cp->buf_dir[id][0]);
#else
  cp->buf = (char *) &cp->buf_dir[id][0];
#endif

  return 0;
}
#endif


// Manufacture a UPC equivalent of comm_world if UPC
// Make our communicator (cpp) understand comm_world if MPI
void comm_world(comm_ptr *cpp)
{
  if (NULL == cpp) return;

  comm_ptr cp = (comm_ptr) malloc(sizeof (struct comm));
  
  if (NULL != cp) {
    // manual says: MPI_Comm_rank fills 2nd arg with the rank of *this* process in the 1st arg
    // ie, what rank am I
#ifdef MPI
    cp->h = MPI_COMM_WORLD;
    MPI_Comm_size(cp->h,&(cp->np));
    MPI_Comm_rank(cp->h,&(cp->id));
#elif __UPC__
    cp->h = 0;
    cp->id = MYTHREAD;
    cp->np = THREADS;  
    cp->buf_len = 0;
    cp->buf_dir = NULL;
    cp->buf = NULL;
    cp->flgs = NULL;
#else
    cp->h = 0;
    cp->np = 0;
    cp->id = -1;
#endif

    *cpp = cp;
  }
}

// Duplicate a communicator
void comm_dup(comm_ptr *cpp, const comm_ptr cp)
{
  if (NULL == cpp || NULL == cp) return;

  comm_ptr cpd = (comm_ptr) malloc(sizeof (struct comm));

  if (NULL != cpd) {
#ifdef MPI
    MPI_Comm_dup(cp->h, &(cpd->h));
#else
    cpd->h = cp->h;
#endif

    cpd->np = cp->np;
    cpd->id = cp->id;

#ifdef __UPC__
    cpd->buf_len = 0;
    cpd->buf_dir = NULL;
    cpd->buf = NULL;
    cpd->flgs = NULL;
    if (cp->buf_len > 0) {
      comm_alloc(cpd,cp->buf_len);
      upc_memcpy(cpd->buf_dir[cpd->id], cp->buf_dir[cp->id], cpd->buf_len);
    }
#endif

    *cpp = cpd;
  }
}

// Free off a communicator
void comm_free(comm_ptr *cpp)
{
  if (NULL == cpp) return;

  comm_ptr cp = *cpp;

  if (NULL != cp) {
#ifdef MPI
    MPI_Comm_free(&(cp->h));
#elif __UPC__
    upc_barrier;
    if (cp->buf_dir) {
      upc_free(cp->buf_dir[cp->id]);
    }
    upc_barrier;
    if (cp->id == 0) {
      upc_free(cp->buf_dir);
      upc_free(cp->flgs);
    }
    cp->buf_dir = NULL;
    cp->buf = NULL;
    cp->buf_len = 0;
    cp->flgs = NULL;
#endif
    free(cp);
    *cpp = NULL;
  }
}

// Access function for np
void comm_np(const comm_ptr cp, int *np)
{
  if (NULL != cp && NULL != np) {
    *np = cp->np;
  }
}

// Access function for id
void comm_id(const comm_ptr cp, int *id)
{
  if (NULL != cp && NULL != id) {
    *id = cp->id;
  }
}

// Helper function to get the correct type information for the comms lib for int
void comm_type_int(comm_type *ct)
{
#ifdef MPI
  *ct = MPI_INTEGER;
#elif __UPC__
  *ct = UPC_INT;
#endif
}

// Helper function to get the correct type information for the comms lib for int8
void comm_type_int8(comm_type *ct)
{
  if (NULL == ct) return;
#ifdef MPI
  *ct = MPI_INTEGER8;
#elif __UPC__
  *ct = UPC_INT64;
#endif
}

// Helper function to get the correct type information for the comms lib for real
void comm_type_real(comm_type *ct)
{
  if (NULL == ct) return;
#ifdef MPI
  *ct = MPI_REAL;
#elif __UPC__
  *ct = UPC_FLOAT;
#endif
}

// Helper function to get the correct type information for the comms lib for dp
void comm_type_dp(comm_type *ct)
{
  if (NULL == ct) return;
#ifdef MPI
  *ct = MPI_DOUBLE_PRECISION;
#elif __UPC__
  *ct = UPC_DOUBLE;
#endif
}

// Helper function to get the correct tag information from the comms lib for UB, MPI ONLY
void comm_tag_ub(const comm_ptr cp, int *ub)
{
  if (NULL == cp || NULL == ub) return;
#ifdef MPI
  int val = 0, flag = 0;
  MPI_Attr_get(cp->h,MPI_TAG_UB,&val,&flag);
  *ub = val;
#else
  *ub = 0;
#endif
}


// Helper function to get the time for timing routines.
void comm_time(double *tm)
{
  if (NULL == tm) return;
#ifdef MPI
  *tm = MPI_Wtime();
#else
  time_t timer;
  time(&timer);
  *tm = (double) timer; 
#endif
}

// Helper function for a comms barrier
void comm_barrier(const comm_ptr cp)
{
  if (NULL == cp) return;
#ifdef MPI
  MPI_Barrier(cp->h);
#elif __UPC__
  upc_barrier;
#endif
}

// Broadcast n BYTES from root to other processes
void comm_bcast(const comm_ptr cp, void *p, size_t n, uint root)
{
  if (NULL == cp || NULL == p) return;
  
#ifdef MPI
  MPI_Bcast(p,n,MPI_BYTE,root,cp->h);
#elif __UPC__
  n = n*sizeof(char);

  // Make a temporary space of size np*n (bytes)
  shared char *dst = upc_all_alloc(cp->np, n);
  if (NULL == dst) return;

  // If this process is root, cast csrc as char*, local affinity shuffle trick
  // THEN you can just memcpy from src to dst
  if (root == cp->id) {    
#if defined(__UPC_CASTABLE__)
    char *csrc = (char *) upc_cast(dst + root*n);
#else
    char *csrc = (char *) (dst + root*n);
#endif
    memcpy(csrc, p, n);
  }

  // Do the UPC boradcast where we make dst (on all threads, equal to the n bytes starting from dst[root*n]
  upc_all_broadcast(dst, &dst[root*n], n, UPC_IN_ALLSYNC | UPC_OUT_ALLSYNC);

  if (root != cp->id) {
#if defined(__UPC_CASTABLE__)
    char *cdst = (char *) upc_cast(dst);
#else
    char *cdst = (char *) dst;
#endif
    memcpy(p, cdst, n);
  }

  upc_all_free(dst);
#endif
}


// Implementation of comm_send for UPC
// This seems whack, doing a send/recv using a single sided model...
void comm_send(const comm_ptr cp, void *p, size_t n, uint dst, int tag)
{
#ifdef MPI
  MPI_Send(p,n,MPI_UNSIGNED_CHAR,dst,tag,cp->h);
#elif __UPC__ 

  // Make space for a message, locally affine but in shared space, not collective
  struct message *msg = (struct message*) upc_alloc(sizeof(struct message));

  if (NULL == msg) return;

  // Fill in the message structure, to mimic what would be contained in an MPI message
  msg->src = MYTHREAD;
  msg->tag = tag;
  msg->len = n;
  msg->data = (void*) upc_alloc(n);
  memcpy(msg->data, p, n);
  msg->next = NULL;

  // Then just stuff the message at the next point in the queue of the receiving process
  upc_lock(msg_queue_lock[dst]);

  struct message *queue = msg_queue[dst];

  if (NULL != queue) {
    while (NULL != queue->next) {
      queue = queue->next;
    }
    queue->next = msg;
  }
  else {
    msg_queue[dst] = msg;
  } 
  
  upc_unlock(msg_queue_lock[dst]);
  
#endif
}

// Implementation of MPI_Recv in UPC...
void comm_recv(const comm_ptr cp, void *p, size_t n, uint src, int tag)
{
#ifdef MPI
# ifndef MPI_STATUS_IGNORE
  MPI_Status stat;
  MPI_Recv(p,n,MPI_UNSIGNED_CHAR,src,tag,cp->h,&stat);
# else  
  MPI_Recv(p,n,MPI_UNSIGNED_CHAR,src,tag,cp->h,MPI_STATUS_IGNORE);
# endif
#elif __UPC__

  upc_lock(msg_queue_lock[MYTHREAD]);

  // Get a quick pointer to OUR queue
  // Start reading messages
  struct message *msg = msg_queue[MYTHREAD];
  struct message *prev_msg = NULL;
  int recvd = 0;
  // Whilst there is at least one message
  while (NULL != msg && 0 == recvd) {
    // If the source, length and tag match what we expect, copy from msg->data to p
    if (src == msg->src &&
        tag == msg->tag &&
	  n == msg->len) {
	  
        memcpy(p, msg->data, n);
	recvd = 1;
	
    }
    else {
      // Otherwise, advance through to the next message in OUR queue
      prev_msg = msg;
      msg = msg->next;
    }
  }

  // Do a linked list removal, or mark emptied queue if necessary
  // Do housekeeping on messages.
  if (0 != recvd) {
    if (NULL != prev_msg) {
      prev_msg->next = msg->next;
    }
    else {
      msg_queue[MYTHREAD] = NULL;
    }

    free(msg->data);
    free(msg);
  }
  
  upc_unlock(msg_queue_lock[MYTHREAD]);  
  
#endif
}


uint comm_gbl_id=0, comm_gbl_np=1;


// Init the GATHER SCATTER code.
GS_DEFINE_IDENTITIES()
GS_DEFINE_DOM_SIZES()

// Implementation of SCAN
static void scan_imp(void *scan, const comm_ptr cp, gs_dom dom, gs_op op,
                     const void *v, uint vn, void *buffer)
{
  // If comms handle invalid, quit
  if (NULL == cp) return;

  // We should be clear about exactly which implementation of SCAN this is.
  comm_req req[2];
  size_t vsize = vn*gs_dom_size[dom];
  const uint id=cp->id, np=cp->np;
  uint n = np, c=1, odd=0, base=0;
  void *buf[2];
  void *red = (char*)scan+vsize;
  buf[0]=buffer,buf[1]=(char*)buffer+vsize;
  while(n>1) {
    odd=(odd<<1)|(n&1);
    c<<=1, n>>=1;
    if(id>=base+n) c|=1, base+=n, n+=(odd&1);
  }
  gs_init_array(scan,vn,dom,op);
  memcpy(red,v,vsize);
  while(n<np) {
    if(c&1) n-=(odd&1), base-=n;
    c>>=1, n<<=1, n+=(odd&1);
    odd>>=1;
    if(base==id) {
      comm_irecv(&req[0],cp, buf[0],vsize, id+n/2,id+n/2);
      comm_isend(&req[1],cp, red   ,vsize, id+n/2,id);
      comm_wait(req,2);
      gs_gather_array(red,buf[0],vn,dom,op);
    } else {
      comm_irecv(&req[0],cp, scan,vsize, base,base);
      comm_isend(&req[1],cp, red ,vsize, base,id);
      comm_wait(req,2);
      break;
    }
  }
  while(n>1) {
    if(base==id) {
      comm_send(cp, scan  ,2*vsize, id+n/2,id);
    } else {
      comm_recv(cp, buffer,2*vsize, base,base);
      gs_gather_array(scan,buf[0],vn,dom,op);
      memcpy(red,buf[1],vsize);
    }
    odd=(odd<<1)|(n&1);
    c<<=1, n>>=1;
    if(id>=base+n) c|=1, base+=n, n+=(odd&1);
  }
}

// Allreduce
static void allreduce_imp(const comm_ptr cp, gs_dom dom, gs_op op,
                          void *v, uint vn, void *buf)
{
  if (NULL == cp) return;
  
  size_t total_size = vn*gs_dom_size[dom];
  const uint np=cp->np, id=cp->id;
  uint n = np, c=1, odd=0, base=0;
  while(n>1) {
    odd=(odd<<1)|(n&1);
    c<<=1, n>>=1;
    if(id>=base+n) c|=1, base+=n, n+=(odd&1);
  }
  while(n<np) {
    if(c&1) n-=(odd&1), base-=n;
    c>>=1, n<<=1, n+=(odd&1);
    odd>>=1;
    if(base==id) {
      comm_recv(cp, buf,total_size, id+n/2,id+n/2);
      gs_gather_array(v,buf,vn, dom,op);
    } else {
      comm_send(cp, v,total_size, base,id);
      break;
    }
  }
  while(n>1) {
    if(base==id)
      comm_send(cp, v,total_size, id+n/2,id);
    else
      comm_recv(cp, v,total_size, base,base);
    odd=(odd<<1)|(n&1);
    c<<=1, n>>=1;
    if(id>=base+n) c|=1, base+=n, n+=(odd&1);
  }
}

// Helper function to call scan on a communicator
void comm_scan(void *scan, const comm_ptr cp, gs_dom dom, gs_op op,
               const void *v, uint vn, void *buffer)
{
#ifdef HAVE_MPI
  scan_imp(scan, com,dom,op, v,vn, buffer);
#elif __UPC__
  int d;
  uint D;
  size_t vsize = vn*gs_dom_size[dom]; 
  void *red = (char *)scan + vsize;
  upc_barrier;

  memset(scan, 0, 2 * vsize);
  comm_alloc(cp, vn*gs_dom_size[dom]); 
  upc_barrier;
  memcpy(buffer,v, vn*gs_dom_size[dom]);
    
  cp->flgs[MYTHREAD] = -1;    
  D = ceil(log2(THREADS));
  upc_barrier;

  for (d = 0; d < D; d++) {

    if ((MYTHREAD + (1<<d)) < THREADS) {
      while(cp->flgs[MYTHREAD+(1<<d)] != (d-1)) ;
      upc_memput(cp->buf_dir[MYTHREAD+(1<<d)], buffer, vn*gs_dom_size[dom]);
      cp->flgs[MYTHREAD+(1<<d)] = -2;
    }

    if ((MYTHREAD - (1<<d)) >= 0) {      
      while(cp->flgs[MYTHREAD] != -2) ;    
      gs_gather_array(scan, cp->buf, vn, dom, op);
      gs_gather_array(buffer, cp->buf, vn, dom, op);
    }
    cp->flgs[MYTHREAD] = d;
  }

  upc_barrier;
  comm_allreduce(cp, dom, op, v, vn, red);  
#endif
}

// Helper function to call allreduce on a communicator with a TYPE
void comm_allreduce_cdom(const comm_ptr cp, comm_type cdom, gs_op op,
                         void *v, uint vn, void *buf)
{
  if (NULL == cp || NULL == v || NULL == buf) return;
  
  gs_dom dom;
  int dom_ok = 1;

  switch(cdom) {
#ifdef MPI
    case MPI_INTEGER:          dom = gs_int; break;
    case MPI_INTEGER8:         dom = gs_long; break;
    case MPI_REAL:             dom = gs_float; break;
    case MPI_DOUBLE_PRECISION: dom = gs_double; break;
#elif __UPC__
    case UPC_INT:    dom = gs_int; break;
    case UPC_INT64:  dom = gs_long; break;
    case UPC_FLOAT:  dom = gs_float; break;
    case UPC_DOUBLE: dom = gs_double; break;
#endif
    default: dom_ok = 0;
  }

  if (dom_ok == 1) {
    comm_allreduce(cp,dom,op,v,vn,buf);
  }
  else {
    fail(1,__FILE__,__LINE__,
      "comm_allreduce_cdom: cannot identify cdom P=%u.",cdom);
  }
}


// Implementation of allreduce, once the type has been been deduced.
void comm_allreduce(const comm_ptr cp, gs_dom dom, gs_op op,
                    void *v, uint vn, void *buf)
{
  if (NULL == cp || 0 == vn) return;
    
#ifdef MPI
  {
    MPI_Datatype mpitype;
    MPI_Op mpiop;
    #define DOMAIN_SWITCH() do { \
      switch(dom) { case gs_double:    mpitype=MPI_DOUBLE;    break; \
                    case gs_float:     mpitype=MPI_FLOAT;     break; \
                    case gs_int:       mpitype=MPI_INT;       break; \
                    case gs_long:      mpitype=MPI_LONG;      break; \
     WHEN_LONG_LONG(case gs_long_long: mpitype=MPI_LONG_LONG; break;) \
                  default:        goto comm_allreduce_byhand; \
      } \
    } while(0)
    DOMAIN_SWITCH();
    #undef DOMAIN_SWITCH
    switch(op) { case gs_add: mpiop=MPI_SUM;  break;
                 case gs_mul: mpiop=MPI_PROD; break;
                 case gs_min: mpiop=MPI_MIN;  break;
                 case gs_max: mpiop=MPI_MAX;  break;
                 default:        goto comm_allreduce_byhand;
    }
    MPI_Allreduce(v,buf,vn,mpitype,mpiop,cp->h);
    memcpy(v,buf,vn*gs_dom_size[dom]);
    return;
  }
  
#elif __UPC__
  /*
  upc_type_t ucp_type = UPC_CHAR;
  switch (dom) {
    case gs_double:
      ucp_type = UPC_DOUBLE;
      break;
    case gs_float:
      ucp_type = UPC_FLOAT;
      break;
    case gs_int:
      ucp_type = UPC_INT;
      break;
    case gs_long:
      ucp_type = UPC_LONG;
      break;
    case gs_long_long:
      ucp_type = UPC_LLONG;
      break;
    default: printf("Warning, comm_allreduce defaulting to %d.\n", ucp_type);
  }
		  
  upc_op_t upc_op = UPC_ADD;
  switch (op) {
    case gs_mul:
      upc_op = UPC_MULT;
      break;
    case gs_min:
      upc_op = UPC_MIN;
      break;
    case gs_max:
      upc_op = UPC_MAX;
      break;
    default: printf("Warning, comm_allreduce defaulting to %d.\n", upc_op);
  }


  switch (ucp_type) {
    case UPC_DOUBLE:
      ucp_type = UPC_DOUBLE;
      break;
    case UPC_FLOAT:
      ucp_type = UPC_FLOAT;
      break;
    case UPC_INT:
      ucp_type = UPC_INT;
      break;
    case UPC_LONG:
      ucp_type = UPC_LONG;
      break;
    case UPC_LLONG:
      ucp_type = UPC_LLONG;
      break;
    default: printf("Warning, comm_allreduce defaulting to %d.\n", ucp_type);
  }
  */
  {    
    int np = cp->np, id = cp->id;
    uint D, d, i;

    // Copy v into buf, assume buf can hold vn*gs_dom_size[dom] bytes
    memcpy(buf,v,vn*gs_dom_size[dom]);

    if (np == 1) return;

    // Make a communicator for this operation, each entry in cp->buf_dir should hold vn*gs_dom_size[dom] bytes
    comm_alloc(cp, vn*gs_dom_size[dom]); /* Fixme const comm... */

    // Do flags
    cp->flgs[id] = -10;
    D = floor(log2(np));
    upc_barrier;
        
    if (id >= (1<<D)) {
      while(cp->flgs[id^(1<<D)] != -10) ;
      upc_memput(cp->buf_dir[id^(1<<D)], buf, vn*gs_dom_size[dom]);
      cp->flgs[id^(1<<D)] = -5;
    }
    
    if (id < (np - (1<<D))) {
      while(cp->flgs[id] != -5) ;
      gs_gather_array(buf, cp->buf, vn, dom, op);
    }

    cp->flgs[id] = -1;    

    if (id < (1<<D)) {
      for (d = 0; d < D; d++) {

	while(cp->flgs[id^(1<<d)] != (d-1)) ;
	upc_memput(cp->buf_dir[id^(1<<d)], buf, vn*gs_dom_size[dom]);
	cp->flgs[id^(1<<d)] = -2;

	while(cp->flgs[id] != -2) ;
	gs_gather_array(buf, cp->buf, vn, dom, op);
	cp->flgs[id] = d;
      }
    }

    if (id < (np - (1<<D))) {
      upc_memput(cp->buf_dir[id^(1<<D)], buf, vn*gs_dom_size[dom]);
      cp->flgs[id^(1<<D)] = -20;
    }

    if (id >= (1<<D)) {
      while(cp->flgs[id] != -20) ;
      memcpy(buf, cp->buf, vn*gs_dom_size[dom]);
    }

    memcpy(v,buf,vn*gs_dom_size[dom]);
    return;
  }
#endif


#if (defined MPI || defined __UPC__)
comm_allreduce_byhand:
  allreduce_imp(cp,dom,op, v,vn, buf);
#endif
}

// Helper to call comm_allreduce with value of tensor_dot
// TODO: What do this do, investigate
double comm_dot(const comm_ptr cp, double *v, double *w, uint n)
{
  double s=tensor_dot(v,w,n),b;
  comm_allreduce(cp,gs_double,gs_add, &s,1, &b);
  return s;
}

/* T comm_reduce__T(const comm_ptr cp, gs_op op, const T *in, uint n) */

#define SWITCH_OP_CASE(T,OP) case gs_##OP: WITH_OP(T,OP); break;
#define SWITCH_OP(T,op) do switch(op) { \
    GS_FOR_EACH_OP(T,SWITCH_OP_CASE) case gs_op_n: break; } while(0)

#define WITH_OP(T,OP) \
  do { T v = *in++; GS_DO_##OP(accum,v); } while(--n)

#define DEFINE_REDUCE(T) \
T PREFIXED_NAME(comm_reduce__##T)( \
    const comm_ptr cp, gs_op op, const T *in, uint n) \
{                                                           \
  T accum = gs_identity_##T[op], buf;                       \
  if(n!=0) SWITCH_OP(T,op);                                 \
  comm_allreduce(cp,gs_##T,op, &accum,1, &buf);           \
  return accum;                                             \
}

GS_FOR_EACH_DOMAIN(DEFINE_REDUCE)

#undef DEFINE_REDUCE
#undef WITH_OP
#undef SWITCH_OP
#undef SWITCH_OP_CASE



/*------------------------------------------------------------------------------
  FORTRAN interface
------------------------------------------------------------------------------*/

#undef comm_init
#undef comm_finalize
#undef comm_world
#undef comm_free
#undef comm_np
#undef comm_id
#undef comm_type_int
#undef comm_type_int8
#undef comm_type_real
#undef comm_type_dp
#undef comm_tag_ub
#undef comm_time
#undef comm_barrier
#undef comm_bcast
#undef comm_allreduce_add
#undef comm_allreduce_min
#undef comm_allreduce_max
#undef comm_allreduce_mul

#define ccomm_init      PREFIXED_NAME(comm_init     )
#define ccomm_finalize  PREFIXED_NAME(comm_finalize )
#define ccomm_world     PREFIXED_NAME(comm_world)
#define ccomm_free      PREFIXED_NAME(comm_free     )
#define ccomm_np        PREFIXED_NAME(comm_np       )
#define ccomm_id        PREFIXED_NAME(comm_id       )
#define ccomm_type_int  PREFIXED_NAME(comm_type_int )
#define ccomm_type_int8 PREFIXED_NAME(comm_type_int8)
#define ccomm_type_real PREFIXED_NAME(comm_type_real)
#define ccomm_type_dp   PREFIXED_NAME(comm_type_dp  )
#define ccomm_tag_ub    PREFIXED_NAME(comm_tag_ub   )
#define ccomm_time      PREFIXED_NAME(comm_time     )
#define ccomm_barrier   PREFIXED_NAME(comm_barrier  )
#define ccomm_bcast     PREFIXED_NAME(comm_bcast    )
#define ccomm_allreduce_add PREFIXED_NAME(comm_allreduce_add)
#define ccomm_allreduce_min PREFIXED_NAME(comm_allreduce_min)
#define ccomm_allreduce_max PREFIXED_NAME(comm_allreduce_max)
#define ccomm_allreduce_mul PREFIXED_NAME(comm_allreduce_mul)

#define fcomm_init      FORTRAN_NAME(comm_init     , COMM_INIT     )
#define fcomm_finalize  FORTRAN_NAME(comm_finalize , COMM_FINALIZE )
#define fcomm_world     FORTRAN_NAME(comm_world    , COMM_WORLD    )
#define fcomm_free      FORTRAN_NAME(comm_free     , COMM_FREE     )
#define fcomm_np        FORTRAN_NAME(comm_np       , COMM_NP       )
#define fcomm_id        FORTRAN_NAME(comm_id       , COMM_ID       )
#define fcomm_type_int  FORTRAN_NAME(comm_type_int , COMM_TYPE_INT )
#define fcomm_type_int8 FORTRAN_NAME(comm_type_int8, COMM_TYPE_INT8)
#define fcomm_type_real FORTRAN_NAME(comm_type_real, COMM_TYPE_REAL)
#define fcomm_type_dp   FORTRAN_NAME(comm_type_dp  , COMM_TYPE_DP  )
#define fcomm_tag_ub    FORTRAN_NAME(comm_tag_ub   , COMM_TAG_UB   )
#define fcomm_time      FORTRAN_NAME(comm_time     , COMM_TIME     )
#define fcomm_barrier   FORTRAN_NAME(comm_barrier  , COMM_BARRIER  )
#define fcomm_bcast     FORTRAN_NAME(comm_bcast    , COMM_BCAST    )
#define fcomm_allreduce_add FORTRAN_NAME(comm_allreduce_add, COMM_ALLREDUCE_ADD)
#define fcomm_allreduce_min FORTRAN_NAME(comm_allreduce_min, COMM_ALLREDUCE_MIN)
#define fcomm_allreduce_max FORTRAN_NAME(comm_allreduce_max, COMM_ALLREDUCE_MAX)
#define fcomm_allreduce_mul FORTRAN_NAME(comm_allreduce_mul, COMM_ALLREDUCE_PROD)


void fcomm_init(void)
{
  ccomm_init();
}

void fcomm_finalize(void)
{
  ccomm_finalize();
}


void fcomm_world(comm_ptr *cpp)
{
  ccomm_world(cpp);
}

void fcomm_free(comm_ptr *cpp)
{
  ccomm_free(cpp);
}


void fcomm_np(const comm_ptr *cpp, int *np)
{
  ccomm_np(*cpp, np);
}

void fcomm_id(const comm_ptr *cpp, int *id)
{
  ccomm_id(*cpp, id);
}


void fcomm_type_int(comm_type *ct)
{
  ccomm_type_int(ct);
}

void fcomm_type_int8(comm_type *ct)
{
  ccomm_type_int8(ct);
}

void fcomm_type_real(comm_type *ct)
{
  ccomm_type_real(ct);
}

void fcomm_type_dp(comm_type *ct)
{
  ccomm_type_dp(ct);
}

void fcomm_tag_ub(const comm_ptr *cpp, int *ub)
{
  ccomm_tag_ub(*cpp, ub);
}

void fcomm_time(double *tm)
{
  ccomm_time(tm);
}


void fcomm_barrier(const comm_ptr *cpp)
{
  ccomm_barrier(*cpp);
}

void fcomm_bcast(const comm_ptr *cpp, void *p, size_t *n, uint *root)
{
  ccomm_bcast(*cpp,p,*n,*root);
}


void fcomm_allreduce_add(const comm_ptr *cpp, comm_type *ct,
                         void *v, uint *vn, void *buf)
{
  comm_allreduce_cdom(*cpp,*ct,gs_add,v,*vn,buf);
}

void fcomm_allreduce_min(const comm_ptr *cpp, comm_type *ct,
                         void *v, uint *vn, void *buf)
{
  comm_allreduce_cdom(*cpp,*ct,gs_min,v,*vn,buf);
}

void fcomm_allreduce_max(const comm_ptr *cpp, comm_type *ct,
                         void *v, uint *vn, void *buf)
{
  comm_allreduce_cdom(*cpp,*ct,gs_max,v,*vn,buf);
}

void fcomm_allreduce_mul(const comm_ptr *cpp, comm_type *ct,
                         void *v, uint *vn, void *buf)
{
  comm_allreduce_cdom(*cpp,*ct,gs_mul,v,*vn,buf);
}

