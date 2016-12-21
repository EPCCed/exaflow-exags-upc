#include <stddef.h> /* for size_t */
#include <stdlib.h> /* for exit */
#include <string.h> /* memcpy */
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

uint comm_gbl_id=0, comm_gbl_np=1;

GS_DEFINE_IDENTITIES()
GS_DEFINE_DOM_SIZES()

static void scan_imp(void *scan, const struct comm *com, gs_dom dom, gs_op op,
                     const void *v, uint vn, void *buffer)
{
  comm_req req[2];
  size_t vsize = vn*gs_dom_size[dom];
  const uint id=com->id, np=com->np;
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
      comm_irecv(&req[0],com, buf[0],vsize, id+n/2,id+n/2);
      comm_isend(&req[1],com, red   ,vsize, id+n/2,id);
      comm_wait(req,2);
      gs_gather_array(red,buf[0],vn,dom,op);
    } else {
      comm_irecv(&req[0],com, scan,vsize, base,base);
      comm_isend(&req[1],com, red ,vsize, base,id);
      comm_wait(req,2);
      break;
    }
  }
  while(n>1) {
    if(base==id) {
      comm_send(com, scan  ,2*vsize, id+n/2,id);
    } else {
      comm_recv(com, buffer,2*vsize, base,base);
      gs_gather_array(scan,buf[0],vn,dom,op);
      memcpy(red,buf[1],vsize);
    }
    odd=(odd<<1)|(n&1);
    c<<=1, n>>=1;
    if(id>=base+n) c|=1, base+=n, n+=(odd&1);
  }
}


static void allreduce_imp(const struct comm *com, gs_dom dom, gs_op op,
                          void *v, uint vn, void *buf)
{
  size_t total_size = vn*gs_dom_size[dom];
  const uint id=com->id, np=com->np;
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
      comm_recv(com, buf,total_size, id+n/2,id+n/2);
      gs_gather_array(v,buf,vn, dom,op);
    } else {
      comm_send(com, v,total_size, base,id);
      break;
    }
  }
  while(n>1) {
    if(base==id)
      comm_send(com, v,total_size, id+n/2,id);
    else
      comm_recv(com, v,total_size, base,base);
    odd=(odd<<1)|(n&1);
    c<<=1, n>>=1;
    if(id>=base+n) c|=1, base+=n, n+=(odd&1);
  }
}

void comm_scan(void *scan, const struct comm *com, gs_dom dom, gs_op op,
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
  comm_alloc(com, vn*gs_dom_size[dom]); 
  memcpy(buffer,v, vn*gs_dom_size[dom]);
    
  com->flgs[MYTHREAD] = -1;    
  D = ceil(log2(THREADS));
  upc_barrier;

  for (d = 0; d < D; d++) {

    if ((MYTHREAD + (1<<d)) < THREADS) {
      while(com->flgs[MYTHREAD+(1<<d)] != (d-1)) ;
      upc_memput(com->buf_dir[MYTHREAD+(1<<d)], buffer, vn*gs_dom_size[dom]);
      com->flgs[MYTHREAD+(1<<d)] = -2;
    }

    if ((MYTHREAD - (1<<d)) >= 0) {      
      while(com->flgs[MYTHREAD] != -2) ;    
      gs_gather_array(scan, com->buf, vn, dom, op);
      gs_gather_array(buffer, com->buf, vn, dom, op);
    }
    com->flgs[MYTHREAD] = d;
  }
  
  upc_barrier;
  comm_allreduce(com, dom, op, v, vn, red);  
#endif
}

void comm_allreduce(const struct comm *com, gs_dom dom, gs_op op,
                          void *v, uint vn, void *buf)
{
  uint D, d, i;
  comm_cop col_op;
  if(vn==0) return;
#ifdef HAVE_MPI
  {
    MPI_Datatype mpitype;
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
    switch(op) { case gs_add: col_op=MPI_SUM;  break;
                 case gs_mul: col_op=MPI_PROD; break;
                 case gs_min: col_op=MPI_MIN;  break;
                 case gs_max: col_op=MPI_MAX;  break;
                 default:        goto comm_allreduce_byhand;
    }
    MPI_Allreduce(v,buf,vn,mpitype,col_op,com->c);
    memcpy(v,buf,vn*gs_dom_size[dom]);
    return;
  }
#elif __UPC__
  {
    memcpy(buf,v, vn*gs_dom_size[dom]);

    if (THREADS == 1) return;

    comm_alloc(com, vn*gs_dom_size[dom]); /* Fixme const comm... */

    com->flgs[MYTHREAD] = -10;
    D = floor(log2(THREADS));
    upc_barrier;
        
    if (MYTHREAD >= (1<<D)) {
      while(com->flgs[MYTHREAD^(1<<D)] != -10) ;
      upc_memput(com->buf_dir[MYTHREAD^(1<<D)], buf, vn*gs_dom_size[dom]);
      com->flgs[MYTHREAD^(1<<D)] = -5;
    }
    
    if (MYTHREAD < (THREADS - (1<<D))) {
      while(com->flgs[MYTHREAD] != -5) ;
      gs_gather_array(buf, com->buf, vn, dom, op);
    }

    com->flgs[MYTHREAD] = -1;    

    if (MYTHREAD < (1<<D)) {
      for (d = 0; d < D; d++) {

	while(com->flgs[MYTHREAD^(1<<d)] != (d-1)) ;
	upc_memput(com->buf_dir[MYTHREAD^(1<<d)], buf, vn*gs_dom_size[dom]);
	com->flgs[MYTHREAD^(1<<d)] = -2;

	while(com->flgs[MYTHREAD] != -2) ;
	gs_gather_array(buf, com->buf, vn, dom, op);
	com->flgs[MYTHREAD] = d;
      }
    }

    if (MYTHREAD < (THREADS - (1<<D))) {
      upc_memput(com->buf_dir[MYTHREAD^(1<<D)], buf, vn*gs_dom_size[dom]);
      com->flgs[MYTHREAD^(1<<D)] = -20;
    }

    if (MYTHREAD >= (1<<D)) {
      while(com->flgs[MYTHREAD] != -20) ;
      memcpy(buf, com->buf, vn*gs_dom_size[dom]);
    }

    memcpy(v,buf,vn*gs_dom_size[dom]);
    return;
  }
#endif


#if (defined HAVE_MPI || defined __UPC__)
comm_allreduce_byhand:
  allreduce_imp(com,dom,op, v,vn, buf);
#endif
}

double comm_dot(const struct comm *comm, double *v, double *w, uint n)
{
  double s=tensor_dot(v,w,n),b;
  comm_allreduce(comm,gs_double,gs_add, &s,1, &b);
  return s;
}

/* T comm_reduce__T(const struct comm *comm, gs_op op, const T *in, uint n) */

#define SWITCH_OP_CASE(T,OP) case gs_##OP: WITH_OP(T,OP); break;
#define SWITCH_OP(T,op) do switch(op) { \
    GS_FOR_EACH_OP(T,SWITCH_OP_CASE) case gs_op_n: break; } while(0)

#define WITH_OP(T,OP) \
  do { T v = *in++; GS_DO_##OP(accum,v); } while(--n)

#define DEFINE_REDUCE(T) \
T PREFIXED_NAME(comm_reduce__##T)( \
    const struct comm *comm, gs_op op, const T *in, uint n) \
{                                                           \
  T accum = gs_identity_##T[op], buf;                       \
  if(n!=0) SWITCH_OP(T,op);                                 \
  comm_allreduce(comm,gs_##T,op, &accum,1, &buf);           \
  return accum;                                             \
}

GS_FOR_EACH_DOMAIN(DEFINE_REDUCE)

#undef DEFINE_REDUCE
#undef WITH_OP
#undef SWITCH_OP
#undef SWITCH_OP_CASE

