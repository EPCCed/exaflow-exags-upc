#include <stddef.h> 
#include <stdlib.h> 
#include <stdio.h>
#include <string.h>
#include <c99.h>
#include <name.h>
#include <fail.h>
#include <types.h>
#include <comm.h>
#include <mem.h>
#include <crs.h>

#define M 3

#ifdef HAVE_CHECK

#include <check.h>

void setup() {
}

void teardown() {
}


START_TEST(test_init) {
  uint n; ulong *xid;
  uint nz; uint *Ai, *Aj; double *A;
  uint i;
  double *x, *b, *x2;

  struct crs_xxt_data *crs;
  int id,np;
  comm_ptr cp;

  comm_init();
  comm_world(&cp);

  id = cp->id;
  np = cp->np;

  n = M+1; 
  if(cp->id == (cp->np-1)) --n;
  xid = tmalloc(ulong, n); x = tmalloc(double, 3 * n);
  b = x + n;
  x2 = b + n;

  for(i = 0;i < n; ++i) 
    xid[i] = 1 + cp->id * M + i;
 
  nz = 2 * M; 
  if(cp->id == (cp->np-1)) --nz;
  Ai = tmalloc(uint, 2 * nz);
  Aj = Ai + nz;
  A = tmalloc(double, nz);
  for(i = 0; i < M; ++i) {
    Ai[i] = i;
    Aj[i] = i;
    A[i]=2;
  }

  if(cp->id == 0) A[0]=1;
  if(cp->id == (cp->np-1)) A[n-1]=1;
  for(i = M; i < nz; ++i) {
    Ai[i] = i - M;
    Aj[i] = i - M + 1;
    A[i] = -1;
  }
  crs = crs_xxt_setup(n,xid, nz,Ai,Aj,A, 1, cp);

} END_TEST

Suite *xxt_suite() {
  TCase *tc;
  Suite *s;
  
  s = suite_create("xxt");

  tc = tcase_create("xxt_init");
  tcase_add_test(tc, test_init);
  suite_add_tcase(s, tc);

  return s;    
}

int main(void) {
  int number_failed;
  Suite *s = xxt_suite();
  SRunner *sr = srunner_create(s);
  setup();
  srunner_run_all (sr, CK_NORMAL);
  teardown();
  number_failed = srunner_ntests_failed (sr);
  srunner_free (sr);
  return (number_failed == 0) ? 0 : 1;
}

#else

int main(void) {
  fprintf(stderr, "*** Check is required for XXT tests ***\n");
  return 0;
}

#endif
