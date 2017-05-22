#include <stddef.h> 
#include <stdlib.h> 
#include <stdio.h>
#include <name.h>
#include <fail.h>
#include <types.h>
#include <gs_defs.h>
#include <comm.h>


#ifdef MPI
#include <mpi.h>
#endif


#ifdef HAVE_CHECK

#include <check.h>

void setup() {
  int rank;
#ifdef MPI
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#elif __UPC__
  rank = MYTHREAD;
#endif
  //  if (rank > 0)
    //    freopen ("/dev/null", "w", stdout);
}

void teardown() {
#ifdef MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();  
#endif
}

START_TEST(test_init) {
  comm_ptr cp;
  int np;

  comm_init();

  comm_world(&cp);


#ifdef MPI
  MPI_Comm_size(cp->h, &np);
  fail_unless(cp->np == np);  
#elif defined(__UPC__)
  fail_unless(cp->np == THREADS);
  fail_unless(cp->id == MYTHREAD);
  fail_unless(cp->buf_dir == NULL);
  fail_unless(cp->buf == NULL);
  fail_unless(cp->buf_len == 0);
  fail_unless(cp->flgs == NULL);
#endif

} END_TEST

START_TEST(test_alloc) {
  comm_ptr cp;
  size_t n;
  int i;

  comm_init();
  comm_world(&cp);
#ifdef __UPC__

  n = 42 * sizeof(char);  
  comm_alloc(cp, n);
  fail_unless(cp->buf_dir != NULL);
  fail_unless(cp->buf_dir[MYTHREAD] != NULL);
  fail_unless(cp->buf != NULL);
  fail_unless(cp->buf_len == n);

  memset(cp->buf, 1, n);


  n = 1147 * sizeof(char);
  comm_alloc(cp, n);
  fail_unless(cp->buf_dir != NULL);
  fail_unless(cp->buf_dir[MYTHREAD] != NULL);
  fail_unless(cp->buf != NULL);
  fail_unless(cp->buf_len == n);
  
  for (i = 0; i < 42; i++)
    fail_unless(cp->buf[i] == 1);

  n = 10 * sizeof(char);
  comm_alloc(cp, n);
  fail_unless(cp->buf_dir != NULL);
  fail_unless(cp->buf_dir[MYTHREAD] != NULL);
  fail_unless(cp->buf != NULL);
  fail_unless(cp->buf_len != n);

  for (i = 0; i < 42; i++)
    fail_unless(cp->buf[i] == 1);
#endif
} END_TEST


START_TEST(test_free) {
  comm_ptr cp;
  comm_ptr cp_old;

  comm_init();
  comm_world(&cp);
  cp_old = &cp;
  comm_free(&cp);
  fail_unless(cp == NULL);
  comm_free(&cp);  
  fail_unless(cp == NULL);

} END_TEST

START_TEST(test_reduce) {
  comm_ptr cp;
  int int_v, int_glb;
  long long_v, long_glb;
  float float_v, float_glb;
  double double_v, double_glb;

  int np;

  comm_init();
  comm_world(&cp);

  np = cp->np;
  
  int_v = cp->id + 1;
  int_glb = comm_reduce_int(cp, gs_add, (int *) &int_v, 1);
  fail_unless(int_glb == (np * (np + 1)>>1));

  long_v = cp->id + 1;
  long_glb = comm_reduce_long(cp, gs_add, (long *) &long_v, 1);
  fail_unless(long_glb == (np * (np + 1)>>1));

  float_v = (float) cp->id + 1;
  float_glb= comm_reduce_float(cp, gs_add, (float *) &float_v, 1);
  fail_unless(float_glb == (np * (np + 1)>>1));

  double_v= cp->id + 1;
  double_glb = comm_reduce_double(cp, gs_add, (double *) &double_v, 1);
  fail_unless(double_glb == (np * (np + 1)>>1));

  comm_free(&cp);
} END_TEST

START_TEST(test_allreduce) {
  comm_ptr cp;
  int int_v[2], int_glb[2];
  long long_v[2], long_glb[2];
  float float_v[2], float_glb[2];
  double double_v[2], double_glb[2];
  int size;

  comm_init();
  comm_world(&cp);

  size = cp->np;
  
  int_v[0] = cp->id + 1;
  comm_allreduce(cp, gs_int, gs_add, int_v, 1, int_glb);
  fail_unless(int_glb[0] == (size * (size + 1))>>1);

  long_v[0] = cp->id + 1;
  comm_allreduce(cp, gs_long, gs_add, long_v, 1, long_glb);
  fail_unless(long_glb[0] == (size * (size + 1))>>1);

  float_v[0] = (float) cp->id + 1.0;
  comm_allreduce(cp, gs_float, gs_add, float_v, 1, float_glb);
  fail_unless(float_glb[0] == (float) ((size * (size + 1))>>1));

  double_v[0] = (double) cp->id + 1.0;
  comm_allreduce(cp, gs_double, gs_add, double_v, 1, double_glb);
  fail_unless(double_glb[0] == (double) ((size * (size + 1))>>1));

  int_v[0] = cp->id + 1;
  int_v[1] = cp->id + 1;
  memset(int_glb, 0, 2 * sizeof(int));
  comm_allreduce(cp, gs_int, gs_add, int_v, 2, int_glb);
  fail_unless(int_glb[0] == (size * (size + 1))>>1);
  fail_unless(int_glb[1] == (size * (size + 1))>>1);

  long_v[0] = cp->id + 1;
  long_v[1] = cp->id + 1;
  memset(long_glb, 0, 2 * sizeof(long));
  comm_allreduce(cp, gs_long, gs_add, long_v, 2, long_glb);
  fail_unless(long_glb[0] == (size * (size + 1))>>1);
  fail_unless(long_glb[1] == (size * (size + 1))>>1);

  float_v[0] = (float) cp->id + 1.0;
  float_v[1] = (float) cp->id + 1.0;
  memset(float_glb, 0, 2 * sizeof(float));
  comm_allreduce(cp, gs_float, gs_add, float_v, 2, float_glb);
  fail_unless(float_glb[0] == (size * (size + 1))>>1);
  fail_unless(float_glb[1] == (size * (size + 1))>>1);


  double_v[0] = (double) cp->id + 1.0;
  double_v[1] = (double) cp->id + 1.0;
  memset(double_glb, 0, 2 * sizeof(double));
  comm_allreduce(cp, gs_double, gs_add, double_v, 2, double_glb);
  fail_unless(double_glb[0] == (size * (size + 1))>>1);
  fail_unless(double_glb[1] == (size * (size + 1))>>1);


  comm_free(&cp);
} END_TEST

START_TEST(test_scan) {
  comm_ptr cp;
  ulong sum[2],r[2],v,check_v;
  int rank, size;

  comm_init();
  comm_world(&cp);
  size = cp->np;
  rank = cp->id;

  v = cp->id + 1;
  comm_scan(sum, cp, gs_slong, gs_add, &v, 1, r);
  fail_unless(sum[0] == (rank * (rank + 1)>>1));
  fail_unless(sum[1] == (size * (size + 1)>>1));

  comm_free(&cp);
} END_TEST


START_TEST(test_dot) {
  comm_ptr cp;
  double v[5], w[5], dot;
  int i, rank, size;

  comm_init();
  comm_world(&cp);
  size = cp->np;
  rank = cp->id;

  for (i = 0; i < 5; i++) {
    v[i] = 1.0;
    w[i] = 2.0;
  }

  dot = comm_dot(cp, v, w, 5);
  fail_unless(dot == ((double) size * 10.0));

  comm_free(&cp);
} END_TEST

Suite *comm_suite() {
  TCase *tc;
  Suite *s;
  
  s = suite_create("comm");

  tc = tcase_create("comm_init");
  tcase_add_test(tc, test_init);
  suite_add_tcase(s, tc);

  tc = tcase_create("comm_alloc");
  tcase_add_test(tc, test_alloc);
  suite_add_tcase(s, tc);

  tc = tcase_create("comm_free");
  tcase_add_test(tc, test_free);
  suite_add_tcase(s, tc);

  tc = tcase_create("comm_reduce");
  tcase_add_test(tc, test_reduce);
  suite_add_tcase(s, tc);

  tc = tcase_create("comm_allreduce");
  tcase_add_test(tc, test_allreduce);
  suite_add_tcase(s, tc);

  tc = tcase_create("comm_scan");
  tcase_add_test(tc, test_scan);
  suite_add_tcase(s, tc);

  tc = tcase_create("comm_dot");
  tcase_add_test(tc, test_dot);
  suite_add_tcase(s, tc);
  
  return s;    
}

int main(void) {
  int number_failed;
  Suite *s = comm_suite();
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
  fprintf(stderr, "*** Check is required for COMM tests ***\n");
  return 0;
}

#endif
