#include <stddef.h> 
#include <stdlib.h> 
#include <stdio.h>
#include <name.h>
#include <fail.h>
#include <types.h>
#include <gs_defs.h>
#include <comm.h>


#ifdef HAVE_MPI
#include <mpi.h>
#endif


#ifdef HAVE_CHECK

#include <check.h>

void setup() {
  int rank;
#ifdef HAVE_MPI
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#elif __UPC__
  rank = MYTHREAD;
#endif
  //  if (rank > 0)
    //    freopen ("/dev/null", "w", stdout);
}

void teardown() {
#ifdef HAVE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();  
#endif
}

START_TEST(test_init) {
  struct comm c;
  comm_ext ce;
  int np;

#ifdef HAVE_MPI
  ce = MPI_COMM_WORLD;
#else
  ce = 0;
#endif 

  comm_init(&c, ce);

#ifdef HAVE_MPI
  MPI_Comm_size(ce, &np);
  fail_unless(c.np == np);  
#elif defined(__UPC__)
  fail_unless(c.np == THREADS);
  fail_unless(c.id == MYTHREAD);
  fail_unless(c.buf_dir == NULL);
  fail_unless(c.buf == NULL);
  fail_unless(c.buf_len == 0);
  fail_unless(c.flgs == NULL);
#endif

} END_TEST

#ifdef __UPC__
START_TEST(test_alloc) {
  struct comm c;
  comm_ext ce;
  size_t n;
  int i;

#ifdef HAVE_MPI
  ce = MPI_COMM_WORLD;
#else
  ce = 0;
#endif 

  comm_init(&c, ce);

  n = 42 * sizeof(char);  
  comm_alloc(&c, n);
  fail_unless(c.buf_dir != NULL);
  fail_unless(c.buf_dir[MYTHREAD] != NULL);
  fail_unless(c.buf != NULL);
  fail_unless(c.buf_len == n);

  memset(c.buf, 1, n);


  n = 1147 * sizeof(char);
  comm_alloc(&c, n);
  fail_unless(c.buf_dir != NULL);
  fail_unless(c.buf_dir[MYTHREAD] != NULL);
  fail_unless(c.buf != NULL);
  fail_unless(c.buf_len == n);
  
  for (i = 0; i < 42; i++)
    fail_unless(c.buf[i] == 1);

  n = 10 * sizeof(char);
  comm_alloc(&c, n);
  fail_unless(c.buf_dir != NULL);
  fail_unless(c.buf_dir[MYTHREAD] != NULL);
  fail_unless(c.buf != NULL);
  fail_unless(c.buf_len != n);

  for (i = 0; i < 42; i++)
    fail_unless(c.buf[i] == 1);

} END_TEST
#endif

START_TEST(test_free) {
  struct comm c;
  comm_ext ce;

#ifdef HAVE_MPI
  ce = MPI_COMM_WORLD;
#else
  ce = 0;
#endif 

  comm_init(&c, ce);

  comm_free(&c);  
#ifdef HAVE_MPI
  fail_unless(c.c == MPI_COMM_NULL);
#elif defined(__UPC__)
  fail_unless(c.buf_dir == NULL);
  fail_unless(c.buf == NULL);
  fail_unless(c.buf_len == 0);
  fail_unless(c.flgs == NULL);
#endif


  comm_free(&c);  
#ifdef HAVE_MPI
  fail_unless(c.c == MPI_COMM_NULL);
#elif defined(__UPC__)
  fail_unless(c.buf_dir == NULL);
  fail_unless(c.buf == NULL);
  fail_unless(c.buf_len == 0);
  fail_unless(c.flgs == NULL);
#endif



} END_TEST


START_TEST(test_reduce) {
  struct comm c;
  comm_ext ce;
  int int_v, int_glb;
  long long_v, long_glb;
  float float_v, float_glb;
  double double_v, double_glb;

  int np;

#ifdef HAVE_MPI
  ce = MPI_COMM_WORLD;
  MPI_Comm_size(ce, &np);
#elif __UPC__
  ce = 0;
  np = THREADS;
#else
  ce = 0;
  np = 1;
#endif
  
  comm_init(&c, ce);

  int_v = c.id + 1;
  int_glb = comm_reduce_int(&c, gs_add, (int *) &int_v, 1);
  fail_unless(int_glb == (np * (np + 1)>>1));

  long_v = c.id + 1;
  long_glb = comm_reduce_long(&c, gs_add, (long *) &long_v, 1);
  fail_unless(long_glb == (np * (np + 1)>>1));

  float_v = (float) c.id + 1;
  float_glb= comm_reduce_float(&c, gs_add, (float *) &float_v, 1);
  fail_unless(float_glb == (np * (np + 1)>>1));

  double_v= c.id + 1;
  double_glb = comm_reduce_double(&c, gs_add, (double *) &double_v, 1);
  fail_unless(double_glb == (np * (np + 1)>>1));

  comm_free(&c);
} END_TEST

START_TEST(test_allreduce) {
  struct comm c;
  comm_ext ce;
  int int_v[2], int_glb[2];
  long long_v[2], long_glb[2];
  float float_v[2], float_glb[2];
  double double_v[2], double_glb[2];
  int size;

#ifdef HAVE_MPI
  ce = MPI_COMM_WORLD;
  MPI_Comm_size(ce, &size);
#elif __UPC__
  ce = 0;
  size = THREADS;
#else
  ce = 0;
  size = 1;
#endif
  
  comm_init(&c, ce);

  /*
   * Scalar reductions
   */
  int_v[0] = c.id + 1;
  comm_allreduce(&c, gs_int, gs_add, int_v, 1, int_glb);
  fail_unless(int_glb[0] == (size * (size + 1))>>1);

  long_v[0] = c.id + 1;
  comm_allreduce(&c, gs_long, gs_add, long_v, 1, long_glb);
  fail_unless(long_glb[0] == (size * (size + 1))>>1);

  float_v[0] = (float) c.id + 1.0;
  comm_allreduce(&c, gs_float, gs_add, float_v, 1, float_glb);
  fail_unless(float_glb[0] == (float) ((size * (size + 1))>>1));

  double_v[0] = (double) c.id + 1.0;
  comm_allreduce(&c, gs_double, gs_add, double_v, 1, double_glb);
  fail_unless(double_glb[0] == (double) ((size * (size + 1))>>1));

  /*
   * Vector reductions (element-wise)
   */
  int_v[0] = c.id + 1;
  int_v[1] = c.id + 1;
  memset(int_glb, 0, 2 * sizeof(int));
  comm_allreduce(&c, gs_int, gs_add, int_v, 2, int_glb);
  fail_unless(int_glb[0] == (size * (size + 1))>>1);
  fail_unless(int_glb[1] == (size * (size + 1))>>1);

  long_v[0] = c.id + 1;
  long_v[1] = c.id + 1;
  memset(long_glb, 0, 2 * sizeof(long));
  comm_allreduce(&c, gs_long, gs_add, long_v, 2, long_glb);
  fail_unless(long_glb[0] == (size * (size + 1))>>1);
  fail_unless(long_glb[1] == (size * (size + 1))>>1);

  float_v[0] = (float) c.id + 1.0;
  float_v[1] = (float) c.id + 1.0;
  memset(float_glb, 0, 2 * sizeof(float));
  comm_allreduce(&c, gs_float, gs_add, float_v, 2, float_glb);
  fail_unless(float_glb[0] == (size * (size + 1))>>1);
  fail_unless(float_glb[1] == (size * (size + 1))>>1);


  double_v[0] = (double) c.id + 1.0;
  double_v[1] = (double) c.id + 1.0;
  memset(double_glb, 0, 2 * sizeof(double));
  comm_allreduce(&c, gs_double, gs_add, double_v, 2, double_glb);
  fail_unless(double_glb[0] == (size * (size + 1))>>1);
  fail_unless(double_glb[1] == (size * (size + 1))>>1);


  comm_free(&c);
} END_TEST

START_TEST(test_scan) {
  struct comm c;
  comm_ext ce;
  int scan[2],buf[2],v,check_v;
  int rank, size;
#ifdef HAVE_MPI
  ce = MPI_COMM_WORLD;    
  MPI_Comm_rank(ce, &rank);
  MPI_Comm_size(ce, &size);
#elif __UPC__
  ce = 0;
  size = THREADS;
  rank = MYTHREAD;
#else
  ce = 0;
  size = 1;
  rank = 0;
#endif
  
  comm_init(&c, ce);

  v = c.id + 1;
  comm_scan(scan, &c, gs_int, gs_add, &v, 1, buf);
  fail_unless(scan[0] == (rank * (rank + 1)>>1));
  fail_unless(scan[1] == (size * (size + 1)>>1));
  comm_free(&c);
} END_TEST

START_TEST(test_dot) {
  struct comm c;
  comm_ext ce;
  double v[5], w[5], dot;
  int i, rank, size;
#ifdef HAVE_MPI
  ce = MPI_COMM_WORLD;    
  MPI_Comm_rank(ce, &rank);
  MPI_Comm_size(ce, &size);
#elif __UPC__
  ce = 0;
  size = THREADS;
  rank = MYTHREAD;
#else
  ce = 0;
  size = 1;
  rank = 0;
#endif
  
  comm_init(&c, ce);

  for (i = 0; i < 5; i++) {
    v[i] = 1.0;
    w[i] = 2.0;
  }

  dot = comm_dot(&c, v, w, 5);
  fail_unless(dot == ((double) size * 10.0));

  comm_free(&c);
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
