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
#include <crystal.h>



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
  struct crystal p;

  comm_init(&c, 0);
  crystal_init(&p, &c);
  fail_unless(p.comm.np == THREADS);
  fail_unless(p.comm.np == c.np);
} END_TEST

START_TEST(test_free) {
  struct comm c;
  struct crystal p;

  comm_init(&c, 0);
  crystal_init(&p, &c);
  crystal_free(&p);

  fail_unless(p.comm.buf_dir == NULL);
  fail_unless(p.comm.buf == NULL);
  fail_unless(p.comm.buf_len == 0);
  fail_unless(p.comm.flgs == NULL);

} END_TEST

START_TEST(test_router) {
  struct comm c;
  struct crystal cr;
  uint i, sum, *data, *end;

  comm_init(&c, 0);
  crystal_init(&cr, &c);
  cr.data.n = (4 + (c.id&1)) * c.np;
  buffer_reserve(&cr.data, cr.data.n * sizeof(uint));
  data = cr.data.ptr;

  for (i = 0; i < c.np; i++, data += 3+  data[2]) {
    data[0] = i;
    data[1] = c.id;
    data[2] = 1;
    data[3] = 2*c.id;
    if(c.id&1) {
      data[2] = 2;
      data[4] = data[3]+1;
    }
  }

  crystal_router(&cr);

  fail_unless(cr.data.n == c.np * 4 + (c.np>>1), "(thrd: %d %d (exp: %d)\n", 
	      MYTHREAD, cr.data.n, c.np * 4 * (c.np/2));
  sum = 0;
  data = cr.data.ptr;
  end = data + cr.data.n;

  for(; data != end; data += 3+data[2]) {
    sum += data[1];
    fail_unless(data[3] == data[1] * 2);
    fail_if(data[1]&1 && (data[2] != 2 || data[4] != data[3] + 1));
  }
  fail_unless(sum == (c.np * (c.np - 1)>>1));

  crystal_free(&cr);

  
} END_TEST

Suite *comm_suite() {
  TCase *tc;
  Suite *s;
  
  s = suite_create("crystal");

  tc = tcase_create("crystal_init");
  tcase_add_test(tc, test_init);
  suite_add_tcase(s, tc);

  tc = tcase_create("crystal_free");
  tcase_add_test(tc, test_free);
  suite_add_tcase(s, tc);

  tc = tcase_create("crystal_router");
  tcase_add_test(tc, test_router);
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
  fprintf(stderr, "*** Check is required for CRYSTAL tests ***\n");
  return 0;
}

#endif
