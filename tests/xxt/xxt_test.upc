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
