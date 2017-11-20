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
#include <gs_defs.h>
#include <gs.h>


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

START_TEST(test_setup_pairwise) {
  comm_ptr cp;
  struct gs_data *gsh;

  comm_init();
  comm_world(&cp);

  const uint np = cp->np;
  slong *id = tmalloc(slong,np+4);
  double *v = tmalloc(double,np+4);
  uint i;
  id[0] = -(slong)(np+10+3*cp->id);
  for(i=0;i<np;++i) id[i+1] = -(sint)(i+1);
  id[np+1] = cp->id+1;
  id[np+2] = cp->id+1;
  id[np+3] = np-cp->id;

  gsh = gs_setup(id,np+4,cp,0,gs_pairwise,1);
  free(id);
  gs_free(gsh);

} END_TEST

START_TEST(test_setup_crystal) {
  comm_ptr cp;
  struct gs_data *gsh;

  comm_init();
  comm_world(&cp);

  const uint np = cp->np;
  slong *id = tmalloc(slong,np+4);
  double *v = tmalloc(double,np+4);
  uint i;
  id[0] = -(slong)(np+10+3*cp->id);
  for(i=0;i<np;++i) id[i+1] = -(sint)(i+1);
  id[np+1] = cp->id+1;
  id[np+2] = cp->id+1;
  id[np+3] = np-cp->id;

  gsh = gs_setup(id,np+4,cp,0,gs_crystal_router,1);
  free(id);
  gs_free(gsh);

} END_TEST


START_TEST(test_setup_all_reduce) {
  comm_ptr cp;
  struct gs_data *gsh;

  comm_init();
  comm_world(&cp);

  const uint np = cp->np;
  slong *id = tmalloc(slong,np+4);
  double *v = tmalloc(double,np+4);
  uint i;
  id[0] = -(slong)(np+10+3*cp->id);
  for(i=0;i<np;++i) id[i+1] = -(sint)(i+1);
  id[np+1] = cp->id+1;
  id[np+2] = cp->id+1;
  id[np+3] = np-cp->id;

  gsh = gs_setup(id,np+4,cp,0,gs_all_reduce,1);
  free(id);
  gs_free(gsh);

} END_TEST

START_TEST(test_setup_auto) {
  comm_ptr cp;
  struct gs_data *gsh;

  comm_init();
  comm_world(&cp);

  const uint np = cp->np;
  slong *id = tmalloc(slong,np+4);
  double *v = tmalloc(double,np+4);
  uint i;
  id[0] = -(slong)(np+10+3*cp->id);
  for(i=0;i<np;++i) id[i+1] = -(sint)(i+1);
  id[np+1] = cp->id+1;
  id[np+2] = cp->id+1;
  id[np+3] = np-cp->id;

  gsh = gs_setup(id,np+4,cp,0,gs_auto,1);
  free(id);
  gs_free(gsh);

} END_TEST

START_TEST(test_gs_pairwise) {
  comm_ptr cp;
  struct gs_data *gsh;

  comm_init();
  comm_world(&cp);

  const uint np = cp->np;
  slong *id = tmalloc(slong,np+4);
  double *v = tmalloc(double,np+4);
  uint i;
  id[0] = -(slong)(np+10+3*cp->id);
  for(i=0;i<np;++i) id[i+1] = -(sint)(i+1);
  id[np+1] = cp->id+1;
  id[np+2] = cp->id+1;
  id[np+3] = np-cp->id;

  gsh = gs_setup(id,np+4,cp,0,gs_pairwise,1);
  free(id);
  
  for(i=0;i<np+4;++i) v[i] = 1;
  gs(v,gs_double,gs_add,0,gsh,0);
  fail_unless(v[np+3] == 3);

  for(i=0;i<np+4;++i) v[i] = 1;
  gs(v,gs_double,gs_add,1,gsh,0);
  fail_unless(v[np+3] == np + 3);

  gs_free(gsh);
} END_TEST

START_TEST(test_gs_crystal_router) {
  comm_ptr cp;
  struct gs_data *gsh;

  comm_init();
  comm_world(&cp);

  const uint np = cp->np;
  slong *id = tmalloc(slong,np+4);
  double *v = tmalloc(double,np+4);
  uint i;
  id[0] = -(slong)(np+10+3*cp->id);
  for(i=0;i<np;++i) id[i+1] = -(sint)(i+1);
  id[np+1] = cp->id+1;
  id[np+2] = cp->id+1;
  id[np+3] = np-cp->id;

  gsh = gs_setup(id,np+4,cp,0,gs_crystal_router,1);
  free(id);
  
  for(i=0;i<np+4;++i) v[i] = 1;
  gs(v,gs_double,gs_add,0,gsh,0);
  fail_unless(v[np+3] == 3);

  for(i=0;i<np+4;++i) v[i] = 1;
  gs(v,gs_double,gs_add,1,gsh,0);
  fail_unless(v[np+3] == np + 3);

  gs_free(gsh);
} END_TEST

START_TEST(test_gs_all_reduce) {
  comm_ptr cp;
  struct gs_data *gsh;

  comm_init();
  comm_world(&cp);

  const uint np = cp->np;
  slong *id = tmalloc(slong,np+4);
  double *v = tmalloc(double,np+4);
  uint i;
  id[0] = -(slong)(np+10+3*cp->id);
  for(i=0;i<np;++i) id[i+1] = -(sint)(i+1);
  id[np+1] = cp->id+1;
  id[np+2] = cp->id+1;
  id[np+3] = np-cp->id;

  gsh = gs_setup(id,np+4,cp,0,gs_all_reduce,1);
  free(id);
  
  for(i=0;i<np+4;++i) v[i] = 1;
  gs(v,gs_double,gs_add,0,gsh,0);
  fail_unless(v[np+3] == 3);

  for(i=0;i<np+4;++i) v[i] = 1;
  gs(v,gs_double,gs_add,1,gsh,0);
  fail_unless(v[np+3] == np + 3);

  gs_free(gsh);
} END_TEST

START_TEST(test_gs_auto) {
  comm_ptr cp;
  struct gs_data *gsh;

  comm_init();
  comm_world(&cp);

  const uint np = cp->np;
  slong *id = tmalloc(slong,np+4);
  double *v = tmalloc(double,np+4);
  uint i;
  id[0] = -(slong)(np+10+3*cp->id);
  for(i=0;i<np;++i) id[i+1] = -(sint)(i+1);
  id[np+1] = cp->id+1;
  id[np+2] = cp->id+1;
  id[np+3] = np-cp->id;

  gsh = gs_setup(id,np+4,cp,0,gs_auto,1);
  free(id);
  
  for(i=0;i<np+4;++i) v[i] = 1;
  gs(v,gs_double,gs_add,0,gsh,0);
  fail_unless(v[np+3] == 3);

  for(i=0;i<np+4;++i) v[i] = 1;
  gs(v,gs_double,gs_add,1,gsh,0);
  fail_unless(v[np+3] == np + 3);

  gs_free(gsh);
} END_TEST

Suite *comm_suite() {
  TCase *tc;
  Suite *s;
  
  s = suite_create("gs");

  tc = tcase_create("gs_setup_pairwise");
  tcase_add_test(tc, test_setup_pairwise);
  suite_add_tcase(s, tc);

  tc = tcase_create("gs_setup_crystal");
  tcase_add_test(tc, test_setup_crystal);
  suite_add_tcase(s, tc);

  tc = tcase_create("gs_setup_all_reduce");
  tcase_add_test(tc, test_setup_all_reduce);
  suite_add_tcase(s, tc);

  tc = tcase_create("gs_setup_auto");
  tcase_add_test(tc, test_setup_auto);
  suite_add_tcase(s, tc);

  tc = tcase_create("gs_pairwise");
  tcase_add_test(tc, test_gs_pairwise);
  suite_add_tcase(s, tc);

  tc = tcase_create("gs_crystal_router");
  tcase_add_test(tc, test_gs_crystal_router);
  suite_add_tcase(s, tc);

  tc = tcase_create("gs_all_reduce");
  tcase_add_test(tc, test_gs_all_reduce);
  suite_add_tcase(s, tc);

  tc = tcase_create("gs_auto");
  tcase_add_test(tc, test_gs_auto);
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
  fprintf(stderr, "*** Check is required for GS tests ***\n");
  return 0;
}

#endif
