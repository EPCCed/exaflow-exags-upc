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


#ifdef HAVE_CHECK

#include <check.h>

void setup() {
}

void teardown() {
}

START_TEST(test_init) {
  comm_ptr cp;
  struct crystal p;

  comm_init();
  comm_world(&cp);
  crystal_init(&p, cp);
  fail_unless(p.comm->np == THREADS);
  fail_unless(p.comm->np == cp->np);
} END_TEST

START_TEST(test_free) {
  comm_ptr cp;
  struct crystal p;

  comm_init();
  comm_world(&cp);
  crystal_init(&p, cp);
  crystal_free(&p);

  fail_unless(p.comm == NULL);
  fail_unless(p.size == NULL);

} END_TEST

START_TEST(test_router) {
  comm_ptr cp;
  struct crystal cr;
  uint i, sum, *data, *end;

  comm_init();
  comm_world(&cp);
  crystal_init(&cr, cp);

  cr.data.n = (4 + (cp->id&1)) * cp->np;
  buffer_reserve(&cr.data, cr.data.n * sizeof(uint));
  data = cr.data.ptr;

  for (i = 0; i < cp->np; i++, data += 3+  data[2]) {
    data[0] = i;
    data[1] = cp->id;
    data[2] = 1;
    data[3] = 2*cp->id;
    if(cp->id&1) {
      data[2] = 2;
      data[4] = data[3]+1;
    }
  }

  crystal_router(&cr);

  fail_unless(cr.data.n == cp->np * 4 + (cp->np>>1),
	      "(thrd: %d %d (exp: %d)\n", 
	      MYTHREAD, cr.data.n, cp->np * 4 * (cp->np/2));
  sum = 0;
  data = cr.data.ptr;
  end = data + cr.data.n;

  for(; data != end; data += 3+data[2]) {
    sum += data[1];
    fail_unless(data[3] == data[1] * 2);
    fail_if(data[1]&1 && (data[2] != 2 || data[4] != data[3] + 1));
  }
  fail_unless(sum == (cp->np * (cp->np - 1)>>1));

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
