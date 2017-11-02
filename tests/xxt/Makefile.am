check_PROGRAMS = xxt_test1 xxt_test2 xxt_test4 xxt_test8 xxt_test16
xxt_test1_SOURCES = xxt_test.upc
xxt_test1_UPCFLAGS = -I@top_srcdir@/src/ @CHECK_CFLAGS@ -fupc-threads-1
xxt_test2_SOURCES = xxt_test.upc
xxt_test2_UPCFLAGS = -I@top_srcdir@/src/ @CHECK_CFLAGS@ -fupc-threads-2
xxt_test4_SOURCES = xxt_test.upc
xxt_test4_UPCFLAGS = -I@top_srcdir@/src/ @CHECK_CFLAGS@ -fupc-threads-4
xxt_test8_SOURCES = xxt_test.upc
xxt_test8_UPCFLAGS = -I@top_srcdir@/src/ @CHECK_CFLAGS@ -fupc-threads-8
xxt_test16_SOURCES = xxt_test.upc
xxt_test16_UPCFLAGS = -I@top_srcdir@/src/ @CHECK_CFLAGS@ -fupc-threads-16
LDADD = $(top_builddir)/src/libexags.a @CHECK_LIBS@

