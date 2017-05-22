check_PROGRAMS = gs_test1 gs_test2 gs_test4 gs_test8 gs_test16
gs_test1_SOURCES = gs_test.upc
gs_test2_SOURCES = gs_test.upc
gs_test4_SOURCES = gs_test.upc
gs_test8_SOURCES = gs_test.upc
gs_test16_SOURCES = gs_test.upc
gs_test1_UPCFLAGS = -I@top_srcdir@/src/ @CHECK_CFLAGS@ -fupc-threads-1
gs_test2_UPCFLAGS = -I@top_srcdir@/src/ @CHECK_CFLAGS@ -fupc-threads-2
gs_test4_UPCFLAGS = -I@top_srcdir@/src/ @CHECK_CFLAGS@ -fupc-threads-4
gs_test8_UPCFLAGS = -I@top_srcdir@/src/ @CHECK_CFLAGS@ -fupc-threads-8
gs_test16_UPCFLAGS = -I@top_srcdir@/src/ @CHECK_CFLAGS@ -fupc-threads-16
LDADD = $(top_builddir)/src/libexags.a @CHECK_LIBS@

