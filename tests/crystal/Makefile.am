check_PROGRAMS = crystal_test1 crystal_test2 crystal_test4 crystal_test8 crystal_test16
crystal_test1_SOURCES = crystal_test.upc
crystal_test2_SOURCES = crystal_test.upc
crystal_test4_SOURCES = crystal_test.upc
crystal_test8_SOURCES = crystal_test.upc
crystal_test16_SOURCES = crystal_test.upc
crystal_test1_UPCFLAGS = -I@top_srcdir@/src/ @CHECK_CFLAGS@ -fupc-threads-1
crystal_test2_UPCFLAGS = -I@top_srcdir@/src/ @CHECK_CFLAGS@ -fupc-threads-2
crystal_test4_UPCFLAGS = -I@top_srcdir@/src/ @CHECK_CFLAGS@ -fupc-threads-4
crystal_test8_UPCFLAGS = -I@top_srcdir@/src/ @CHECK_CFLAGS@ -fupc-threads-8
crystal_test16_UPCFLAGS = -I@top_srcdir@/src/ @CHECK_CFLAGS@ -fupc-threads-16
LDADD = $(top_builddir)/src/libexags.a @CHECK_LIBS@

