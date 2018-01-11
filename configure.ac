AC_INIT([ExaGS],[0.0.1])
AM_INIT_AUTOMAKE([foreign])
AM_MAINTAINER_MODE
AC_CONFIG_MACRO_DIR([m4])
AC_LANG(C)

AC_ARG_ENABLE(mpi, AC_HELP_STRING([--enable-mpi],
              [Compile with support for MPI]),
              [enable_mpi=${enableval}], [enable_mpi=no])

AC_ARG_ENABLE(check, AC_HELP_STRING([--enable-check], 
              [Compile with support for unit testing, requires libcheck]),
              [enable_check=${enableval}],[enable_check=no])

AC_ARG_ENABLE(underscore, AC_HELP_STRING([--enable-underscore],
	     [Underscore name mangling for the Fortran interface]),
	     [enable_underscore=${enableval}],[enable_underscore=no])

AC_ARG_ENABLE(naive-blas, AC_HELP_STRING([--enable-naive-blas],
	     [Compile with naive blas]),
	     [enable_naive_blas=${enableval}],[enable_naive_blas=yes])

AC_ARG_ENABLE(nek5000, AC_HELP_STRING([--enable-nek5000],
	     [Compile with support for Nek5000]),
	     [enable_nek5000=${enableval}],[enable_nek5000=no])

AC_ARG_ENABLE(atomics, AC_HELP_STRING([--enable-atomics],
             [Use UPC atomics for synchronization]),
	     [enable_atomics=${enableval}],[enable_atomics=no])

AC_PROG_CC
AC_PROG_RANLIB
AC_PROG_INSTALL
AM_PROG_UPC
AX_UPC_COMPILER_VENDOR

# Test for unit testing framework
if test "x${enable_check}" = xyes; then
   PKG_CHECK_EXISTS([check],[have_check=yes],[have_check=no])
   if test "x${have_check}" = xyes; then
      PKG_CHECK_MODULES([CHECK], [check >= 0.9.4])
      AC_DEFINE(HAVE_CHECK,[1],[Have CHECK library])
   else
      AC_MSG_ERROR([CHECK not found, needed for unit testing])
   fi
fi

CPPFLAGS="-DAMG_BLOCK_ROWS=2400 $CPPFLAGS"

if test "x${enable_underscore}" = xyes; then
   CPPFLAGS="-DUNDERSCORE $CPPFLAGS"
fi

if test "x${enable_naive_blas}" = xyes; then
   CPPFLAGS="-DUSE_NAIVE_BLAS $CPPFLAGS"
fi

if test "x${enable_nek5000}" = xyes; then
   CPPFLAGS="-DPREFIX=gslib_ -DFPREFIX=fgslib_ -DGLOBAL_LONG_LONG -DUSR_USR_EXIT $CPPFLAGS"
else
   CPPFLAGS="-DPREFIX=exags_ $CPPFLAGS"
fi

if test "x${enable_atomics}" = xyes; then
   if test "x${have_upc_atomics}" = xyes; then
      CPPFLAGS="-DUSE_ATOMIC $CPPFLAGS"
   else
      AC_MSG_ERROR([UPC compiler does not support the atomic extension])
   fi
fi



AC_CONFIG_FILES([exags.pc\
		 Makefile\
		 src/Makefile\
		 tests/Makefile\
		 tests/comm/Makefile\
		 tests/crystal/Makefile\
		 tests/gs/Makefile\
		 tests/xxt/Makefile])
AC_OUTPUT

# Display help text
echo \
"
---------------------------------------------------------

Configuration of ExaGS ${PACKAGE_VERSION} finished.

Prefix: '${prefix}'.
Compiler: ${UPC} ${UPCFLAGS} ${CPPFLAGS}


---------------------------------------------------------
"
