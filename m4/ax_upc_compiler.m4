#
# ----------------------------------------------------------------------------
# "THE BEER-WARE LICENSE" (Revision 42):
# <leifniclas.jansson@riken.jp> wrote this file. As long as you retain
# this notice you can do whatever you want with this stuff. If we meet
# some day, and you think this stuff is worth it, you can buy me a
# beer in return Niclas Jansson
# ----------------------------------------------------------------------------
#

AC_DEFUN([AX_UPC_COMPILER_VENDOR],
	 # Ugly hack to run UPC test's
	 _CC=$CC
	 _CFLAGS=$CFLAGS
	 CC=$UPC
	 CFLAGS=""
	 [AC_CACHE_CHECK([for UPC compiler vendor], 
	 ax_cv_upc_compiler_vendor,
	 [ax_cv_upc_compiler_vendor=unknown

	 for vendor in cray:_CRAYC clang:__clang_upc__ gupc:__GUPC__ bupc:__BERKELEY_UPC__ gccupc:__GCC_UPC__; do
  	   tmp="defined("`echo $vendor | cut -d: -f2 | sed 's/,/) || defined(/g'`")"
    	AC_COMPILE_IFELSE([AC_LANG_PROGRAM(,[
#if !($tmp)
      thisisanerror;
#endif
	])],[ax_cv_]upc[_compiler_vendor=`echo $vendor | cut -d: -f1`; break])
	done
	])

	# Check if the UPC runtime support C-style libraries
	AC_MSG_CHECKING([UPC runtime support C-style libraries])
	enable_library="yes"
	if test "x${ax_cv_upc_compiler_vendor}" = xbupc; then
	   enable_library="no"
	else	
	   AC_COMPILE_IFELSE([AC_LANG_PROGRAM(,[
	   #if !(defined(__BERKELEY_UPC_RUNTIME__))
	     thisisanerror;
	   #endif
	   ])],[enable_library="no"])
	fi
		
	AM_CONDITIONAL([ENABLE_LIBRARY], 
		      [test "x${enable_library}" = xyes])

	if test "x${enable_library}" = xyes; then
	   AC_MSG_RESULT([yes])
	else
	   AC_MSG_RESULT([no])
	fi

	# Check if the UPC compiler support the UPC atomic extensions
	AC_MSG_CHECKING([UPC compiler support atomic])
	have_upc_atomics="no"
	AC_COMPILE_IFELSE([AC_LANG_PROGRAM(,[
	#if !(defined(__UPC_ATOMIC__))
	  thisisanerror;
	#endif
	])],[have_upc_atomics="yes"])
	AC_SUBST(have_upc_atomics)

	if test "x${have_upc_atomics}" = xyes; then
	   AC_MSG_RESULT([yes])
	else
	   AC_MSG_RESULT([no])
	fi

	CC=$_CC
	CFLAGS=$_CFLAGS
])					

