pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = exags.pc
ACLOCAL_AMFLAGS = -I m4 --install

SUBDIRS = src

if ENABLE_LIBRARY
SUBDIRS += tests
endif