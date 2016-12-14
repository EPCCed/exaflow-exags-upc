# exaflow-exags-upc
UPC version of the ExaGS library from the EU funded ExaFLOW Project

To compile run the following commands.

export UPC=cc
./configure UPCFLAGS="-DPREFIX=jl_ -DNO_NEK_EXITT -DUSE_NAIVE_BLAS -DUNDERSCORE"
make
make install
