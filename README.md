# exaflow-exags-upc
This is a UPC version of the ExaGS library from the EU funded ExaFLOW Project.

## Library Compilation
To compile run the following commands, this will generate the static library ```libexags.a```. Change the first line to point to your UPC compiler of choice.

```C
export UPC=cc
./configure UPCFLAGS="-DPREFIX=jl_ -DNO_NEK_EXITT -DUSE_NAIVE_BLAS -DUNDERSCORE"
make
make install
```

## MPI
To enable _MPI_ in the library, add ```--enable-mpi``` to the configure line.

## Testing
To enable compilation of the tests, add ```--enable-check``` to the configure line. This will generate a number of UPC-based test codes (compiled with a fixed number of threads) in the ```tests/``` subtree.