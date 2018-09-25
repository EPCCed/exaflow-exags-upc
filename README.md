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

## Nek5000
To enable _Nek5000_ support in the library, add ```--enable-nek5000``` to the configure line. This will define necessary ```UPCFLAGS``` such that Nek5000 could link against ExaGS.

## Atomics
To enable the use of _UPC atomics_ in the library, add ```--enable-atomics``` to the configure line.

## Testing
To enable compilation of the tests, add ```--enable-check``` to the configure line. This will generate a number of UPC-based test codes (compiled with a fixed number of threads) in the ```tests/``` subtree. To run the entire test suite, issue ```make check``` from the top level directory.