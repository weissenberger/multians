# MULTIANS - Massively Parallel ANS decoding on GPUs

An implementation of a novel algorithm for ANS (Asymmetric Numeral Systems) decoding on GPUs.

The algorithm is capable of decoding raw (unpartitioned) ANS-encoded datastreams of variable size at extremely high throughput rates.

> The method does not require any vendor-specific features. Although this implementation uses the CUDA toolkit, porting it to related parallel programming frameworks, such as OpenCL, should be straightforward.

State count and alphabet size are configurable. At its current increment, the decoder supports input data encoded using a single table and a radix of `b = 2` (i.e. encoder emits single bits during renormalization), and alphabet sizes of up to `256` symbols. Another implementation supporting multiple tables / multiple states is subject of future work.

The sourcecode also includes a (very basic) single-state tANS encoder for testing, as well as a multicore-based implementation of the method for comparison with the GPU version.

## Requirements

* CUDA-enabled GPU with compute capability 3.0 or higher
* GNU/Linux
* CUDA SDK 9 or higher
* latest proprietary graphics drivers

## Compilation process

### Configuration

Please edit the Makefile:

Set `ARCH` to the compute capability of your GPU, i.e. `ARCH = 35` for compute capability 3.5. If you'd like to compile the decoder for multiple generations of GPUs, please edit `NVCC_FLAGS` accordingly.

### Test program

The test program will generate multiple random datasets (256 symbols) of user-specified size. The symbols are exponentially distributed with increasing rate parameters (λ), yielding different compression ratios for different sets.

For each dataset, the program will:

1. encode the data into a single compressed stream using tANS
2. copy / decode the compressed data on a specified GPU
3. decode the compressed data using a specified number of CPU threads
4. print the time elapsed for each decoding process

#### Compiling the test program

To compile the test program, configure the Makefile as described above. Run:

`make`

#### Running the test program

`./bin/demo <compute device index> <size of input in megabytes> <number of CPU threads>`
