#include <cuda.h>
#include <cuda_runtime.h>
#include <utility>
#include "common.h"
#include "naive.h"

constexpr int BLOCK_SIZE = 512;

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        __global__ void kernNaiveIter(int n, int d, int* odata, const int* idata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int shift = 1 << d; // start on d = 0 (to calculate the depth 1 buffer)
            if (idx < n) {
                if (idx < shift) {
                    odata[idx] = idata[idx]; // copy if already been reached in lower iteration
                }
                else {
                    odata[idx] = idata[idx] + idata[idx - shift]; // combine
                }
            }
            return;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int iters = ilog2ceil(n);
            int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int* dev_in = nullptr;
            int* dev_out = nullptr;
            cudaMalloc((void**) & dev_in, n * sizeof(int));
            cudaMalloc((void**) & dev_out, n * sizeof(int));
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            for (int d = 0; d < iters; ++d) {
                kernNaiveIter << <num_blocks, BLOCK_SIZE >> > (n, d, dev_out, dev_in);
                std::swap(dev_in, dev_out);
            }
            cudaDeviceSynchronize();
            timer().endGpuTimer();
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_in, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost); // convert to exclusive
            cudaFree(dev_in);
            cudaFree(dev_out);
            return;
        }
    }
}
