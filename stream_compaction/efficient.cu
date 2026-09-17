#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "common.h"
#include "efficient.h"

constexpr int BLOCK_SIZE = 256;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        // assumed to be called on power of 2 (padding done first)
        __global__ void kernUpIter(int n, int d, int* odata, const int* idata) {
            int stride = 1 << d;
            int skip = stride << 1; // e.g., layer one every 2 combine 1
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                if ((idx + 1) % skip == 0) {  // + 1 for 0 indexing
                    odata[idx] = idata[idx] + idata[idx - stride]; // consider performance improvement
                }
                else {
                    odata[idx] = idata[idx]; // copy if unchanged
                }
            }
            return;
        }

        // assumed to be called on power of 2 (padding done first)
        __global__ void kernUpIter_improved(int n, int d, int* odata, const int* idata) {
            int stride = 1 << d;
            int skip = stride << 1; // e.g., layer one every 2 combine 1
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < n / skip) {
                int idx_2 = (idx + 1) * skip - 1;
                odata[idx_2] = idata[idx_2] + idata[idx_2 - stride];
            }

            return;
        }

        // assumes you replaced rightmost with 0.
        __global__ void kernDownIter(int n, int d, int* odata, const int* idata) {
            int stride = n >> (d + 1); // start with d=0
            int dbl_stride = stride << 1;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                if ((idx + 1) % stride == 0) {
                    if ((idx + 1) % dbl_stride == 0) {
                        odata[idx] = idata[idx] + idata[idx - stride];
                    }
                    else {
                        odata[idx] = idata[idx + stride];
                    }
                }
                else {
                    odata[idx] = idata[idx]; // unchanged in this iter
                }
            }
            return;
        }

        // assumes you replaced rightmost with 0.
        __global__ void kernDownIter_improved(int n, int d, int* odata, const int* idata) {
            int stride = n >> (d + 1); // start with d=0
            int dbl_stride = stride << 1;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < n / dbl_stride) {
                int idx_2 = (idx + 1) * dbl_stride - 1; //right child
                int t = idata[idx_2 - stride]; //left child
                odata[idx_2 - stride] = idata[idx_2];
                odata[idx_2] = idata[idx_2] + t;
            }


            return;
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int iters = ilog2ceil(n);
            int padded_size = 1 << iters;
            //int num_blocks = (padded_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

            std::vector<int> padded(padded_size, 0);
            std::copy(idata, idata + n, padded.begin());
            int* dev_in = nullptr;
            int* dev_out = nullptr;
            cudaMalloc((void**)&dev_in, padded_size * sizeof(int));
            cudaMalloc((void**)&dev_out, padded_size * sizeof(int));
            cudaMemcpy(dev_in, padded.data(), padded_size * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            // upsweep
            for (int d = 0; d < iters; ++d) {
                int num_blocks = ((padded_size >> (d + 1)) + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernUpIter_improved << <num_blocks, BLOCK_SIZE >> > (padded_size, d, dev_in, dev_in); // dont need both
            }

            // is this the best way to copy in the 0?
            int zero = 0;
            cudaMemcpy(dev_in + padded_size - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);

            // downsweep
            for (int d = 0; d < iters; ++d) {
                int num_blocks = ((1 << d) + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernDownIter_improved << <num_blocks, BLOCK_SIZE >> > (padded_size, d, dev_in, dev_in);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_in);
            cudaFree(dev_out);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // copied code from scan
            int iters = ilog2ceil(n);
            int padded_size = 1 << iters;
            int num_blocks = (padded_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

            std::vector<int> padded(padded_size, 0);
            std::copy(idata, idata + n, padded.begin());
            int* dev_in = nullptr;
            int* dev_out = nullptr;
            int* dev_bool = nullptr;
            int* dev_idata = nullptr;
            cudaMalloc((void**)&dev_bool, n * sizeof(int));
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_in, padded_size * sizeof(int));
            cudaMalloc((void**)&dev_out, padded_size * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_in, padded.data(), padded_size * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            Common::kernMapToBoolean << <num_blocks, BLOCK_SIZE >> > (n, dev_bool, dev_idata);
            
            // store the raw bools before scanning on them
            cudaMemcpy(dev_in, dev_bool, n * sizeof(int), cudaMemcpyDeviceToDevice);

            // upsweep
            for (int d = 0; d < iters; ++d) {
                kernUpIter << <num_blocks, BLOCK_SIZE >> > (padded_size, d, dev_out, dev_in);
                std::swap(dev_in, dev_out);
            }

            // is this the best way to copy in the 0?
            int zero = 0;
            cudaMemcpy(dev_in + padded_size - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);

            // downsweep
            for (int d = 0; d < iters; ++d) {
                kernDownIter << <num_blocks, BLOCK_SIZE >> > (padded_size, d, dev_out, dev_in);
                std::swap(dev_in, dev_out);
            }

            num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; // reset block size to not include padding

            
            // reuse dev_out but only copy from front of it
            Common::kernScatter << <num_blocks, BLOCK_SIZE >> > (n, dev_out, dev_idata, dev_bool, dev_in);

            timer().endGpuTimer();

            int lastIdx = 0, lastBool = 0;
            cudaMemcpy(&lastIdx, dev_in + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, dev_bool + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIdx + lastBool;

            cudaMemcpy(odata, dev_out, count * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_bool);
            cudaFree(dev_idata);

            return count;
        }
    }
}
