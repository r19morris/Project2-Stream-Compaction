#include <cstdio>
#include <vector>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // Simple for loop version
            odata[0] = 0;
            for (int i = 1; i < n; ++i) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i]) {
                    odata[count++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // scan first into odata?
            scan(n, odata, idata);
            std::vector<int> temp(n);
            for (int i = 0; i < n; ++i) {
                if (idata[i]) {
                    temp[i] = 1;
                }
                else {
                    temp[i] = 0;
                }
            }
            std::vector<int> scan_output(n);
            scan(n, scan_output.data(), temp.data());
            for (int i = 0; i < n; ++i) {
                if (temp[i]) {
                    odata[scan_output[i]] = idata[i];
                }
            }



            timer().endCpuTimer();
            return scan_output[n] + 1;
        }
    }
}
