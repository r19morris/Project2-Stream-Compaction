CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Ryan Morris
  * [LinkedIn](www.linkedin.com/in/r19)
* Tested on: Windows 11, Intel i7-12700H @ 2.3GHz 64GB, GeForce RTX 3070 Ti Laptop GPU 8GB

### Project Description

This project is a CUDA parallel stream compaction algorithm. Below, we compare the performance of the CPU implementation, naive parallel implementation, and effecient parallel implementation.

### Performance Discussion

1. Block Size Optimization

* For both naive and efficient GPU implementations, I tested block sizes ranging from 128 to 1024. There was a lot of variance in the results, but for the Naive implementation a block size of 512 consistently worked the best on our tests and for the efficient implementation slightly smaller block sizes (in this case 256) worked best on the test.

<table>
  <tr>
    <th rowspan="2">Implementation</th>
    <th rowspan="2">Block Size</th>
    <th colspan="2">Power of Two</th>
    <th colspan="2">Non-Power of Two</th>
  </tr>
  <tr>
    <th>Test 1</th><th>Test 2</th><th>Test 1</th><th>Test 2</th>
  </tr>
  <tr><td>Naive</td><td>128</td><td>0.1189</td><td>0.2488</td><td>0.06554</td><td>0.1403</td></tr>
  <tr><td>Naive</td><td>256</td><td>0.1159</td><td>0.1219</td><td>0.0952</td><td>0.0788</td></tr>
  <tr><td><b>Naive</b></td><td><b>512</b></td><td><b>0.1055</b></td><td><b>0.1065</b></td><td><b>0.0963</b></td><td><b>0.0748</b></td></tr>
  <tr><td>Naive</td><td>1024</td><td>0.1055</td><td>0.1629</td><td>0.0901</td><td>0.08813</td></tr>
  <tr><td>Efficient</td><td>128</td><td>0.1741</td><td>0.1546</td><td>0.1085</td><td>0.1014</td></tr>
  <tr><td><b>Efficient</b></td><td><b>256</b></td><td><b>0.1638</b></td><td><b>0.1577</b></td><td><b>0.1884</b></td><td><b>0.0963</b></td></tr>
  <tr><td>Efficient</td><td>512</td><td>0.1393</td><td>0.1690</td><td>0.1198</td><td>0.1055</td></tr>
  <tr><td>Efficient</td><td>1024</td><td>0.2030</td><td>0.1649</td><td>0.1106</td><td>0.1024</td></tr>
</table>

2. Comparison of GPU Scan implementations (initial implementation)

1. 

3. Program Output (initial implementation)'


### Extra Credit



Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

