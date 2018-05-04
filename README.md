## Layrub
### Introduction
Layrub is a runtime data placement strategy for extreme-scale training. It achieves layer-centric memory reuse to reduce memory consumption for extreme-scale DNN models that could not previously be run on a single GPU. 
We deploy Layrub on the basis of BVLC Caffe(https://github.com/BVLC/caffe).


### Feature
- Intra-layer strategy: demonstrate the opportunities for memory reuse at the intra-layer level, which works effectively for wide networks, using the memory space of activation data as the space for gradient data.  
  Inter-layer strategy: demonstrate the opportunities for memory reuse at the inter-layer level, which works effectively for deep networks.   
  Hybrid strategy: combines both intra-layer and inter-layer strategy, which work sperfectly with most DNNs.
- Compared to the original Caﬀe, Layrub can cut down the memory usage rate by an average of 58.2% and by up to 98.9%, at the moderate cost of 24.1% higher training execution time on average.
- Layrub on Caﬀe is user-friendly, the changes on source code are transparent to users, the use of C++ interface is not different from caffe.

### Installation
Please refer to the installation steps in BVLC Caffe(http://caffe.berkeleyvision.org/installation.html)

### Reference Paper
> Bo Liu, Wenbin Jiang, Hai Jin, Xuanhua Shi, and Yang Ma. Layrub: Layer-centric GPU memory reuse and data migration in extreme-scale deep learning systems. In Proceedings of ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming, Vienna, Austria, February 24–28, 2018 (PPoPP'18).
