# Introduction

**FaultNet:**
Seismic data fault detection has recently been regarded as a 3D image segmentation task. The nature of fault structures in seismic image makes it difficult to manually label faults. Manual labeling often has many false negative labels (abnormal annotations), which will seriously harm the training process. In this work, we find that region-based loss significantly outperforms distribution-based loss when dealing with false negative labels, therefore we proposed Mask Dice loss (MD loss), which is the first reported region-based loss function for training 3D image segmentation networks using sparse 2D slice labels. In addition, fault is an edge feature, and the current network widely used for fault segmentation downsamples the features multiple times, which is not conducive to edge representation and thus requires many parameters and computational effort to preserve the features. We proposed Fault-Net, which uses a high-resolution and shallow structure to propagate multiple-scale features in parallel, fully preserving edge features. Meanwhile, in order to efficiently fuse multi-scale features, we decouple the convolution process into feature selection and channel fusion, and proposed a lightweight feature fusion block, Multi-Scale Compression Fusion (MCF). Because the Fault-Net always keeps the edge features during propagation, only few parameters and computation are required. Experimental results show that MD loss can clearly weaken the effect of anomalous labels. The Fault-Net parameter is only 0.42MB, support up to 528^3(Float32) 640^3(Float16) size cuboid inference on 16GB video ram, its inference speed on CPU and GPU is significantly faster than other networks. It works well on most of the open data seismic images, and the result of our method is the state-of-the-art in the FORCE fault identification competition.

# Quick start
Get test data (Kerry3D): https://drive.google.com/file/d/1FXQ7mF-1noUm3pWBPTFbXaWkHpmA2ZnY/view?usp=sharing
    
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    pip install segyio,opencv_python
    cp ./download/kerry.npy ./FaultNet/data/
    cd FaultNet
    python prediction.py --input data/kerry.npy --infer_size 272 608 256

# Results
<div align=center><img src="https://github.com/douyimin/FaultNet/blob/main/results/New_Zealand.png" width="760" height="580" alt="New_Zealand"/><br/>
<img src="https://github.com/douyimin/FaultNet/blob/main/results/FORCE_ML.png" width="760" height="1123" alt="FORCE fault identification competition]"/><br/></div>

# Cite us
   
     Dou, Yimin, et al. 
     "Efficient Training of 3D Seismic Image Fault Segmentation Network under Sparse Labels by Weakening Anomaly Annotation."
     arXiv preprint arXiv:2110.05319 (2021).

# Contact us
emindou3015@gmail.com
