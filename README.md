# Introduction

**FaultNet: **
Seismic data fault detection has recently been regarded as a 3D image segmentation task. The nature of fault structures in seismic image makes it difficult to manually label faults. Manual labeling often has many false negative labels (abnormal labels), which will seriously harm the training process. In this work, we find that region-based loss significantly outperforms distribution-based loss when dealing with falsenegative labels, therefore we propose Mask Dice loss (MD loss), which is the first reported region-based loss function for training 3D image segmentation models using sparse 2D slice labels. In addition, fault is an edge feature, and the current network widely used for fault segmentation downsamples the features multiple times, which is not conducive to edge characterization and thus requires many parameters and computational effort to preserve the features. We propose Fault-Net, which always maintains the high-resolution features of seismic images, and the inference process preserves the edge information of faults and performs effective feature fusion to achieve high-quality fault segmentation with only a few parameters and computational effort. Experimental results show that MD loss can clearly weaken the effect of anomalous labels. The Fault-Net parameter is only 0.42MB, support up to 528^3(1.5×10^8, Float32) size cuboid inference on 16GB video ram, and its inference speed on CPU and GPU is significantly faster than other networks, but the result of our method is the state-of-the-art in the FORCE fault identification competition.

# Quick start

    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    pip install segyio,opencv_python

    cd FaultNet
    python prediction.py --input data/kerry.npy

# Results
<div align=center><img src="https://github.com/douyimin/FaultNet/blob/main/results/New_Zealand.png" width="760" height="580" alt="New_Zealand"/><br/>
<img src="https://github.com/douyimin/FaultNet/blob/main/results/FORCE_ML.png" width="760" height="1123" alt="FORCE fault identification competition]"/><br/></div>
