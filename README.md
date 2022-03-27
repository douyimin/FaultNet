# Introduction

**FaultNet:**
Data-driven fault detection has been regarded as a 3D image segmentation task. The models trained from synthetic data are difficult to generalize in some surveys. Recently, training 3D fault segmentation using sparse manual 2D slices is thought to yield promising results, but manual labeling has many false negative labels (abnormal annotations), which is detrimental to training and consequently to detection performance. Motivated to train 3D fault segmentation networks under sparse 2D labels while suppressing false negative labels, we analyze the training process gradient and propose the Mask Dice (MD) loss. Moreover, the fault is an edge feature, and current encoderdecoder architectures widely used for fault detection (e.g., Ushape network) are not conducive to edge representation and have redundant parameters. Consequently, Fault-Net is proposed, which is designed for the characteristics of faults, employs high-resolution propagation features, and embeds Multi-Scale Compression Fusion module to fuse multi-scale information, which allows the edge information to be fully preserved during propagation and fusion, thus enabling advanced performance via few computational resources. Experimental demonstrates that MD loss supports the inclusion of human experience in training and suppresses false negative labels therein, enabling baseline models to improve performance and generalize to more surveys. The Fault-Net parameter is only 0.42MB, support up to 528^3 (FP32) and 640^3 (FP16) size cuboid inference on 16GB RAM, its inference speed is significantly faster than other models. Our approach employs fewer computational resources while providing more reliable and clearer interpretations of seismic faults..

# Quick start
Get test data (F3 and Kerry3D): https://drive.google.com/drive/folders/1LEHd2VO9TZTOjrMuAQ7I446OfYDgcdWo?usp=sharing
    
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    pip install segyio,opencv_python
    cp ./download/kerry.npy ./FaultNet/data/
    cd FaultNet
    python prediction.py --input data/F3.npy

# Results
<div align=center><img src="https://github.com/douyimin/FaultNet/blob/main/results/output.png" width="805" height="517" alt="Results"/><br/></div>

# Cite us
   
     Dou, Yimin, et al. 
     "Efficient Training of 3D Seismic Image Fault Segmentation Network under Sparse Labels by Weakening Anomaly Annotation."
     arXiv preprint arXiv:2110.05319 (2021).

# Contact us
emindou3015@gmail.com
