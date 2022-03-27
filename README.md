# Introduction

**FaultNet:**
Data-driven fault detection has been regarded as a 3D image segmentation task. The models trained from synthetic data are difficult to generalize in some surveys. Recently, training 3D fault segmentation using sparse manual 2D slices is thought to yield promising results, but manual labeling has many false negative labels (abnormal annotations), which is detrimental to training and consequently to detection performance. Motivated to train 3D fault segmentation networks under sparse 2D labels while suppressing false negative labels, we analyze the training process gradient and propose the Mask Dice (MD) loss. Moreover, the fault is an edge feature, and current encoder-decoder architectures widely used for fault detection (e.g., U-shape network) are not conducive to edge representation. Consequently, Fault-Net is proposed, which is designed for the characteristics of faults, employs high-resolution propagation features, and embeds MultiScale Compression Fusion block to fuse multi-scale information, which allows the edge information to be fully preserved during propagation and fusion, thus enabling advanced performance via few computational resources. Experimental demonstrates that MD loss supports the inclusion of human experience in training and suppresses false negative labels therein, enabling baseline models to improve performance and generalize to more surveys. Fault-Net is capable to provide a more stable and reliable interpretation of faults, it uses extremely low computational resources and inference is significantly faster than other models. Our method indicates optimal performance in comparison with several mainstream methods.

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
