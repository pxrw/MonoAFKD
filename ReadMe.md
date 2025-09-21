## MonoAFKD: Align and Frequency cross Knowledge Distillation for Monocular 3D Object Detection

![](/readme/framework.jpg)

---

This is the repo of MonoAFKD: Align and Frequency cross Knowledge Distillation for Monocular 3D Object Detection.

## 🍒Abstract

Monocular 3D object detection holds great promise but remains fundamentally challenging due to its ill-posed nature stemming from the absence of reliable depth information, which is critical for accurate spatial reasoning. In contrast, LiDAR data provides precise depth cues, making it an effective tool for improving monocular detection through cross-modal knowledge distillation. This approach transfers depth-related knowledge from a LiDAR-trained teacher model to an RGB-trained student model. However, the inherent feature heterogeneity between the teacher and student models often introduces noise, leading to negative transfer. Additionally, differences in their focus on high- and low-frequency features hinder knowledge transfer. To overcome these issues, we introduce MonoAFKD, a Spatial feature-aligned, multi-Frequency cross-modal Knowledge Distillation method. MonoAFKD integrates two core modules: an Attention-based heterogeneous Feature Alignment (AFA) module and a Wavelet-based multi-Frequency Distillation (WFD) module. The AFA module spatially aligns features with heterogeneous characteristics, while the WFD module enhances knowledge transfer by capturing information in the frequency domain. Together, these modules enable the effective transfer of depth cues, significantly improving distillation performance. Extensive experiments on the KITTI 3D detection benchmark and NuScenes dataset demonstrate the efficacy of MonoAFKD. Our method achieves state-of-the-art results without introducing additional inference costs, validating its practicality and efficiency.

---

## 🍓 Getting Start

### Dataset Preparation

*   Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows:

    ~~~
    this repo
    ├── data
    │   │── KITTI3D
    |   │   │── training
    |   │   │   ├──calib & label_2 & image_2 & depth_dense
    |   │   │── testing
    |   │   │   ├──calib & image_2
    ├── config
    ├── ...
    ~~~

*   You can also choose to link your KITTI dataset path by

    ~~~
     KITTI_DATA_PATH=~/data/kitti_object
     ln -s $KITTI_DATA_PATH ./data/KITTI3D
    ~~~

*   To ease the usage,  the pre-generated dense depth files at: [Google Drive](https://drive.google.com/file/d/1mlHtG8ZXLfjm0lSpUOXHulGF9fsthRtM/view?usp=sharing) 

---

### 🍇Training & Testing

## Training

~~~
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_val.py --config configs/monoafkd.yaml
The pretrained teacher and student model can get from here[通过网盘分享的文件：pretrained
链接: https://pan.baidu.com/s/1ZR0PbZPF1B5SZBfoQWYgkA 提取码: s7v6 ]
~~~

#### Test and evaluate

~~~
CUDA_VISIBLE_DEVICES=0 python tools/train_val.py --config configs/monoafkd.yaml -e
~~~

---

## Acknowledgements

This respository is mainly based on [**DID-M3d**](https://github.com/SPengLiang/DID-M3D)

