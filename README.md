# Awesome-ECCV2020-Low-Level-Vision[![Awesome](https://camo.githubusercontent.com/13c4e50d88df7178ae1882a203ed57b641674f94/68747470733a2f2f63646e2e7261776769742e636f6d2f73696e647265736f726875732f617765736f6d652f643733303566333864323966656437386661383536353265336136336531353464643865383832392f6d656469612f62616467652e737667)](https://github.com/sindresorhus/awesome)

A Collection of Papers and Codes for ECCV2020 Low Level Vision or Image Reconstruction

<font  size=5>整理汇总了下今年ECCV图像重建/底层视觉(Low-Level Vision)相关的一些论文，包括超分辨率，图像恢复，去雨，去雾，去模糊，去噪等方向。大家如果觉得有帮助，欢迎star~~</font>

2020年ECCV（European Conference on Computer Vision）将于8月2日到8月28日在线上召开。目前ECCV2020已经放榜，有效投稿数为5025，最终收录1361篇论文，录取率是27%。其中104篇 Oral、161篇 Spotlights，其余的均为Poster。
- ECCV2020的官网：[https://eccv2020.eu/](https://eccv2020.eu/)
- ECCV2020接收论文列表：[https://eccv2020.eu/accepted-papers/](https://eccv2020.eu/accepted-papers/)

**【Contents】**
- [1.超分辨率（Super-Resolution）](#1.超分辨率)
- [2.图像去雨（Image Deraining）](#2.图像去雨)
- [3.图像去雾（Image Dehazing）](#3.图像去雾)
- [4.去模糊（Deblurring）](#4.去模糊)
- [5.去噪（Denoising）](#5.去噪)
- [6.图像恢复（Image Restoration）](#6.图像恢复)
- [7.图像增强（Image Enhancement）](#7.图像增强)
- [8.图像去摩尔纹（Image Demoireing）](#8.图像去摩尔纹)
- [9.图像修复（Inpainting）](#9.图像修复)
- [10.图像质量评价（Image Quality Assessment）](#10.图像质量评价)

<a name="1.超分辨率"></a>
# 1.超分辨率（Super-Resolution）
## 图像超分辨率
### Invertible Image Rescaling
- Paper：[https://arxiv.org/abs/2005.05650](https://arxiv.org/abs/2005.05650)
- Code：[https://github.com/pkuxmq/Invertible-Image-Rescaling](https://github.com/pkuxmq/Invertible-Image-Rescaling)
- Homepage：
- Analysis：[ECCV 2020 Oral | 可逆图像缩放：完美恢复降采样后的高清图片](https://zhuanlan.zhihu.com/p/150340687)
### Component Divide-and-Conquer for Real-World Image Super-Resolution
- Paper：[https://arxiv.org/abs/2008.01928](https://arxiv.org/abs/2008.01928)
- Code：[https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution)
### SRFlow: Learning the Super-Resolution Space with Normalizing Flow
- Paper：[https://arxiv.org/abs/2006.14200?context=eess](https://arxiv.org/abs/2006.14200?context=eess)
- Code：[https://github.com/andreas128/SRFlow](https://github.com/andreas128/SRFlow)
### Single Image Super-Resolution via a Holistic Attention Network
- Analysis：[ECCV2020最新图像超分辨重建文章](https://zhuanlan.zhihu.com/p/158083010)
### Stochastic Frequency Masking to Improve Super-Resolution and Denoising Networks
- Paper：[https://arxiv.org/abs/2003.07119](https://arxiv.org/abs/2003.07119)
- Code：[https://github.com/majedelhelou/SFM](https://github.com/majedelhelou/SFM)
### VarSR: Variational Super-Resolution Network for Very Low Resolution Images
### Learning with Privileged Information for Efficient Image Super-Resolutionq
- Paper：[https://arxiv.org/abs/2007.07524](https://arxiv.org/abs/2007.07524)
- Code：[https://github.com/pkuxmq/Invertible-Image-Rescaling](https://github.com/pkuxmq/Invertible-Image-Rescaling)
- Homepage：[https://cvlab.yonsei.ac.kr/projects/PISR/](https://cvlab.yonsei.ac.kr/projects/PISR/)
### Binarized Neural Network for Single Image Super Resolution
## 视频超分辨率
### Across Scales & Across Dimensions: Temporal Super-Resolution using Deep Internal Learning
- Paper：[https://arxiv.org/abs/2003.08872](https://arxiv.org/abs/2003.08872)
- Code:
- Homepage：[http://www.wisdom.weizmann.ac.il/~vision/DeepTemporalSR/](http://www.wisdom.weizmann.ac.il/~vision/DeepTemporalSR/)
- Analysis：
### MuCAN: Multi-Correspondence Aggregation Network for Video Super-Resolution
- Paper：[https://arxiv.org/abs/2007.11803v1](https://arxiv.org/abs/2007.11803v1)
### Video Super-Resolution with Recurrent Structure-Detail Network
- Paper：[https://arxiv.org/abs/2008.00455](https://arxiv.org/abs/2008.00455)
- Code：[https://github.com/junpan19/RSDN](https://github.com/junpan19/RSDN)
- Homepage：[http://www.wisdom.weizmann.ac.il/~vision/DeepTemporalSR/](http://www.wisdom.weizmann.ac.il/~vision/DeepTemporalSR/)
## 人脸超分辨率
### Face Super-Resolution Guided by 3D Facial Priors
- Paper：[https://arxiv.org/abs/2007.09454v1](https://arxiv.org/abs/2007.09454v1)
## 深度图超分辨率
## 光场图像超分辨率
### Spatial-Angular Interaction for Light Field Image Super-Resolution
- Paper：[https://arxiv.org/abs/1912.07849](https://arxiv.org/abs/1912.07849)
- Code：[https://github.com/YingqianWang/LF-InterNet](https://github.com/YingqianWang/LF-InterNet)
- Presentation：[https://wyqdatabase.s3-us-west-1.amazonaws.com/LF-InterNet.mp4](https://wyqdatabase.s3-us-west-1.amazonaws.com/LF-InterNet.mp4)
- Analysis：[ECCV 2020 | 空间-角度信息交互的光场图像超分辨，性能优异代码已开源](https://zhuanlan.zhihu.com/p/157115310)
## 高光谱图像超分辨率
### Cross-Attention in Coupled Unmixing Nets for Unsupervised Hyperspectral Super-Resolution
- Paper：[https://arxiv.org/abs/2007.05230](https://arxiv.org/abs/2007.05230)
- Code：[https://github.com/danfenghong/ECCV2020_CUCaNet](https://github.com/danfenghong/ECCV2020_CUCaNet)
## 零样本超分辨率
### Zero-Shot Image Super-Resolution with Depth Guided Internal Degradation Learning
### Fast Adaptation to Super-Resolution Networks via Meta-Learning
- Paper:[https://arxiv.org/abs/2001.02905v1](https://arxiv.org/abs/2001.02905v1)
## 文本超分辨率
### PlugNet: Degradation Aware Scene Text Recognition Supervised by a Pluggable Super-Resolution Unit
- Analysis：[ECCV 2020 | 图匠数据、华中师范提出低质退化文本识别算法PlugNet](https://zhuanlan.zhihu.com/p/157789166?from_voters_page=true)
### Scene Text Image Super-Resolution in the Wild
- Paper：[https://arxiv.org/abs/2005.03341v1](https://arxiv.org/abs/2005.03341v1)
- Code：[https://github.com/JasonBoy1/TextZoom](https://github.com/JasonBoy1/TextZoom)
## 绘画超分辨率
### Texture Hallucination for Large-Factor Painting Super-Resolution
- Paper：[https://arxiv.org/abs/1912.00515?context=eess.IV](https://arxiv.org/abs/1912.00515?context=eess.IV)
## 用于超分辨率的数据增广
## 超分辨率用于语义分割
## 超分辨率模型压缩/轻量化
### Journey Towards Tiny Perceptual Super-Resolution
- Paper:[https://arxiv.org/abs/2007.04356](https://arxiv.org/abs/2007.04356)
### LatticeNet: Towards Lightweight Image Super-resolution with Lattice Block
## 其他超分（暂存）
### Towards Content-independent Multi-Reference Super-Resolution: Adaptive Pattern Matching and Feature Aggregation
### PAMS: Quantized Super-Resolution via Parameterized Max Scale
### Mining self-similarity: Label super-resolution with epitomic representations

<a name="2.图像去雨"></a>
# 2.图像去雨（Image Deraining）
### Rethinking Image Deraining via Rain Streaks and Vapors
- Paper：[https://arxiv.org/abs/2008.00823](https://arxiv.org/abs/2008.00823)
- Code：[https://github.com/yluestc/derain](https://github.com/yluestc/derain)
### Beyond Monocular Deraining: Paired Rain Removal Networks via Unpaired Semantic Understanding

<a name="3.图像去雾"></a>
# 3.图像去雾（Image Dehazing）
### HardGAN: A Haze-Aware Representation Distillation GAN for Single Image Dehazing
### Physics-based Feature Dehazing Networks

<a name="4.去模糊"></a>
# 4.去模糊（Deblurring）
### End-to-end Interpretable Learning of Non-blind Image Deblurring
- Paper：[https://arxiv.org/abs/2007.01769](https://arxiv.org/abs/2007.01769)
- Code：
### Spatio-Temporal Efficient Recurrent Neural Network for Video Deblurring
### Multi-Temporal Recurrent Neural Networks For Progressive Non-Uniform Single Image Deblurring With Incremental Temporal Training
### Learning Event-Driven Video Deblurring and Interpolation
### Defocus Deblurring Using Dual-Pixel Data
### Real-World Blur Dataset for Learning and Benchmarking Deblurring Algorithms
### OID: Outlier Identifying and Discarding in Blind Image Deblurring
### Enhanced Sparse Model for Blind Deblurring

<a name="5.去噪"></a>
# 5.去噪（Denoising）
### Unpaired Learning of Deep Blind Image Denoising
### Practical Deep Raw Image Denoising on Mobile Devices
### Reconstructing the Noise Manifold for Image Denoising
### Burst Denoising via Temporally Shifted Wavelet Transforms
### Stochastic Frequency Masking to Improve Super-Resolution and Denoising Networks
### Learning Graph-Convolutional Representations for Point Cloud Denoising
### Spatial Hierarchy Aware Residual Pyramid Network for Time-of-Flight Depth Denoising
### A Decoupled Learning Scheme for Real-world Burst Denoising from Raw Images
### Robust and On-the-fly Dataset Denoising for Image Classification
### Spatial-Adaptive Network for Single Image Denoising

<a name="6.图像恢复"></a>
# 6.图像恢复（Image Restoration）
### Exploiting Deep Generative Prior for Versatile Image Restoration and Manipulation
- Paper：[https://arxiv.org/abs/2003.13659](https://arxiv.org/abs/2003.13659)
- Code：[https://github.com/XingangPan/deep-generative-prior](https://github.com/XingangPan/deep-generative-prior)
### PIPAL: a Large-Scale Image Quality Assessment Dataset for Perceptual Image Restoration
### Stacking Networks Dynamically for Image Restoration Based on the Plug-and-Play Framework
### LIRA: Lifelong Image Restoration from Unknown Blended Distortions
### Interactive Multi-Dimension Modulation with Dynamic Controllable Residual Learning for Image Restoration
### Microscopy Image Restoration with Deep Wiener-Kolmogorov filters
### Fully Trainable and Interpretable Non-Local Sparse Models for Image Restoration
### Learning Enriched Features for Real Image Restoration and Enhancement
### Learning Disentangled Feature Representation for Hybrid-distorted Image Restoration

<a name="7.图像增强"></a>
# 7.图像增强（Image Enhancement）
### URIE: Universal Image Enhancement for Visual Recognition in the Wild
### Early Exit Or Not: Resource-Efficient Blind Quality Enhancement for Compressed Images
### Global and Local Enhancement Networks For Paired and Unpaired Image Enhancement
### PieNet: Personalized Image Enhancement Network

<a name="8.图像去摩尔纹"></a>
# 8.图像去摩尔纹（Image Demoireing）
### Wavelet-Based Dual-Branch Neural Network for Image Demoireing
- Paper：[https://arxiv.org/abs/2007.07173](https://arxiv.org/abs/2007.07173)
- Analysis：[#每日五分钟一读# Image Demoireing](https://zhuanlan.zhihu.com/p/164778442)

<a name="9.图像修复"></a>
# 9.图像修复（Inpainting）
### Learning Joint Spatial-Temporal Transformations for Video Inpainting
- Paper：[https://arxiv.org/abs/2007.10247](https://arxiv.org/abs/2007.10247)
- Code：[https://github.com/researchmm/STTN](https://github.com/researchmm/STTN)
### Rethinking Image Inpainting via a Mutual Encoder-Decoder with Feature Equalizations
- Paper：[https://arxiv.org/abs/2007.06929](https://arxiv.org/abs/2007.06929)
- Code：[https://github.com/KumapowerLIU/ECCV2020oralRethinking-Image-Inpainting-via-a-Mutual-Encoder-Decoder-with-Feature-Equalizations](https://github.com/KumapowerLIU/ECCV2020oralRethinking-Image-Inpainting-via-a-Mutual-Encoder-Decoder-with-Feature-Equalizations)
- Analysis：[ECCV2020(Oral) Rethinking image inpainting](https://zhuanlan.zhihu.com/p/156893265)
### High-Resolution Image Inpainting with Iterative Confidence Feedback and Guided Upsampling
- Paper：[https://arxiv.org/abs/2005.11742v1](https://arxiv.org/abs/2005.11742v1)
### Short-Term and Long-Term Context Aggregation Network for Video Inpainting
### Learning Object Placement by Inpainting for Compositional Data Augmentation
### High-Resolution Image Inpainting with Iterative Confidence Feedback and Guided Upsampling
### DVI: Depth Guided Video Inpainting for Autonomous Driving
### VCNet: A Robust Approach to Blind Image Inpainting
### Guidance and Evaluation: Semantic-Aware Image Inpainting for Mixed Scenes

<a name="10.图像质量评价"></a>
# 10.图像质量评价（Image Quality Assessment）
### GIQA: Generated Image Quality Assessment
### PIPAL: a Large-Scale Image Quality Assessment Dataset for Perceptual Image Restoration
<font color=red size=5>持续更新~</font>

# 参考
<div class="output_wrapper" id="output_wrapper_id" style="font-size: 16px; color: rgb(62, 62, 62); line-height: 1.6; word-spacing: 0px; letter-spacing: 0px; font-family: 'Helvetica Neue', Helvetica, 'Hiragino Sans GB', 'Microsoft YaHei', Arial, sans-serif;"><p style="font-size: inherit; color: inherit; line-height: inherit; padding: 0px; margin: 1.5em 0px;"><a href="https://blog.csdn.net/yamengxi/article/details/107463400" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[1] ECCV 2020 超分辨率方向上接收文章总结</a><br><a href="https://blog.csdn.net/yyywxk/article/details/107116197" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[2] ECCV 2020 超分辨率方向上接收文章总结（持续更新）持续更新</a><br><a href="https://zhuanlan.zhihu.com/p/157115310" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[3] ECCV 2020 | 空间-角度信息交互的光场图像超分辨，性能优异代码已开源</a><br><a href="https://github.com/amusi/ECCV2020-Code" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[4] ECCV2020-Code</a><br><a href="https://zhuanlan.zhihu.com/p/157789166?from_voters_page=true" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[5] ECCV 2020 | 图匠数据、华中师范提出低质退化文本识别算法PlugNet</a><br><a href="https://zhuanlan.zhihu.com/p/156893265" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[6] ECCV2020(Oral) Rethinking image inpainting</a><br><a href="https://zhuanlan.zhihu.com/p/157569669" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[7] ECCV 2020 Oral 论文汇总！</a><br><a href="https://zhuanlan.zhihu.com/p/164778442" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[8] #每日五分钟一读# Image Demoireing</a></p></div>

<font color=red size=5>码字不易，如果您觉得有帮助，欢迎star~~</font>
