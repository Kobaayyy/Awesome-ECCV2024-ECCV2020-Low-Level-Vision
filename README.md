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
- [11.插帧（Frame Interpolation）](#11.插帧)
- [12.视频/图像压缩（Video/Image Compression）](#12.视频压缩)
- [13.其他多任务](#13.其他)

<a name="1.超分辨率"></a>
# 1.超分辨率（Super-Resolution）
## 图像超分辨率
### Invertible Image Rescaling
- Paper：[https://arxiv.org/abs/2005.05650](https://arxiv.org/abs/2005.05650)
- Code：[https://github.com/pkuxmq/Invertible-Image-Rescaling](https://github.com/pkuxmq/Invertible-Image-Rescaling)
- Analysis：[ECCV 2020 Oral | 可逆图像缩放：完美恢复降采样后的高清图片](https://zhuanlan.zhihu.com/p/150340687)
### Component Divide-and-Conquer for Real-World Image Super-Resolution
- Paper：[https://arxiv.org/abs/2008.01928](https://arxiv.org/abs/2008.01928)
- Code：[https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution)
### SRFlow: Learning the Super-Resolution Space with Normalizing Flow
- Paper：[https://arxiv.org/abs/2006.14200?context=eess](https://arxiv.org/abs/2006.14200?context=eess)
- Code：[https://github.com/andreas128/SRFlow](https://github.com/andreas128/SRFlow)
### Single Image Super-Resolution via a Holistic Attention Network
- Paper：[https://arxiv.org/abs/2008.08767](https://arxiv.org/abs/2008.08767)
- Code：[https://github.com/wwlCape/HAN](https://github.com/wwlCape/HAN)
- Analysis：[ECCV2020最新图像超分辨重建文章](https://zhuanlan.zhihu.com/p/158083010)
### Stochastic Frequency Masking to Improve Super-Resolution and Denoising Networks
- Paper：[https://arxiv.org/abs/2003.07119](https://arxiv.org/abs/2003.07119)
- Code：[https://github.com/majedelhelou/SFM](https://github.com/majedelhelou/SFM)
### VarSR: Variational Super-Resolution Network for Very Low Resolution Images
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680426.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680426.pdf)
### Learning with Privileged Information for Efficient Image Super-Resolutionq
- Paper：[https://arxiv.org/abs/2007.07524](https://arxiv.org/abs/2007.07524)
- Code：[https://github.com/cvlab-yonsei/PISR](https://github.com/cvlab-yonsei/PISR)
- Homepage：[https://cvlab.yonsei.ac.kr/projects/PISR/](https://cvlab.yonsei.ac.kr/projects/PISR/)
### Binarized Neural Network for Single Image Super Resolution
- Paper：[http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490086.pdf](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490086.pdf)
### Towards Content-independent Multi-Reference Super-Resolution: Adaptive Pattern Matching and Feature Aggregation
- Paper:[http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700052.pdf](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700052.pdf)
## 视频超分辨率
### Across Scales & Across Dimensions: Temporal Super-Resolution using Deep Internal Learning
- Paper：[https://arxiv.org/abs/2003.08872](https://arxiv.org/abs/2003.08872)
- Code:[https://github.com/eyalnaor/DeepTemporalSR](https://github.com/eyalnaor/DeepTemporalSR)
- Homepage：[http://www.wisdom.weizmann.ac.il/~vision/DeepTemporalSR/](http://www.wisdom.weizmann.ac.il/~vision/DeepTemporalSR/)
### MuCAN: Multi-Correspondence Aggregation Network for Video Super-Resolution
- Paper：[https://arxiv.org/abs/2007.11803v1](https://arxiv.org/abs/2007.11803v1)
### Video Super-Resolution with Recurrent Structure-Detail Network
- Paper：[https://arxiv.org/abs/2008.00455](https://arxiv.org/abs/2008.00455)
- Code：[https://github.com/junpan19/RSDN](https://github.com/junpan19/RSDN)
- Homepage：[http://www.wisdom.weizmann.ac.il/~vision/DeepTemporalSR/](http://www.wisdom.weizmann.ac.il/~vision/DeepTemporalSR/)
## 人脸超分辨率
### Face Super-Resolution Guided by 3D Facial Priors
- Paper：[https://arxiv.org/abs/2007.09454v1](https://arxiv.org/abs/2007.09454v1)
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
- Paper:[http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620256.pdf](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620256.pdf)
### Fast Adaptation to Super-Resolution Networks via Meta-Learning
- Paper:[https://arxiv.org/abs/2001.02905v1](https://arxiv.org/abs/2001.02905v1)
- Code:[https://github.com/parkseobin/MLSR](https://github.com/parkseobin/MLSR)
## 文本超分辨率
### PlugNet: Degradation Aware Scene Text Recognition Supervised by a Pluggable Super-Resolution Unit
- Paper:[http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600154.pdf](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600154.pdf)
- Analysis：[ECCV 2020 | 图匠数据、华中师范提出低质退化文本识别算法PlugNet](https://zhuanlan.zhihu.com/p/157789166?from_voters_page=true)
### Scene Text Image Super-Resolution in the Wild
- Paper：[https://arxiv.org/abs/2005.03341v1](https://arxiv.org/abs/2005.03341v1)
- Code：[https://github.com/JasonBoy1/TextZoom](https://github.com/JasonBoy1/TextZoom)
## 绘画超分辨率
### Texture Hallucination for Large-Factor Painting Super-Resolution
- Paper：[https://arxiv.org/abs/1912.00515?context=eess.IV](https://arxiv.org/abs/1912.00515?context=eess.IV)
## 超分辨率模型压缩/轻量化
### Journey Towards Tiny Perceptual Super-Resolution
- Paper:[https://arxiv.org/abs/2007.04356](https://arxiv.org/abs/2007.04356)
### LatticeNet: Towards Lightweight Image Super-resolution with Lattice Block
- Paper:[http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670273.pdf](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670273.pdf)
### PAMS: Quantized Super-Resolution via Parameterized Max Scale
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700562.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700562.pdf)
## 标记超分
### Mining self-similarity: Label super-resolution with epitomic representations
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710528.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710528.pdf)
- Code:[https://github.com/anthonymlortiz/epitomes_lsr](https://github.com/anthonymlortiz/epitomes_lsr)

<a name="2.图像去雨"></a>
# 2.图像去雨（Image Deraining）
### Rethinking Image Deraining via Rain Streaks and Vapors
- Paper：[https://arxiv.org/abs/2008.00823](https://arxiv.org/abs/2008.00823)
- Code：[https://github.com/yluestc/derain](https://github.com/yluestc/derain)
### Beyond Monocular Deraining: Paired Rain Removal Networks via Unpaired Semantic Understanding
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720069.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720069.pdf)

<a name="3.图像去雾"></a>
# 3.图像去雾（Image Dehazing）
### HardGAN: A Haze-Aware Representation Distillation GAN for Single Image Dehazing
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510715.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510715.pdf)
- Code:[https://github.com/huangzilingcv/HardGAN](https://github.com/huangzilingcv/HardGAN)
### Physics-based Feature Dehazing Networks
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750188.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750188.pdf)

<a name="4.去模糊"></a>
# 4.去模糊（Deblurring）
### End-to-end Interpretable Learning of Non-blind Image Deblurring
- Paper：[https://arxiv.org/abs/2007.01769](https://arxiv.org/abs/2007.01769)
- Code：[https://github.com/teboli/CPCR](https://github.com/teboli/CPCR)
### Efficient Spatio-Temporal Recurrent Neural Network for Video Deblurring
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510188.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510188.pdf)
- Code：[https://github.com/zzh-tech/ESTRNN](https://github.com/zzh-tech/ESTRNN)
### Multi-Temporal Recurrent Neural Networks For Progressive Non-Uniform Single Image Deblurring With Incremental Temporal Training
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510324.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510324.pdf)
- Code:[https://github.com/Dong1P/MTRNN](https://github.com/Dong1P/MTRNN)
### Learning Event-Driven Video Deblurring and Interpolation
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530681.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530681.pdf)
### Defocus Deblurring Using Dual-Pixel Data
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550120.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550120.pdf)
- Code:[https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel)
### Real-World Blur Dataset for Learning and Benchmarking Deblurring Algorithms
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700188.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700188.pdf)
- Code:[https://github.com/rimchang/RealBlur](https://github.com/rimchang/RealBlur)
### OID: Outlier Identifying and Discarding in Blind Image Deblurring
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700596.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700596.pdf)
### Enhanced Sparse Model for Blind Deblurring
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700630.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700630.pdf)

<a name="5.去噪"></a>
# 5.去噪（Denoising）
### Unpaired Learning of Deep Image Denoising
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490341.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490341.pdf)
- Code:[https://github.com/XHWXD/DBSN](https://github.com/XHWXD/DBSN)
### Practical Deep Raw Image Denoising on Mobile Devices
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510001.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510001.pdf)
### Reconstructing the Noise Variance Manifold for Image Denoising
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540596.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540596.pdf)
### Burst Denoising via Temporally Shifted Wavelet Transforms
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580239.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580239.pdf)
### Stochastic Frequency Masking to Improve Super-Resolution and Denoising Networks
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610732.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610732.pdf)
- Code：[https://github.com/majedelhelou/SFM](https://github.com/majedelhelou/SFM)
### Learning Graph-Convolutional Representations for Point Cloud Denoising
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650103.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650103.pdf)
- Code：[https://github.com/diegovalsesia/GPDNet](https://github.com/diegovalsesia/GPDNet)
### Spatial Hierarchy Aware Residual Pyramid Network for Time-of-Flight Depth Denoising
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690035.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690035.pdf)
### A Decoupled Learning Scheme for Real-world Burst Denoising from Raw Images
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700154.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700154.pdf)
### Spatial-Adaptive Network for Single Image Denoising
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750171.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750171.pdf)
- Code:[https://github.com/JimmyChame/SADNet](https://github.com/JimmyChame/SADNet)

<a name="6.图像恢复"></a>
# 6.图像恢复（Image Restoration）
### Exploiting Deep Generative Prior for Versatile Image Restoration and Manipulation
- Paper：[https://arxiv.org/abs/2003.13659](https://arxiv.org/abs/2003.13659)
- Code：[https://github.com/XingangPan/deep-generative-prior](https://github.com/XingangPan/deep-generative-prior)
### Stacking Networks Dynamically for Image Restoration Based on the Plug-and-Play Framework
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580443.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580443.pdf)
### LIRA: Lifelong Image Restoration from Unknown Blended Distortions
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630596.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630596.pdf)
### Interactive Multi-Dimension Modulation with Dynamic Controllable Residual Learning for Image Restoration
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650052.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650052.pdf)
- Code：[https://github.com/hejingwenhejingwen/CResMD](https://github.com/hejingwenhejingwen/CResMD)
### Microscopy Image Restoration with Deep Wiener-Kolmogorov filters
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650188.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650188.pdf)
- Code：[https://github.com/vpronina/DeepWienerRestoration/](https://github.com/vpronina/DeepWienerRestoration/)
### Fully Trainable and Interpretable Non-Local Sparse Models for Image Restoration
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670239.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670239.pdf)
- Code：[https://github.com/bruno-31/groupsc](https://github.com/bruno-31/groupsc)
### Learning Enriched Features for Real Image Restoration and Enhancement
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700494.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700494.pdf)
- Code：[https://github.com/swz30/MIRNet](https://github.com/swz30/MIRNet)
### Learning Disentangled Feature Representation for Hybrid-distorted Image Restoration
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740307.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740307.pdf)
### Blind Face Restoration via Deep Multi-scale Component Dictionaries
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540375.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540375.pdf)
- Code：[https://github.com/csxmli2016/DFDNet](https://github.com/csxmli2016/DFDNet)

<a name="7.图像增强"></a>
# 7.图像增强（Image Enhancement）
### URIE: Universal Image Enhancement for Visual Recognition in the Wild
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540715.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540715.pdf)
- Code：[https://github.com/taeyoungson/urie](https://github.com/taeyoungson/urie)
### Early Exit Or Not: Resource-Efficient Blind Quality Enhancement for Compressed Images
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610273.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610273.pdf)
- Code：[https://github.com/RyanXingQL/RBQE](https://github.com/RyanXingQL/RBQE)
### Global and Local Enhancement Networks For Paired and Unpaired Image Enhancement
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700341.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700341.pdf)
### PieNet: Personalized Image Enhancement Network
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750375.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750375.pdf)
### Low Light Video Enhancement using Synthetic Data Produced with an Intermediate Domain Mapping
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580103.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580103.pdf)
- Code：[https://github.com/sjmoran/SIDGAN](https://github.com/sjmoran/SIDGAN)
### Multi-level Wavelet-based Generative Adversarial Network for Perceptual Quality Enhancement of Compressed Video
- Paper:[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590392.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590392.pdf)
- Code：[https://github.com/IceClear/MW-GAN](https://github.com/IceClear/MW-GAN)
### Conditional Sequential Modulation for Efficient Global Image Retouching
- Paper:[http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580664.pdf](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580664.pdf)
- Code：[https://github.com/hejingwenhejingwen/CSRNet)
<a name="8.图像去摩尔纹"></a>
# 8.图像去摩尔纹（Image Demoireing）
### Wavelet-Based Dual-Branch Neural Network for Image Demoireing
- Paper：[https://arxiv.org/abs/2007.07173](https://arxiv.org/abs/2007.07173)
- Analysis：[#每日五分钟一读# Image Demoireing](https://zhuanlan.zhihu.com/p/164778442)
### FHDe²Net: Full High Definition Demoireing Network
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670715.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670715.pdf)

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
- Homepage：[https://zengxianyu.github.io/iic/](https://zengxianyu.github.io/iic/)
### Short-Term and Long-Term Context Aggregation Network for Video Inpainting
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490698.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490698.pdf)
### Learning Object Placement by Inpainting for Compositional Data Augmentation
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580562.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580562.pdf)
### DVI: Depth Guided Video Inpainting for Autonomous Driving
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660001.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660001.pdf)
- Code：[https://github.com/sibozhang/Depth-Guided-Inpainting](https://github.com/sibozhang/Depth-Guided-Inpainting)
### VCNet: A Robust Approach to Blind Image Inpainting
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700749.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700749.pdf)
- Code：[https://github.com/shepnerd/blindinpainting_vcnet](https://github.com/shepnerd/blindinpainting_vcnet)
### Guidance and Evaluation: Semantic-Aware Image Inpainting for Mixed Scenes
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720681.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720681.pdf)

<a name="10.图像质量评价"></a>
# 10.图像质量评价（Image Quality Assessment）
### PIPAL: a Large-Scale Image Quality Assessment Dataset for Perceptual Image Restoration
- Paper:[https://arxiv.org/pdf/2007.12142.pdf](https://arxiv.org/pdf/2007.12142.pdf)
### GIQA: Generated Image Quality Assessment
- Paper：[https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700749.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700749.pdf)
- Code：[https://github.com/cientgu/GIQA](https://github.com/cientgu/GIQA)

<a name="11.插帧"></a>
# 11.插帧（Frame Interpolation）

<a name="12.视频压缩"></a>
# 12.视频/图像压缩（Video/Image Compression）
<font color=red size=5>持续更新~</font>

<a name="13.其他"></a>
# 13.其他多任务

# 参考
<div class="output_wrapper" id="output_wrapper_id" style="font-size: 16px; color: rgb(62, 62, 62); line-height: 1.6; word-spacing: 0px; letter-spacing: 0px; font-family: 'Helvetica Neue', Helvetica, 'Hiragino Sans GB', 'Microsoft YaHei', Arial, sans-serif;"><p style="font-size: inherit; color: inherit; line-height: inherit; padding: 0px; margin: 1.5em 0px;"><a href="https://blog.csdn.net/yamengxi/article/details/107463400" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[1] ECCV 2020 超分辨率方向上接收文章总结</a><br><a href="https://blog.csdn.net/yyywxk/article/details/107116197" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[2] ECCV 2020 超分辨率方向上接收文章总结（持续更新）持续更新</a><br><a href="https://zhuanlan.zhihu.com/p/157115310" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[3] ECCV 2020 | 空间-角度信息交互的光场图像超分辨，性能优异代码已开源</a><br><a href="https://github.com/amusi/ECCV2020-Code" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[4] ECCV2020-Code</a><br><a href="https://zhuanlan.zhihu.com/p/157789166?from_voters_page=true" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[5] ECCV 2020 | 图匠数据、华中师范提出低质退化文本识别算法PlugNet</a><br><a href="https://zhuanlan.zhihu.com/p/156893265" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[6] ECCV2020(Oral) Rethinking image inpainting</a><br><a href="https://zhuanlan.zhihu.com/p/157569669" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[7] ECCV 2020 Oral 论文汇总！</a><br><a href="https://zhuanlan.zhihu.com/p/164778442" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[8] #每日五分钟一读# Image Demoireing</a></p></div>

<font color=red size=5>码字不易，如果您觉得有帮助，欢迎star~~</font>

# 相关Low-Level-Vision整理
- [Awesome-CVPR2021/CVPR2020-Low-Level-Vision](https://github.com/Kobaayyy/Awesome-CVPR2021-CVPR2020-Low-Level-Vision)
