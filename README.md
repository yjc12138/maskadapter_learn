<div align ="center">
<img src="./assets/logo.jpg" width="20%">
<h1> Mask-Adapter </h1>
<h3> The Devil is in the Masks for Open-Vocabulary Segmentation </h3>

YongKang Li<sup>1,\*</sup>, [Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ&hl=zh-CN)<sup>1,\*</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu)<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1,📧</sup>

<sup>1</sup> Huazhong University of Science and Technology,


(\* equal contribution, 📧 corresponding author)

[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)]()
[![checkpoints](https://img.shields.io/badge/HuggingFace-🤗_Weight-orange)](https://huggingface.co/owl10/Mask-Adapter)
[![🤗 HuggingFace Demo](https://img.shields.io/badge/Mask_Adapter-🤗_HF_Demo-orange)](https://huggingface.co/spaces/wondervictor/Mask-Adapter)

</div>


<div align="center">
<img src="./assets/main_fig.png">
</div>

## Highlights

* Mask-Adapter is a simple yet remarkably effective method and can be seamlessly integrated into open-vocabulary segmentation methods, e.g., [FC-CLIP](https://github.com/bytedance/fc-clip) and [MAFT-Plus](https://github.com/jiaosiyu1999/MAFT-Plus), to tackle the existing bottlenecks.

* Mask-Adapter effectively extends to SAM without training, achieving impressive results across multiple open-vocabulary segmentation benchmarks.

## Updates
- [x] Release code
- [x] Release weights
- [x] Release demo with SAM-2👉 [🤗 Mask-Adapter](https://huggingface.co/spaces/wondervictor/Mask-Adapter)
- [ ] Release weights training with addtional data


## Getting Started
+ [Installation](INSTALL.md).

+ [Preparing Datasets for Mask-Adapter](datasets/README.md).

+ [Getting Started with Mask-Adapter](GETTING_STARTED.md).

## Models

| Model | Backbone | A-847 | A-150 | PC-459 | PC-59 | PAS-20 | Download |
|:---: |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|FC-CLIP + Mask-Adapter|ConvNeXt-L|14.1|36.6|19.3|59.7|95.5|[model](https://drive.google.com/file/d/13_sr30_Q0Geubijik0BpVC_JgyFAmyQU/view?usp=drive_link) |
|MAFTP-Base + Mask-Adapter|ConvNeXt-B|14.2|35.6|17.9|58.4|95.1 |[model](https://drive.google.com/file/d/1v0rdETOJl6oOKmef1L7WbtG16-zKvp2b/view?usp=drive_link)|
|MAFTP-Large + Mask-Adapter|ConvNeXt-L|16.1|38.2|22.7|60.4|95.8 |[model](https://drive.google.com/file/d/12eqDnTYaQlj9QUmWO1Vh9vvB81tKABl5/view?usp=drive_link) |

## Citation
If you Mask-Adapter useful in your research or applications, please consider giving us a star &#127775; and citing it by the following BibTeX entry.

```BibTeX

```
## License
All code in this repository is under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).


## Acknowledgement

Mask-Adapter is based on the following projects: [detectron2](https://github.com/facebookresearch/detectron2), [Mask2Former](https://github.com/facebookresearch/Mask2Former), [FC-CLIP](https://github.com/bytedance/fc-clip) and [MAFTP](https://github.com/jiaosiyu1999/MAFT-Plus). Many thanks for their excellent contributions to the community.



