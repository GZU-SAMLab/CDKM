# CDKM: Common and Distinct Knowledge Mining Network with Content Interaction for Dense Captioning
The dense captioning task aims at detecting multiple salient regions of an image and describing them separately in natural language. Although significant advancements have been made in recent years in the field
of dense captioning, there are still some limitations to existing methods. On the one hand, most dense captioning methods lack strong target detection capabilities and struggle to cover all relevant content when dealing with target-intensive images. On the other hand, current transformer-based methods are powerful but neglect the acquisition and utilization of contextual information, hindering the visual understanding of local areas. To address these issues, we propose a common and distinct knowledge-mining network with content interaction for the task of dense captioning. Our network has a knowledge mining mechanism that improves the detection of salient targets by capturing common and distinct knowledge from multi-scale features. We further propose a content interaction module that combines region features into a unique context based on their correlation. Our experiments on various benchmarks have shown that the proposed method outperforms the current state-of-the-art methods.

## Overview
![The framework of the proposed method](./images/p2.jpg)


## Visualization
![Some of the visualizations results from our proposed method on VG v1.2 test split](./images/p4.jpg)


## Acknowledgement
Part of our code comes from the following work, and we are very grateful for their contributions to relevant research:
[GRiT](https://github.com/JialianW/GRiT),
[Detic](https://github.com/facebookresearch/Detic),
[CenterNet2](https://github.com/xingyizhou/CenterNet2),
[detectron2](https://github.com/facebookresearch/detectron2),
[GIT](https://github.com/microsoft/GenerativeImage2Text), and
[transformers](https://github.com/huggingface/transformers). 
We thank the authors and appreciate their great works!