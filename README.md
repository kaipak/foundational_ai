# foundational_ai
A re-implementation and study of seminal deep learning architectures and breakthroughs. This repository is loosely organized into topics of computer vision (CV), natural language processing (NLP), and other emerging areas as identified.

Each section is dedicated to a landmark paper in AI. The repository includes PyTorch code representing the model, along with scripts for running the models with minimal setup. Notes and some results are also provided.

# Requirements
All code in this repository is written in Python 3.12 and PyTorch 2.x. Conda is used to manage Python environments, and an `environment.yml` file is provided for each section. This is prototype-level code, primarily tested on Ubuntu Server Linux distributions (22.04 as of April 2024). The code may work on similar systems but is not guaranteed to be production quality.

# AI Topics
The papers are organized into sections covering major areas of research in AI.

## Computer Vision (CV)
### [ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) by Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
- **Synopsis:** Introduced AlexNet, which significantly outperformed existing models in the ImageNet competition, marking a major advancement in the use of deep learning for image processing.

### [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet)](https://arxiv.org/abs/1409.1556) by Karen Simonyan, Andrew Zisserman
- **Synopsis:** This paper introduces VGGNet, enhancing the depth of convolutional networks significantly and showing improvements in accuracy are possible by increasing network depth.

### [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Synopsis:** ResNet introduces a residual learning framework to ease the training of networks that are substantially deeper than those used previously, demonstrating substantial improvements over previous architectures.

### [You Only Look Once: Unified, Real-Time Object Detection (YOLO)](https://arxiv.org/abs/1506.02640) by Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
- **Synopsis:** YOLO frames object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities, achieving state-of-the-art detection speeds.

### [Mask R-CNN](https://arxiv.org/abs/1703.06870) by Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick
- **Synopsis:** Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation masks on each Region of Interest, combining the benefits of both detection and segmentation.

### [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) by Mingxing Tan, Quoc V. Le
- **Synopsis:** This paper introduces EfficientNet, a systematic method for scaling up CNNs that uses a compound coefficient to scale up CNNs in a more structured manner.

## Natural Language Processing (NLP) including Large Language Models (LLMs)
### [Efficient Estimation of Word Representations in Vector Space (Word2Vec)](https://arxiv.org/abs/1301.3781) by Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
- **Synopsis:** Introduces models for efficient estimation of word representations in vector space, leading to significant advancements in the quality of vector representations for words.

### [Attention is All You Need (Transformer)](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit
- **Synopsis:** Introduces the Transformer model, which dispenses with recurrence and convolutions entirely, relying solely on an attention mechanism to draw global dependencies between input and output.

### [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- **Synopsis:** BERT introduces a new method of pre-training language representations which obtain state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks.

### [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) by Tom B. Brown, Benjamin Mann, Nick Ryder, et al.
- **Synopsis:** Demonstrates that scaling up language models significantly improves task-agnostic, few-shot performance, paving the way for models that can learn from a minimal amount of data.

## Science Applications of Deep Learning
### [Highly accurate protein structure prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) by John Jumper, Richard Evans, Alexander Pritzel, et al.
- **Synopsis:** AlphaFold demonstrates a significant leap in the capability of computational methods to predict protein structure from amino acid sequence alone, providing insights into protein folding and stability.

### [Deep learning and process understanding for data-driven Earth system science](https://www.nature.com/articles/s41586-019-0912-1) by Markus Reichstein, Gustau Camps-Valls, Bjorn Stevens, et al.
- **Synopsis:** This paper explores the use of deep learning in understanding complex Earth systems, focusing on climate and ecological sciences, highlighting the potential for AI to help in tackling some of the most pressing environmental challenges.

# Licensing
This project is licensed under the MIT License.
