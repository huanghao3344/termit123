# termit123
termit dataset
# A Lightweight Deep Learning Model Based on Improved Mo-bileNetV2 for Fine-Grained Termite Identification With Color Attention
# Research Background

# Model Architecture

Overall Architecture of MobileNetV2-CAB-TL2

The overall network architecture of the proposed MobileNetV2-CAB-TL2 model. To meet the critical requirements of real-time pest monitoring on resource-constrained edge platforms, MobileNetV2 is adopted as the foundational backbone network due to its exceptional trade-off between computational lightweightness and feature representation capability.
To further enhance the model's sensitivity to fine-grained biological traits under complex field environments, two key structural enhancements are integrated into the pipeline. Specifically, the model consists of three primary sequential stages:

1.Feature Extraction Backbone: The input image is first processed by the preliminary convolutional layers of MobileNetV2 to extract shallow-level spatial features.

2.Color-Aware Attention Embedding: A custom-designed Color-Aware Attention Block (CAB) is embedded into the shallow layers of the backbone network.
By utilizing a specialized channel attention mechanism, the CAB adaptively recalibrates feature responses to selectively amplify discriminative color-sensitive patterns and subtle body surface textures of polymorphic termites.

3.Hierarchical Classification Head: The refined feature maps are aggregated through a global average pooling layer and fed into the final classification head. 
To optimize adaptation, a hierarchical transfer learning strategy (TL2) is deployed, which assigns differentiated learning rates to the backbone network and the classification head, 
ensuring robust fine-grained adaptation while preserving generic semantic representations.

# Funding & Data Statement
This research was supported by Zhejiang Provincial Natural Science Foundation of China under Grant No. LY23C140004. In collaboration with local government departments, 
this project has signed relevant confidentiality agreements. For this reason, we only display partial data samples instead of sharing the complete raw data.
