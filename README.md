# termit123
termit dataset
# A Lightweight Deep Learning Model Based on Improved Mo-bileNetV2 for Fine-Grained Termite Identification With Color Attention
# Research Background
Termites are among the most destructive, highly cryptic structural and agricultural insect pests worldwide, threatening the functional stability of agroecosystems by inflicting devastating, irreversible damage on food crop cultivation, economic forestry production, reservoir dams, vital agricultural water-conservancy infrastructure, and rustic building structures.
It is estimated that subterranean and wood-feeding termite infestations propagate rapidly beneath the soil profile, inducing massive global direct economic losses exceed-ing $40 billion annually. Crucially, the high cryptic foraging behavior of these bio-logical threats renders early-stage detection extremely challenging, making field-level surveillance a matter of critical temporal urgency before irreversible agricultural infrastructure or root system degradation occurs. Within functional termite colonies, distinct castes—predominantly soldiers and workers—execute specialized behavioral configurations spanning colony defense, aggressive foraging, and complex nest engineering. From the perspective of Integrated Pest Management (IPM), distinct termite genera and species exhibit highly divergent ecological niches and nesting habits; for instance, subterranean termites (e.g., Coptotermes or Odontotermes) construct deep-seated nests threatening water-conservancy infrastructure, whereas drywood termites (e.g., Cryptotermes) confined themselves entirely within rustic timber structures without soil contact. As a consequence of these specialized biological configurations and generic disparities, they exhibit highly disparate crop damage dynamics, spatial migration patterns, and unique physiological sensitivities to tailored chemical or bait-based baiting treatments. Furthermore, tracking the specific caste compositions (such as the soldier-to-worker ratio) serves as a critical biological barometer for evaluating the horizontal transfer efficacy of chronic baiting systems within the colony.Therefore, accurate and rapid fine-grained identification of termite species and castes is a critical prerequisite for assessing infestation severity and developing targeted integrated pest management strategies, such as physical barriers, baiting systems, and localized chemical treatments. However, as polymorphic insects, termites are difficult to collect in their adult form, and exhibit high morphological similarity within colonies. Key discriminative features among different species and castes, such as head capsule proportions, mandible morphology, and subtle body coloration patterns, are highly similar. The complexity of these fine-grained features poses significant challenges for real-time field identification, particularly for non-experts.


# Model Architecture /[train_termite_cls.py](https://github.com/huanghao3344/termit123/blob/main/train_termite_cls.py)

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
