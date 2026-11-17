<h1 align="center">D-OPSD<br><sub><sup>On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models</sup></sub></h1>

<div align="center">

[![Project Page](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage)](https://vvvvvjdy.github.io/d-opsd/)&#160;
  <a href="https://github.com/vvvvvjdy/D-OPSD"><img src="https://img.shields.io/badge/GitHub(DOPSD)-9E95B7?logo=github"></a> &nbsp;
    <a href="https://github.com/Tongyi-MAI/Z-Image"><img src="https://img.shields.io/badge/GitHub(ZImage)-9E95B7?logo=github"></a> &nbsp;
<a href="https://arxiv.org/abs/2605.05204" target="_blank"><img src="https://img.shields.io/badge/Paper-b5212f.svg?logo=arxiv" height="21px"></a>

</div>

<p align="center">
  <img src="assets/method_01.jpg" alt="Result" style="width:100%;">
</p>

### 🛕 Environment Setup
```bash
git clone https://github.com/vvvvvjdy/D-OPSD.git
conda create -n dopsd python=3.12 -y
conda activate dopsd
pip install -r requirements.txt
```
****

### 🤖 Agent Guide for Adapting D-OPSD to New Models

If you want to adapt D-OPSD to new models or datasets, you can refer to our [agent guide](agent-guide/README.md) to help your coding agent understand the core ideas and implement the adaptation.


Note that teacher context can be constructed in different ways depending on the model and task. Further ideas and questions are welcome in [Issues](https://github.com/vvvvvjdy/D-OPSD/issues).

****


### 🍬 Training 

#### 1. D-OPSD Z-Image-Turbo LoRA training with self-distilled vlm context (align with our paper):

Refer to [ (z-image-turbo_self-distill-vlm)](z-image-turbo_self-distill-vlm/README.md) for training guidance.

#### 2. D-OPSD FLUX2-klein LoRA training with self-distilled eidting branch context for scenario of high id accuracy requirement:

Refer to [ (flux2-klein_self-distill-edit)](flux2-klein_self-distill-edit/README.md) for training guidance and more discussion.

#### 3. D-OPSD FLUX2-klein LoRA training for image editing with self-distilled target image as reference imgae in teacher:

Refer to [ (flux2-klein-edit_self-distill-edit)](flux2-klein-edit-self-distill-gt-ref/README.md) for training guidance and more discussion.

****

### 🍬 Evaluation

Refer to [ (z-image-turbo_self-distill-vlm-eval)](z-image-turbo_self-distill-vlm/eval/README.md) for evaluation guidance with Z-Image-Turbo and more discussion.

****


### 🎀 Highlight

D-OPSD is an on-policy self-distillation training framework for diffusion models especially timestep-distilled ones. It features in:

- D-OPSD identify an emergent property of modern text to image diffusion models with
LLM/VLM encoders and utilize this property to the continuous tuning of step-distilled
diffusion model.
- D-OPSD is a novel diffusion models on-policy self-distillation framework. By
assigning the same model two roles with different contexts, D-OPSD enables supervised
tuning on the student’s own roll-outs without requiring any external reward function or
extra modules.
- D-OPSD is validated in different settings. The results show that our method enables
the model to learn new concepts, styles, and domain preferences while preserving its
original few-step inference capability and previous knowledge.



---


<p align="center">
   <figcaption style="text-align: center; margin-top: 10px; font-size: 0.92em;">
           In full fine-tuning, D-OPSD adapts the model toward the target domain (anime) while retaining original-domain knowledge and few-step inference capability.
        </figcaption>
  <img src="assets/full_compare_01.jpg" alt="Result" style="width:96%;">
</p>

---


<p align="center">
   <figcaption style="text-align: center; margin-top: 10px; font-size: 0.92em;">
           In small customized LoRA training, D-OPSD learns new concepts from only a few image-text pairs while maintaining few-step generation quality and generalizing to unseen prompts.
        </figcaption>
  <img src="assets/lora_compare_01.jpg" alt="Result" style="width:90%;">
</p>

---




### 🌺 Citation
If you find D-OPSD useful, please kindly cite our paper:
```bibtex
@article{jiang2026dopsd,
      title={D-OPSD: On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models},
      author={Jiang, Dengyang and Jin, Xin and Liu, Dongyang and Wang, Zanyi and Zheng, Mingzhe and Du, Ruoyi and Yang, Xiangpeng and Wu, Qilong and Li, Zhen and Gao, Peng and Yang, Harry and Hoi, Steven},
      journal={arXiv preprint arXiv:2605.05204},
      year={2026}
}
```
