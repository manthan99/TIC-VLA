 # <img src="docs/img/logo.png" width="35" height="35" style="vertical-align: bottom; margin-right: 10px;"> TIC-VLA

[![website](https://img.shields.io/badge/Website-Explore%20Now-blueviolet?style=flat&logo=google-chrome)](https://ucla-mobility.github.io/TIC-VLA/)
[![paper](https://img.shields.io/badge/ICML-2026-red.svg)](https://arxiv.org/abs/2602.02459)
[![dataset](https://img.shields.io/badge/Dataset-HuggingFace-F9D371.svg)](https://huggingface.co/datasets/handsomeYun/TIC-VLA)
<!-- [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)]() -->

<!-- This is the official implementation for the following paper: -->

**[ICML 2026] TIC-VLA: A Think-in-Control Vision-Language-Action Model for Robot Navigation in Dynamic Environments**

**[Zhiyu Huang](https://mczhi.github.io/)**<sup>†</sup>, **[Yun Zhang](https://handsomeyun.github.io/)**<sup>†</sup>, [Johnson Liu](https://www.linkedin.com/in/johnsonliu367/), [Rui Song](https://rruisong.github.io/), [Chen Tang](https://chentangmark.github.io/), [Jiaqi Ma](https://mobility-lab.seas.ucla.edu/about/)  

University of California, Los Angeles (UCLA)  
<sup>†</sup> Equal contribution

![overview](docs/img/framework.png)

---

## Overview

Stay tuned for new updates!

TIC-VLA introduces a **latency-aware Think-in-Control (TIC) architecture** for vision-language-action (VLA) model for robot navigation in **dynamic, human-centric environments**.  

- 🧠 **Think-in-Control Architecture**  
  Decouples slow vision-language reasoning from fast reactive control through an explicit **delayed semantic–control interface**.

- ⏱️ **Latency-Aware Action Generation**  
  Conditions control on current observations, cached VLM hidden states, and explicit delay metadata to mitigate stale semantics.

- 🧪 **Latency-Consistent Training Pipeline**  
  Combines vision-language reasoning distillation, latency-induced imitation learning, and online reinforcement learning.

- 🚶 **Dynamic, Human-Centric Navigation**  
  Evaluated in physics-accurate, photo-realistic environments with human-robot interactions and long-horizon instructions.

---

## Benchmark: DynaNav

We introduce **DynaNav**, a language-conditioned navigation benchmark designed to test VLA systems under realistic scenarios.

- 85 task configurations across **Hospital**, **Office**, **Warehouse**, and **Outdoor** scenes
- Varying **crowd density**, **navigation distance**, and **scene layout**

![benchmark](docs/img/benchmark.png)

---
## Release Plan
We are currently organizing the project for public release.
- 📦 **Code Release:** June 2026
- 🗂️ **DynaNav Dataset Release:** June 2026
- 🤖 Training scripts, evaluation benchmarks, and pretrained checkpoints will also be released.
Stay tuned for updates!
---

## Citation
If you find this repository useful for your research, please consider giving us a star 🌟 and citing our paper.
 ```bibtex
@inproceedings{huang2026ticvla,
  title={TIC-VLA: A Think-in-Control Vision-Language-Action Model for Robot Navigation in Dynamic Environments},
  author={Zhiyu Huang and Yun Zhang and Johnson Liu and Rui Song and Chen Tang and Jiaqi Ma},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2026}
}
