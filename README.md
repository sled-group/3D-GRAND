# 3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination

# 3D-GRAND

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2406.05132)
[![Project Page](https://img.shields.io/badge/Project-Website-5B7493?logo=googlechrome&logoColor=5B7493)](https://3d-grand.github.io/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-%20Hugging%20Face-ED7D31)](https://huggingface.co/spaces/jedyang97/3D-GRAND)


This repository is the implementation of

### 3D-GRAND: Towards Better Grounding and Less Hallucination for 3D-LLMs

- **Authors**: [Jianing (Jed) Yang](https://jedyang.com/)<sup>*,1</sup>
- **Authors**: [Xuweiyi Chen](https://xuweiyichen.github.io/)<sup>*,1</sup>
- **Authors**: [Nikhil Madaan](https://madaan-nikhil.github.io/)
- **Authors**: [Madhavan Iyengar](https://madhavaniyengar.github.io/)<sup>1</sup>
- **Authors**: [Shengyi Qian](https://jasonqsy.github.io/)<sup>1, 2</sup>
- **Authors**: [David Fouhey](https://web.eecs.umich.edu/~fouhey/)<sup>2</sup>
- **Authors**: [Joyce Y. Chai](https://web.eecs.umich.edu/~chaijy/)<sup>1</sup>

**Affiliation**: <sup>1</sup>University of Michigan, <sup>2</sup>New York University

<sup>*</sup>*Equal contribution*

### [Project page](https://3d-grand.github.io/) | [Paper](https://arxiv.org/abs/2406.05132) | [Demo](https://huggingface.co/spaces/jedyang97/3D-GRAND)
## Updates🔥 

- Our demo code about 3D-GRAND is released and you can checkout our [paper](https://arxiv.org/abs/2406.05132) as well!

## Overview 📖

The integration of language and 3D perception is crucial for developing embodied agents and robots that comprehend and interact with the physical world. While large language models (LLMs) have demonstrated impressive language understanding and generation capabilities, their adaptation to 3D environments (3D-LLMs) remains in its early stages. A primary challenge is the absence of large-scale datasets that provide dense grounding between language and 3D scenes. In this paper, we introduce 3D-GRAND, a pioneering large-scale dataset comprising 40,087 household scenes paired with 6.2 million densely-grounded scene-language instructions. Our results show that instruction tuning with 3D-GRAND significantly reduces hallucinations and enhances the grounding capabilities of 3D-LLMs compared to models trained without dense grounding. As part of our contributions, we propose a comprehensive benchmark 3D-POPE to systematically evaluate hallucination in 3D-LLMs, enabling fair comparisons among future models. Our experiments underscore a scaling effect between dataset size and 3D-LLM performance, emphasizing the critical role of large-scale 3D-text datasets in advancing embodied AI research. Through 3D-GRAND and 3D-POPE, we aim to equip the embodied AI community with essential resources and insights, setting the stage for more reliable and better-grounded 3D-LLMs.

In this repository, we release demo code for a model trained with 3D-GRAND.

## Quick Start🔨

### 1. Clone Repo

```
git clone https://github.com/3d-grand/3d_grand_demo.git
cd 3d_grand_demo
```

### 2. Prepare Environment

```
conda create -n 3d_grand_hf python=3.10 -y
conda activate 3d_grand_hf
pip install -r demo/requirements.txt
pip install spaces
```

### 3. Download Checkpoints
Quickstart guide
```
git lfs install
git clone https://huggingface.co/spaces/jedyang97/3D-GRAND
```

### 🤗 Gradio Demo

We provide a Gradio Demo to demonstrate our method with UI.

```
gradio 3d-grand-demo.py
```
Alternatively, you can try the online demo hosted on Hugging Face: [[demo link]](https://huggingface.co/).

## Citation :fountain_pen: 

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
   @misc{3d_grand,
       title={3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination},
       author={Jianing Yang and Xuweiyi Chen and Nikhil Madaan and Madhavan Iyengar and Shengyi Qian and David F. Fouhey and Joyce Chai},
       year={2024},
       eprint={2406.05132},
       archivePrefix={arXiv},
       primaryClass={cs.CV}
   }
   ```
