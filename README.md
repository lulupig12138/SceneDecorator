<div align="center">
<h1>
SceneDecorator: Towards Scene-Oriented Story Generation with Scene Planning and Scene Consistency
</h1>

<p align="center">
    <span>
        <a href="https://arxiv.org/pdf/2510.22994" target="_blank"> 
        <img src='https://img.shields.io/badge/arXiv%202510.22994-SceneDecorator-red' alt='Paper PDF'></a> &emsp;  &emsp; 
    </span>
    <span> 
        <a href='https://lulupig12138.github.io/SceneDecorator' target="_blank">
        <img src='https://img.shields.io/badge/Project_Page-SceneDecorator-green' alt='Project Page'></a>  &emsp;  &emsp;
    </span>
    <span> 
        <a href='https://huggingface.co/papers/2510.22994' target="_blank"> 
        <img src='https://img.shields.io/badge/Hugging_Face-SceneDecorator-yellow' alt='Hugging Face'></a> &emsp;  &emsp;
    </span>
</p>


</div>

## 🎉 News
<pre>
• <strong>2025.10</strong>: 🔥 Our paper, code, and project page are released.
• <strong>2025.09</strong>: 🔥 SceneDecorator has been accepted by NeurIPS 2025.
</pre>


## 🎬 Overview
In this work, we design a training-free framework called <b>SceneDecorator</b>, to address two key challenges in story generation: <i>scene planning</i> and <i>scene consistency</i>. SceneDecorator comprises two core techniques: (i) <i>VLM-Guided Scene Planning.</i> Leveraging a powerful Vision-Language Model (VLM) as a director, it decomposes user-provided themes into local scenes and story sub-prompts in a ''global-to-local'' manner. (ii) <i> Long-Term Scene-Sharing Attention. </i> By simultaneously integrating mask-guided scene injection, scene-sharing attention, and extrapolable noise blending, it maintains subject style diversity and long-term scene consistency in story generation.
Overall framework is shown below:
![Overall Framework](assets/overall_pipeline.png)

## 🔧 Environment
```
git clone https://github.com/lulupig12138/SceneDecorator.git
# Installation with the requirement.txt
conda create -n SceneDecorator python=3.10
conda activate SceneDecorator
pip install -r requirements.txt
# Or installation with environment.yaml
conda env create -f environment.yml
```

## 🚀 Start
```
bash start.sh
```


## 🎓 Bibtex
🤗 If you find this code helpful for your research, please cite:
```
@article{song2026scenedecorator,
  title={Scenedecorator: Towards scene-oriented story generation with scene planning and scene consistency},
  author={Song, Quanjian and Zhou, Donghao and Lin, Jingyu and Shen, Fei and Wang, Jiaze and Hu, Xiaowei and Chen, Cunjian and Heng, Pheng-Ann},
  journal={Advances in Neural Information Processing Systems},
  year={2026}
}
```
