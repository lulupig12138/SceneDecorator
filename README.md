<div align="center">
<h1>
SceneDecorator: Towards Scene-Oriented Story Generation with Scene Planning and Scene Consistency
</h1>

<div>
    <a href='' target='_blank' style='text-decoration:none'>Quanjian Song<sup>1,*</sup></a>, &ensp;
    <a href='' target='_blank' style='text-decoration:none'>Donghao Zhou<sup>2,*†</sup></a>, &ensp;
    <a href='' target='_blank' style='text-decoration:none'>Jingyu Lin<sup>1,*†</sup></a>, &ensp;
    <a href='' target='_blank' style='text-decoration:none'>Fei Shen<sup>3</sup></a>, &ensp;
    <br>
    <a href='' target='_blank' style='text-decoration:none'>Jiaze Wang<sup>2</sup></a>, &ensp;
    <a href='' target='_blank' style='text-decoration:none'>Xiaowei Hu<sup>4,‡</sup></a>, &ensp;
    <a href='' target='_blank' style='text-decoration:none'>Cunjian Chen<sup>1,‡</sup></a>, &ensp;
    <a href='' target='_blank' style='text-decoration:none'>Pheng-Ann Heng<sup>2</sup></a>
</div>

<div>
    <sup>1</sup>Monash University,  &ensp;
    <sup>2</sup>The Chinese University of Hong Kong
    <br>
    <sup>3</sup>National University of Singapore  &ensp;
    <sup>4</sup>South China University of Technology
    <br>
    <sub>
        <sup>*</sup>Equal contribution.   &ensp;
        <sup>†</sup>Project lead.   &ensp;
        <sup>‡</sup>Corresponding authors.
    </sub>
</div>

<sub></sub>

<p align="center">
    <span>
        <a href="https://arxiv.org/pdf/2510.22994" target="_blank"> 
        <img src='https://img.shields.io/badge/arXiv%20202510.22994-SceneDecorator-red' alt='Paper PDF'></a> &emsp;  &emsp; 
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
@article{song2025scenedecorator,
  title={SceneDecorator: Towards Scene-Oriented Story Generation with Scene Planning and Scene Consistency},
  author={Song, Quanjian and Zhou, Donghao and Lin, Jingyu and Shen, Fei and Wang, Jiaze and Hu, Xiaowei and Chen, Cunjian and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:2510.22994},
  year={2025}
}
```
