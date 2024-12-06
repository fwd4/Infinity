# Infinity $\infty$: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis

<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2412.04431-b31b1b.svg)](https://arxiv.org/abs/2412.04431)&nbsp;

</div>
<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/abs/2412.04431">Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis</a>
</p>


<p align="center">
<img src="assets/show_images.jpg" width=95%>
<p>

## 🔥 Updates!!
* Dec 5, 2024: 🤗 Paper release

## 📑 Open-Source Plan

- Infinity-2B (Text-to-Image Model)
  - [ ] Web Demo 
  - [ ] Inference 
  - [ ] Checkpoints


## 📖 Introduction
We present Infinity, a Bitwise Visual AutoRegressive Modeling capable of generating high-resolution, photorealistic images following language instruction.  Infinity refactors visual autoregressive model under a bitwise token prediction framework with an infinite-vocabulary classifier and bitwise self-correction mechanism. By theoretically expanding the tokenizer vocabulary size to infinity in Transformer, our method significantly unleashes powerful scaling capabilities to infinity compared to vanilla VAR. Extensive experiments indicate Infinity outperforms AutoRegressive Text-to-Image models by large margins, matches or exceeds leading diffusion models. Without extra optimization, Infinity generates a 1024 $\times$ 1024 image in 0.8s, 2.6 $\times$ faster than SD3-Medium, making it the fastest Text-to-Image model. Models and codes are released to promote further exploration of Infinity for visual generation. 



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
