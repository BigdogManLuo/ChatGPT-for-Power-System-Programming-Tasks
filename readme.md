# ChatGPT for Power System Programming Tasks

![](https://img.shields.io/badge/license-MIT-brightgreen)
![](https://img.shields.io/badge/envs-python-blue)

This repository contains the numerical experiments for "LLMs for Power System Programming Tasks". We aim to provide an innovative solution for power system programming tasks using the large language model. By harnessing the capabilities of LLMs, power engineers can streamline their coding tasks, enhance productivity, and unlock new possibilities in power system programming.

## Overview📰
The rapid digitalization of power systems has resulted in a significant surge in coding tasks for power engineers.

**Case Study 1️⃣: Normal Unit Commitment**
see '01_normal_UC'

**Case Study 2️⃣: Accelerating Unit Commitment**
see '02_accelerating_UC'




This work corresponds to our paper [A framework for leveraging ChatGPT on programming tasks in energy systems
](https://arxiv.org/abs/2305.11202), where we elaborate on our methodology, implementation details, and experimental results.

## Reference
If you find this work useful and use it in your research, please consider citing our paper:

```bibtex
@misc{li2023leveraging,
      title={A framework for leveraging ChatGPT on programming tasks in energy systems}, 
      author={Ran Li and Chuanqing Pu and Feilong Fan and Junyi Tao and Yue Xiang},
      year={2023},
      eprint={2305.11202},
      archivePrefix={arXiv},
      primaryClass={cs.HC}
}
```

## Getting Started💾

### Prerequisites
- ✅Python 3.7+ <br>
- ✅PyTorch 1.8.0+ <br>

### Installation

Clone this repository and install the requirements:
```bash
git clone https://github.com/yourusername/ChatGPT-for-Power-System-Programming-Tasks.git
cd ChatGPT-for-Power-System-Programming-Tasks
pip install -r requirements.txt
```

### Comprehensive Evaluation of Normal UC
take GPT4.0 as an example:
```shell
cd 01_normal_UC
python GPT4.0/x.x.py
...
```

### Accelerating UC
take 'success one' as an example, others are same.
```shell
cd success one
python generate_uc_instances.py
python main.py
```


**Notice📌**


## License
This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgements
Thanks RanLi, Sijie Chen, Feilong Fan, Yue Xiang For their guidance in this work, Thanks [JunyiTao]() for his contribution in this comprehensive evaluation.
