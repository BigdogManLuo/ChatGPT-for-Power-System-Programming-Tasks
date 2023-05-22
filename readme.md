# ChatGPT for Power System Programming Tasks

![](https://img.shields.io/badge/license-MIT-brightgreen)
![](https://img.shields.io/badge/envs-maltab-orange)
![](https://img.shields.io/badge/envs-python-blue)

This repository contains the code implementation for "ChatGPT for Power System Programming Tasks". We aim to provide an innovative solution for power system programming tasks using the state-of-the-art language model, GPT-4, developed by OpenAI.By harnessing the capabilities of ChatGPT, power engineers can streamline their coding tasks, enhance productivity, and unlock new possibilities in power system programming.

## Abstract📰
---
The rapid digitalization of power systems has resulted in a significant surge in coding tasks for power engineers. This repository explores the utilization of ChatGPT, an advanced AI language model, to assist power engineers and researchers in various coding tasks. By leveraging the capabilities of ChatGPT, we demonstrate its efficacy through three case studies that span different coding scenarios.

**Case Study 1️⃣: Daily Unit Commitment**
ChatGPT proves invaluable for routine tasks like daily unit commitment. By directly generating a batch of codes, it enhances efficiency and reduces the repetitive programming and debugging time for power engineers. 

**Case Study 2️⃣: Decentralized Optimization of Multi-Vector Energy Systems**
In complex problems such as decentralized optimization of multi-agent energy systems, ChatGPT streamlines the learning process for power engineers. It facilitates problem formulation and helps in selecting suitable numerical solvers, thereby reducing the learning cost associated with such challenges.

**Case Study 3️⃣: Neural Diving for Unit Commitment**
When faced with new problems that lack readily available solutions, ChatGPT shines as a valuable resource. It aids in organizing technology roadmaps, generating data, and developing models and codes for tackling these novel problems, such as fast unit commitment.

This work corresponds to our paper [Leveraging ChatGPT for Power System Optimization A Novel Approach for Programming Tasks](https://arxiv.org/abs/2305.11202), where we elaborate on our methodology, implementation details, and experimental results.

## Reference
---
If you find this work useful and use it in your research, please consider citing our paper:

```bibtex
@misc{li2023leveraging,
      title={Leveraging ChatGPT for Power System Programming Tasks}, 
      author={Ran Li and Chuanqing Pu and Feilong Fan and Junyi Tao and Yue Xiang},
      year={2023},
      eprint={2305.11202},
      archivePrefix={arXiv},
      primaryClass={cs.HC}
}
```

## Getting Started💾
---
### Prerequisites
- ✅Python 3.7+ <br>
- ✅PyTorch 1.8.0+ <br>
- ✅Matlab R2019b+ <br>


### Installation

Clone this repository and install the requirements:
```bash
git clone https://github.com/yourusername/ChatGPT-for-Power-System-Programming-Tasks.git
cd ChatGPT-for-Power-System-Programming-Tasks
pip install -r requirements.txt
```

### Unit Commitment
To run this code, you can open a command line and type:
```bash
cd <filePath here>
python uc.py
```
### Distributed Optimization by ADMM
<br>

**File Description📝**
- tset.m and admm_solve.m: ADMM implementation using the fmincon function in MATLAB.
- tset_yamip_gurobi and admm_solve_modified_yalmip_gurobi.m: ADMM implementation using second-order cone constraints with the YALMIP and Gurobi environment in MATLAB.
- test_normal_LP.m: Results for centralized optimization, serving as a benchmark for comparing the ADMM algorithm and checking the consistency of results.

**Notice📌**

We believe that the ADMM implementation can serve as a valuable reference and facilitate understanding of distributed optimization techniques in power system programming. However, please be aware that the comprehensive energy system modeling and solving code is not included in this repository.The main reason for this absence lies in the nature of academic research work and intellectual property.

In order to ensure respect for these constraints, and to protect the rights of all those involved in this project, we are not currently able to make the entire codebase publicly available on GitHub. We are more than happy to discuss the concepts, methodologies, and results of our work. Feel free to contact us through the provided channels if you have any questions, suggestions, or if you wish to collaborate with us.

### Neural Diving for Unit Commitment
The following files are included for Neural Diving in Unit Commitment:

1. generate_uc_instances.py: Generates unit commitment instances for training and testing.
2. generate_dataset.py: Generates the dataset required for training the neural diving model.
3. layers.py: Contains the implementation of neural network layers used in the model.
4. models.py: Defines the architecture of the neural diving model.
5. train.py: Trains the neural diving model using the generated dataset.
6. test.py: Performs testing and evaluation of the trained neural diving model.
7. utilities.py: Provides utility functions used in data processing and model training.

The code is used in the following order:
```bash
cd <filePath here>
python generate_uc_instances.py
python generate_dataset.py
python train.py
python test.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgements
We would like to express our gratitude to OpenAI for providing chatGPT for this research.