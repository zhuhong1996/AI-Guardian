# AI-Guardian: Defeating Adversarial Attacks using Backdoors

### ABOUT

This repository contains code implementation of the paper "AI-Guardian: Defeating Adversarial Attacks using Backdoors, at *IEEE Security and Privacy 2023*. The datasets and pre-trained models can be downloaded [here](https://mailsucasaccn-my.sharepoint.com/:f:/g/personal/zhuhong18_mails_ucas_ac_cn/ElszbOldyolDr5EqCdi7R_ABppV2FP8rhLRu6npSs_SC_A?e=pCEoy0).

### DEPENDENCIES

Our code is implemented and tested on TensorFlow. Following packages are used by our code.
- `python==3.6.13`
- `numpy==1.17.0`
- `tensorflow-gpu==1.15.4`
- `opencv==3.4.2`
- `adversarial-robustness-toolbox==1.8.1`

### HOWTO

#### Bijection Backdoor Embedding

Please run the following command.

```bash
python bijection_backdoor_embedding.py
```

This script will load a pre-trained clean model and embed the bijection backdoor into it. 

#### Backdoor Robustness Enhancement

Please run the following command.

```bash
python backdoor_robustness_enhancement.py
```

This script will load the model trained in the previous step and improve the robustness of the bijection backdoor against adversarial examples. 

#### Generating Adversarial Examples

We include a sample script to perform adversarial attacks using the [Carlini and Wagner Attack](https://ieeexplore.ieee.org/abstract/document/7958570). Please run the following command.

```bash
python CW_attack.py
```

#### Testing Performances

Please run the following command.

```bash
python test_performance.py
```

This script will load the DNN model and test its performance both on the clean data and adversarial examples. 






















