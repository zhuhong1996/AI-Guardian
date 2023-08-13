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


### About the adaptive attack:

Nicholas Carlini propose an adaptive attack against AI-Guardian in https://arxiv.org/pdf/2307.15008.pdf. Here are our comments about it.

1. Some background:

Nicholas and we had a good number of rounds of email threads and an online Zoom discussion about this. From the very beginning, he requested our defense model and source code. Then, we helped verify his attack results, and finally, we talked and discussed how some of the potential defenses presented in our paper could be implemented to defeat his attack. He also brain-stormed how to break the improved defense. We believe this kind of discussion is beneficial to the entire security community and are always open to such discussion.

2. Technical side: How well does Nicholas’ approach break AI-Guardian?

First, we want to point out that Nicholas’s approach needs to access the confidence vector from our defense model to perform the first step, mask recovery. In the real world, however, such confidence vector information is not always available, especially when the model deployers already considered using some defense like AI-Guardian. They typically will just provide the output itself and not expose the confidence vector information to customers due to security concerns. Without such important information, Nicholas’ approach may not work, i.e., the first step, mask recovery, will fail.

Even with the confidence vector information available, we believe the attack proposed by Nicholas did not break the idea of AI-Guardian, but agree that it, indeed, posed a threat against the prototype provided by the AI-Guardian paper. The idea of AI-Guardian is quite simple, using an injected backdoor to defeat adversarial attacks since the former suppresses the latter based on our findings. To demonstrate the idea, in our paper, we chose to implement a prototype using a patch-based backdoor trigger, which is simply a specific pattern attached to the inputs. Such a type of trigger is intuitive, and we believe it is sufficient to demonstrate the idea of AI-Guardian. Nicholas’ approach starts by recovering the mask of the patch-based trigger, which definitely is possible and smart since the "key" space of the mask is limited, thus suffering from a simple brute force attack. That is where the approach begins to break our provided prototype in the paper.

However, we want to point out that there are other types of triggers that are complex and can be used by AI-Guardian defense, e.g., feature space trigger (like image transformation), a trigger injected into middle layers of the model (like manipulating the outputs of the middle layers), etc. They either provide a much larger key space for the mask or even make the mask non-accessible to attackers to explore. Specially, we tried to implement another AI-Guardian prototype by injecting a trigger via manipulating the outputs of the first convolution layer. This new prototype provides a similar performance as the one used in our paper but does not suffer from the mask recovery proposed by Nicholas.

3. Other thoughts:

Nicholas’s work is really interesting, especially with the assistance of LLM. We have seen LLM has been used in a wide array of tasks, but it is the first time to see it assists ML security research in this way, almost totally taking over the implementation work. Meanwhile, we can also see that GPT-4 is not that "intelligent" yet to break a security defense by itself. Right now, it serves as assistance, following human guidance to implement the idea of humans. It is also reported that GPT-4 has been used to summarize and help understand research papers. So it is possible that we will see a research project in the near future, tuning GPT-4 or other kinds of LLMs to understand a security defense, identify vulnerabilities, and implement a proof-of-concept exploit, all by itself in an automated fashion. From a defender’s point of view, however, we would like it to integrate the last step, fixing the vulnerability and testing the fix as well, so we can just relax 



















