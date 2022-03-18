# Schema-guided paradigm for zero-shot dialogue state tracking (SGP-DST)

Code for the submitted system SGP-DST on DSTC8-Track4 (schema-guided dialog state tracking).

## Installation

The code is based on PyTorch. You can find installation instructions [here](http://pytorch.org/).


## Usage

pytorch_pretrained_bert: model scripts

dataset: path for SGD dataset

eval_scripts: scripts for evaluation

inference_scripts: generate the final prediction results and corresponding json file for evalutaion according to the prediction results of each module.

data_utils_v2.py: scripts for data processing used for model training

data_utils_test.py: scripts for data processing used for inference

train_combine.py: module training, inculding categorical slot value prediction, free-form slot value prediction, requested slot prediction

train_copy_slot.py: in-domain slot transfer training

train_cross_copy.py: cross-domain slot transfer training

train_intents.py: intent prediction training

train_slotNotCare.py: not_care slot value prediction training

run.sh: scripts for run training/inference.


## Cite

If you use the code, please cite the following paper:
**"Fine-tuning bert for schema-guided zero-shot dialogue state tracking"**
Yu-Ping Ruan, Zhen-Hua Ling, Jia-Chen Gu, and Quan Liu. _Proc. AAAI 2020 workshop on DSTC8_

```
@article{ruan2020fine,
  title={Fine-tuning bert for schema-guided zero-shot dialogue state tracking},
  author={Ruan, Yu-Ping and Ling, Zhen-Hua and Gu, Jia-Chen and Liu, Quan},
  journal={Proc. AAAI 2020 workshop on DSTC8},
  year={2020}
}
```
