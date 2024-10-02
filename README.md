# Harmful Bias: A General Label-Leakage Attack on Federated Learning from Bias Gradients

This repository is based on the Breaching repository created by Geiping et al., that can be found [here]([url](https://github.com/JonasGeiping/breaching)).
It contains the code for the Label Leakage from Bias Gradients (LLBG) attack, and only files and code from the original repo to experiment with this attack.
Also included are the baselines used - random label reconstruction attack, and different variants of the Label Leakage from Gradients (LLG) attack suggested by Wainakh et al.
The only data reconstruction attack that appears in this repository is the "Inverting Gradients" attack suggested by Geiping et al., as it is the one we experiment with in our paper.

## Overview:
This repository implements two main components. A list of modular attacks under `breaching.attacks` and a list of relevant use cases (including server threat model, user setup, model architecture and dataset) under `breaching.cases`.  Only cases explored in our paper are included.

### Usage

Table #1, Untrained CIFAR100 models:
```
python benchmark_breaches.py --multirun case.model=MLP,convnetsmall,VGG19,ResNet32-10 attack.label_strategy=LLG,EBI,LLBG
```

Table #2, Untrained ImageNet models:
```
python benchmark_breaches.py --multirun case=7_large_batch_imagenet case.model=VGG19,resnet50,efficientnet_b0,mnasnet1_0,shufflenet_v2_x1_0 attack.label_strategy=LLG,EBI,LLBG
```

Table #3, Untrained VGG19 on different batch sizes of CIFAR100 data:
```
python benchmark_breaches.py --multirun case.model=VGG19 case.data.batch_size=128,256,512,1024 attack.label_strategy=LLG,EBI,LLBG
```

Table #4, Untrained CIFAR100 models with uniform label distribution:
```
python benchmark_breaches.py --multirun case.model=MLP,convnetsmall,VGG19,ResNet32-10 case.data.batch_size=100 case.data.partition=balanced attack.label_strategy=LLG,EBI,LLBG
```

Table #5, Trained ImageNet models:
```
python benchmark_breaches.py --multirun case=7_large_batch_imagenet case.model=VGG19,resnet50,efficientnet_b0,mnasnet1_0,shufflenet_v2_x1_0 case.server.pretrained=true attack.use_aux_data=true attack.label_strategy=LLG,EBI,LLBG
```

Table #6, LLBG_gamma against trained ImageNet models:
```
python benchmark_breaches.py --multirun case=7_large_batch_imagenet case.model=VGG19,resnet50,efficientnet_b0,mnasnet1_0,shufflenet_v2_x1_0 case.server.pretrained=true attack.approx_avg_conf=0.5,0.7,0.9 attack.label_strategy=LLBG
```

Table #8, Untrained CIFAR100 MLPs with different activations:
```
python benchmark_breaches.py --multirun case.model=MLP_relu,MLP_sigmoid,MLP_leaky,MLP_tanh attack.label_strategy=LLG,EBI,LLBG
```

Table #9, ViT models, untrained and trained:
```
python benchmark_breaches.py --multirun case=7_large_batch_imagenet case.model=vit_base attack.label_strategy=LLG,EBI,LLBG
python benchmark_breaches.py --multirun case=7_large_batch_imagenet case.model=vit_base case.server.pretrained=true attack.use_aux_data=true attack.label_strategy=LLG,EBI,LLBG
python benchmark_breaches.py --multirun case=7_large_batch_imagenet case.model=vit_base attack.approx_avg_conf=0.5,0.7,0.9 attack.label_strategy=xLLBG 
```

Table #10, Full reconstruction experiment:
```
python benchmark_breaches.py --multirun case.model=MLP_tanh case.user=local_gradient attack.max_iterations=24000 attack.label_strategy=LLG,LLBG
```

Table #11, Untrained CIFAR100 models with Differtial Privacy applied, with different clipping values:
```
python benchmark_breaches.py --multirun case.model=VGG19 case.user.local_diff_privacy.gradient_noise=0.01 case.user.local_diff_privacy.per_example_clipping=0.1,0.5,1.0,1.5,0.0 attack.label_strategy=LLG,EBI,LLBG
```

Table #12, Untrained CIFAR100 models with Differtial Privacy applied, with different noise values:
```
python benchmark_breaches.py --multirun case.model=VGG19 case.user.local_diff_privacy.gradient_noise=0.0,0.01,0.1,0.3,0.5 case.user.local_diff_privacy.per_example_clipping=1.0 attack.label_strategy=LLG,EBI,LLBG
```

Table #13, Untrained CIFAR100 models with different levels of Gradient Compression:
```
python benchmark_breaches.py --multirun case.model=VGG19 case.user.local_diff_privacy.compression_percent=0.0,0.1,0.4,0.8,0.9,0.99 attack.label_strategy=LLG,EBI,LLBG
```

# TODO: trained VGG11 model

### Datasets
Many examples for vision attacks show `ImageNet` examples. For this to work, you need to download the *ImageNet ILSVRC2012* dataset **manually**. However, almost all attacks require only the small validation set, which can be easily downloaded onto a laptop and do not look for the whole training set. 
`CIFAR10` and `CIFAR100` are also around.
There are a few options to partition the data between users - random, balanced, unique-class (each user only has data from a single class) and imbalanced, the default partition used in our paper and suggested by Wainakh et al.


## Metrics

We implement a range of metrics which can be queried through `breaching.analysis.report`. Several metrics (such as CW-SSIM and R-PSNR) require additional packages to be installed - they will warn about this. For language data we hook into a range of huggingface metrics. Overall though, we note that most of these metrics give only a partial picture of the actual severity of a breach of privacy, and are best handled with care.

## Additional Topics

### Benchmarking
A script to benchmark attacks is included as `benchmark_breaches.py`. This script will iterate over the first valid `num_trials` users, attack each separately and average the resulting metrics. This can be useful for quantitative analysis of these attacks. The default case takes about a day to benchmark on a single GTX2080 GPU for optimization-based attacks, and less than 30 minutes for analytic attacks.
Using the default scripts for benchmarking and cmd-line executes also includes a bunch of convenience based mostly on `hydra`. This entails the creation of separate sub-folders for each experiment in `outputs/`. These folders contain logs, metrics and optionally recovered data for each run. Summary tables are written to `tables/`.

### Options
It is probably best to have a look into `breaching/config` to see all possible options.

### License
As mentioned above, we use the Breaching repository by Geiping et al. as a base for this repository.

For the license of our code, refer to `LICENCE.md`.
