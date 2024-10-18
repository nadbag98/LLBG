# Harmful Bias: A General Label-Leakage Attack on Federated Learning from Bias Gradients

This repository is based on the Breaching repository created by Geiping et al., that can be found [here](https://github.com/JonasGeiping/breaching).
It contains the code for the Label Leakage from Bias Gradients (LLBG) attack (link to paper [here](https://mahmoods01.github.io/files/aisec24-label-leakage.pdf)), and only files and code from the original repo to experiment with this attack.

Also included are the baselines used - random label reconstruction attack, and different variants of the Label Leakage from Gradients (LLG) attack suggested by Wainakh et al.
The only data reconstruction attack that appears in this repository is the "Inverting Gradients" attack suggested by Geiping et al. (link to paper [here](https://proceedings.neurips.cc/paper/2020/file/c4ede56bbd98819ae6112b20ac6bf145-Paper.pdf)), as it is the one we experiment with in our paper.

### Setup
The environment for the code may be set up by running:
```
conda env create -f environment.yml
conda activate LLBG
```

### Datasets
Our experiments are run on 2 image datasets:

1. `ImageNet` - you will need to download the *ImageNet ILSVRC2012* dataset **manually**. However, almost all attacks require only the small validation set. 

2. `CIFAR100` - doesn't require manual download.

The default locations for the data is `~/data/imagenet` and `~/data/`, respectively, but this can be changed by adding `case.data.path=path` to the command line when running experiments.

In order to reproduce the experiment results from our paper, 
below are the specific commands needed to recreate each table.
The full experiment results will appear in the "tables" 
directory.

### Experiment reproduction

Table #1, Untrained `CIFAR100` models:
```
python benchmark_breaches.py --multirun case.model=MLP,convnetsmall,VGG19,ResNet32-10 attack.label_strategy=LLG,EBI,LLBG
```

Table #2, Untrained `ImageNet` models:
```
python benchmark_breaches.py --multirun case=7_large_batch_imagenet case.model=VGG19,resnet50,efficientnet_b0,mnasnet1_0,shufflenet_v2_x1_0 attack.label_strategy=LLG,EBI,LLBG
```

Table #3, Untrained `VGG19` on different batch sizes of `CIFAR100` data:
```
python benchmark_breaches.py --multirun case.model=VGG19 case.data.batch_size=128,256,512,1024 attack.label_strategy=LLG,EBI,LLBG
```

Table #4, Untrained `CIFAR100` models with uniform label distribution:
```
python benchmark_breaches.py --multirun case.model=MLP,convnetsmall,VGG19,ResNet32-10 case.data.batch_size=100 case.data.partition=balanced attack.label_strategy=LLG,EBI,LLBG
```

Table #5, Trained `ImageNet` models:
```
python benchmark_breaches.py --multirun case=7_large_batch_imagenet case.model=VGG19,resnet50,efficientnet_b0,mnasnet1_0,shufflenet_v2_x1_0 case.server.pretrained=true attack.use_aux_data=true attack.label_strategy=LLG,EBI,LLBG
```

Table #6, `LLBG_gamma` against trained ImageNet models:
```
python benchmark_breaches.py --multirun case=7_large_batch_imagenet case.model=VGG19,resnet50,efficientnet_b0,mnasnet1_0,shufflenet_v2_x1_0 case.server.pretrained=true attack.approx_avg_conf=0.5,0.7,0.9 attack.label_strategy=LLBG
```

Table #8, Untrained `CIFAR100` `MLP`s with different activations:
```
python benchmark_breaches.py --multirun case.model=MLP_relu,MLP_sigmoid,MLP_leaky,MLP_tanh attack.label_strategy=LLG,EBI,LLBG
```

Table #9, `ViT` models, untrained and trained:
```
python benchmark_breaches.py --multirun case=7_large_batch_imagenet case.model=vit_base attack.label_strategy=LLG,EBI,LLBG
python benchmark_breaches.py --multirun case=7_large_batch_imagenet case.model=vit_base case.server.pretrained=true attack.use_aux_data=true attack.label_strategy=LLG,EBI,LLBG
python benchmark_breaches.py --multirun case=7_large_batch_imagenet case.model=vit_base attack.approx_avg_conf=0.5,0.7,0.9 attack.label_strategy=xLLBG 
```

Table #10, Full reconstruction experiment:
```
python benchmark_breaches.py --multirun case.model=MLP_tanh case.user=local_gradient attack.max_iterations=24000 attack.label_strategy=LLG,LLBG
```

Table #11, Untrained `CIFAR100` models with Differtial Privacy applied, with different clipping values:
```
python benchmark_breaches.py --multirun case.model=VGG19 case.user.local_diff_privacy.gradient_noise=0.01 case.user.local_diff_privacy.per_example_clipping=0.1,0.5,1.0,1.5,0.0 attack.label_strategy=LLG,EBI,LLBG
```

Table #12, Untrained `CIFAR100` models with Differtial Privacy applied, with different noise values:
```
python benchmark_breaches.py --multirun case.model=VGG19 case.user.local_diff_privacy.gradient_noise=0.0,0.01,0.1,0.3,0.5 case.user.local_diff_privacy.per_example_clipping=1.0 attack.label_strategy=LLG,EBI,LLBG
```

Table #13, Untrained `CIFAR100` models with different levels of Gradient Compression:
```
python benchmark_breaches.py --multirun case.model=VGG19 case.user.local_diff_privacy.compression_percent=0.0,0.1,0.4,0.8,0.9,0.99 attack.label_strategy=LLG,EBI,LLBG
```

Table #14, `VGG11` model w/ and w/o bias:
```
python train.py --model_name=VGG11
python train.py --model_name=VGG11n
python benchmark_breaches.py --multirun case.model=VGG11 attack.label_strategy=LLG,EBI,LLBG
python benchmark_breaches.py --multirun case.model=VGG11 case.server.pretrained=true attack.use_aux_data=true attack.label_strategy=LLG,EBI,LLBG
python benchmark_breaches.py --multirun case.model=VGG11_n case.server.pretrained=false,true attack.label_strategy=LLG
```

### License
As mentioned above, we use the Breaching repository by Geiping et al. as a base for this repository.

For the license of our code, refer to `LICENCE.md`.

### Citation
To cite our paper use the following:
@inproceedings{Gat24LLBG,
  author = {Gat, Nadav and Sharif, Mahmood},
  title = {Harmful Bias: {A} General Label-Leakage Attack on Federated Learning from Bias Gradients},
  booktitle = {ACM Workshop on Artificial Intelligence and Security ({AISec})},
  year = {2024}
}