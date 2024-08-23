# Harmful Bias: A General Label-Leakage Attack on Federated Learning from Bias Gradients

This repository is based on the Breaching repository created by Geiping et al., that can be found [here]([https://github.com/JonasGeiping/breaching]).
It contains the code for the Label Leakage from Bias Gradients (LLBG) attack, and only files and code from the original repo to experiment with this attack.
Also included are the baselines used - random label reconstruction attack, and different variants of the Label Leakage from Gradients (LLG) attack suggested by Wainakh et al.
The only data reconstruction attack that appears in this repository is the "Inverting Gradients" attack suggested by Geiping et al., as it is the one we experiment with in our paper.

## Overview:
This repository implements two main components. A list of modular attacks under `breaching.attacks` and a list of relevant use cases (including server threat model, user setup, model architecture and dataset) under `breaching.cases`.  Only cases explored in our paper are included.

### Usage
You can load any use case by
```
cfg_case = breaching.get_case_config(case="6_large_batch_cifar")
user, server, model, loss = breaching.cases.construct_case(cfg_case)
```
and load the attack by
```
cfg_attack = breaching.get_attack_config(attack="invertinggradients")
attacker = breaching.attacks.prepare_attack(model, loss, cfg_attack)
```

This is a good spot to print out an overview over the loaded threat model and setting, maybe you would want to change some settings?
```
breaching.utils.overview(server, user, attacker)
```

To evaluate the attack, you can then simulate an FL exchange:
```
shared_user_data, payloads, true_user_data = server.run_protocol(user)
```
And then run the attack (which consumes only the user update and the server state):
```
reconstructed_user_data, stats = attacker.reconstruct(payloads, shared_user_data)
```

For more details, have a look at the cmd-line scripts `simulate_breach.py` or `benchmark_breaches.py`.

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
