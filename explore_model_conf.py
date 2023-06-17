import torchvision
import torch
from breaching.cases import construct_dataloader
import hydra
from omegaconf import OmegaConf
from breaching.cases.data.datasets_vision import _get_meanstd, _parse_data_augmentations
import pickle

device = torch.device(f"cuda:0")
#device = torch.device("cpu")

@hydra.main(config_path="breaching/config", config_name="my_cfg", version_base="1.1")
def main_launcher(cfg):
    # print(cfg)
    _default_t = torchvision.transforms.ToTensor()
    model = getattr(torchvision.models, "vgg19")(pretrained=True)
    model.to(device)
    dataset = torchvision.datasets.ImageNet(
            root=cfg.case.data.path, split="val" , transform=_default_t,
        )
    dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))

    if cfg.case.data.mean is None and cfg.data.normalize:
        data_mean, data_std = _get_meanstd(dataset)
        cfg.case.data.mean = data_mean
        cfg.case.data.std = data_std

    transforms = _parse_data_augmentations(cfg.case.data, "val")

    # Apply transformations
    dataset.transform = transforms if transforms is not None else None

    # Save data mean and data std for easy access:
    if cfg.case.data.normalize:
        dataset.mean = cfg.case.data.mean
        dataset.std = cfg.case.data.std
    else:
        dataset.mean = [0]
        dataset.std = [1]

    # print(len(dataset))
    avg_confs = {}
    std_confs = {}
    samples_per_class = 64
    for ind in range(1000):
        indices = [idx for (idx, label) in dataset.lookup.items() if label == ind][:samples_per_class]
        indices = torch.tensor(indices)
        inputs = [dataset[i][0] for i in indices]
        inputs = torch.stack(inputs)
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = outputs.softmax(dim=1).cpu()
        ind_avg_conf = outputs[:, ind].mean()
        ind_std_conf = outputs[:, ind].std()
        avg_confs[ind] = ind_avg_conf.item()
        std_confs[ind] = ind_std_conf.item()
        print(f"{ind}: {ind_avg_conf.item()} +- {ind_std_conf.item()}")
    
    # save avg_confs to a file named "vgg19_imagenet_avg_confs.pkl"
    with open("vgg19_imagenet_avg_confs.pkl", "wb") as f:
        pickle.dump(avg_confs, f)
       
    with open("vgg19_imagenet_std_confs.pkl", "wb") as f:
        pickle.dump(std_confs, f)


    # for i in range(10):
    #     data_ind = torch.randint(0, len(dataset), (1,))
    #     # print(f"data index: {data_ind}")
    #     im, lab = dataset[data_ind]
    #     output = model(im.unsqueeze(0)).detach()
    #     # normalize to distribution using softmax
    #     output = output.softmax(dim=1)
    #     print(f"Confidence in true label {lab}: {output[0][lab].item()}")


if __name__ == "__main__":
    main_launcher()