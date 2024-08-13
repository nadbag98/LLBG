import torchvision
from torchvision.models import resnet50, googlenet
# from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch
from breaching.cases import construct_dataloader
import hydra
from omegaconf import OmegaConf
from breaching.cases.data.datasets_vision import _build_dataset_vision
from breaching.cases.data.data_preparation import construct_dataloader
import pickle
from breaching.cases.models.model_preparation import _construct_vision_model

device = torch.device(f"cuda:0")
#device = torch.device("cpu")

@hydra.main(config_path="breaching/config", config_name="my_cfg", version_base="1.1")
def main_launcher(cfg):
    _default_t = torchvision.transforms.ToTensor()
    cfg.case.data.partition = "unique-class"
    cfg.case.data.batch_size = 10
    cfg.case.user.num_data_points = 10
    cfg.case.model = "VGG11"
    ds_name = "imagenet"
    
    avg_confs = {}
    std_confs = {}
        
    with torch.no_grad():
        model = _construct_vision_model(cfg.case.model, cfg.case.data, cfg.case.server.pretrained)
        model.eval()
        model.to(device)

        for ind in range(100):
            dataloader = construct_dataloader(cfg.case.data, cfg.case.impl, user_idx=ind, return_full_dataset=False)
            for _, data_block in enumerate(dataloader):
                inputs, labels = data_block["inputs"], data_block["labels"]
                break

            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.softmax(dim=1).cpu()

            ind_avg_conf = outputs[:, ind].mean()
            ind_std_conf = outputs[:, ind].std()
            
            avg_confs[ind] = ind_avg_conf.item()
            std_confs[ind] = ind_std_conf.item()
            print(f"{ind}: {ind_avg_conf.item()} +- {ind_std_conf.item()}")
    
    with open(f"{cfg.case.model}_{ds_name}_avg_confs.pkl", "wb") as f:
        pickle.dump(avg_confs, f)
       
    with open(f"{cfg.case.model}_{ds_name}_std_confs.pkl", "wb") as f:
        pickle.dump(std_confs, f)


if __name__ == "__main__":
    main_launcher()