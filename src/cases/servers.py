"""Implement server code. This will be short, if the server is honest, but a lot can happen for the malicious variants."""

import torch
import numpy as np
from scipy import stats
import copy

from .data import construct_dataloader
import logging

log = logging.getLogger(__name__)


def construct_server(
    model, loss_fn, cfg_case, setup=dict(device=torch.device("cpu"), dtype=torch.float), external_dataloader=None
):
    """Interface function."""
    if external_dataloader is None and cfg_case.server.has_external_data:
        user_split = cfg_case.data.examples_from_split
        cfg_case.data.examples_from_split = "training" if "validation" in user_split else "validation"
        dataloader = construct_dataloader(cfg_case.data, cfg_case.impl, user_idx=None, return_full_dataset=True)
        cfg_case.data.examples_from_split = user_split
    else:
        dataloader = external_dataloader
    if cfg_case.server.name == "honest_but_curious":
        server = HonestServer(model, loss_fn, cfg_case, setup, external_dataloader=dataloader)
    else:
        raise ValueError(f"Invalid server type {cfg_case.server} given.")
    return server


class HonestServer:
    """Implement an honest server protocol.

    This class loads and selects the initial model and then sends this model to the (simulated) user.
    If multiple queries are possible, then these have to loop externally over muliple rounds via .run_protocol

    Central output: self.distribute_payload -> Dict[parameters=parameters, buffers=buffers, metadata=DataHyperparams]
    """

    THREAT = "Honest-but-curious"

    def __init__(
        self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device("cpu")), external_dataloader=None
    ):
        """Inialize the server settings."""
        self.model = model
        self.model.eval()

        self.loss = loss
        self.setup = setup

        self.num_queries = cfg_case.server.num_queries

        # Data configuration has to be shared across all parties to keep preprocessing consistent:
        self.cfg_data = cfg_case.data
        self.cfg_server = cfg_case.server

        self.external_dataloader = external_dataloader

        self.secrets = dict()  # Should be nothing in here

    def __repr__(self):
        return f"""Server (of type {self.__class__.__name__}) with settings:
    Threat model: {self.THREAT}
    Number of planned queries: {self.num_queries}
    Has external/public data: {self.cfg_server.has_external_data}

    Model:
        model specification: {str(self.model.name)}
        model state: {self.cfg_server.model_state}
        {f'public buffers: {self.cfg_server.provide_public_buffers}' if len(list(self.model.buffers())) > 0 else ''}

    Secrets: {self.secrets}
    """

    def reconfigure_model(self, model_state, query_id=0):
        """Reinitialize, continue training or otherwise modify model parameters in a benign way."""
        self.model.cpu()  # References might have been used on GPU later on. Return to normal first.
        for name, module in self.model.named_modules():
            if model_state == "untrained":
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            elif model_state == "trained":
                pass  # model was already loaded as pretrained model
            elif model_state == "linearized":
                with torch.no_grad():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        module.weight.data = module.running_var.data.clone()
                        module.bias.data = module.running_mean.data.clone() + 10
                    if isinstance(module, torch.nn.Conv2d) and hasattr(module, "bias"):
                        module.bias.data += 10
            elif model_state == "orthogonal":
                # reinit model with orthogonal parameters:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
                if "conv" in name or "linear" in name:
                    torch.nn.init.orthogonal_(module.weight, gain=1)
            elif model_state == "unchanged":
                # Disregard potential to update this model
                pass

    def reset_model(self):
        pass

    def distribute_payload(self, query_id=0):
        """Server payload to send to users. These are only references to simplfiy the simulation."""

        self.reconfigure_model(self.cfg_server.model_state, query_id)
        honest_model_parameters = [p for p in self.model.parameters()]  # do not send only the generators
        if self.cfg_server.provide_public_buffers:
            honest_model_buffers = [b for b in self.model.buffers()]
        else:
            honest_model_buffers = None
        return dict(parameters=honest_model_parameters, buffers=honest_model_buffers, metadata=self.cfg_data)

    def vet_model(self, model):
        """This server is honest."""
        model = self.model  # Re-reference this everywhere
        return self.model

    def queries(self):
        return range(self.num_queries)

    def run_protocol(self, user):
        """Helper function to simulate multiple queries given a user object."""
        # Simulate a simple FL protocol
        shared_user_data = []
        payloads = []
        for query_id in self.queries():
            server_payload = self.distribute_payload(query_id)  # A malicious server can return something "fun" here
            shared_data_per_round, true_user_data = user.compute_local_updates(server_payload)
            # true_data can only be used for analysis
            payloads += [server_payload]
            shared_user_data += [shared_data_per_round]
        return shared_user_data, payloads, true_user_data


