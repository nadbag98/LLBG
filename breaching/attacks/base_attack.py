"""Implementation for base attacker class.

Inherit from this class for a consistent interface with attack cases."""

import torch
from collections import defaultdict
import copy
import pickle
import os

from .auxiliaries.common import optimizer_lookup
from ..cases.models.transformer_dictionary import lookup_grad_indices

import logging

log = logging.getLogger(__name__)
embedding_layer_names = ["encoder.weight", "word_embeddings.weight", "transformer.wte"]


class _BaseAttacker:
    """This is a template class for an attack.

    A basic assumption for this attacker is that user data is fixed over multiple queries.
    """

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu")), model_name=None, ds_name=None):
        self.cfg = cfg_attack
        self.memory_format = torch.channels_last if cfg_attack.impl.mixed_precision else torch.contiguous_format
        self.setup = dict(device=setup["device"], dtype=getattr(torch, cfg_attack.impl.dtype))
        self.model_template = copy.deepcopy(model)
        self.loss_fn = copy.deepcopy(loss_fn)
        self.model_name = model_name.lower()
        self.ds_name = ds_name.lower()
        
        use_aux_data = cfg_attack.use_aux_data
        v_hat = cfg_attack.approx_avg_conf
        self.conf_stats = {i: v_hat for i in range(1000)}
        if use_aux_data:
            conf_folder = cfg_attack.conf_folder
            model_name = model_name.lower()
            ds_name = ds_name.lower()
            conf_filename = f"{conf_folder}{model_name}_{ds_name}_avg_confs.pkl"
            with open(conf_filename, "rb") as f:
                self.conf_stats = pickle.load(f) 

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        """Overwrite this function to implement a new attack."""
        # Implement the attack here
        # The attack should consume the shared_data and server payloads and reconstruct into a dict
        # with key data, labels
        raise NotImplementedError()

        return reconstructed_data, stats

    def __repr__(self):
        raise NotImplementedError()

    def prepare_attack(self, server_payload, shared_data):
        """Basic startup common to many reconstruction methods."""
        stats = defaultdict(list)

        shared_data = shared_data.copy()  # Shallow copy is enough
        server_payload = server_payload.copy()

        # Load preprocessing constants:
        metadata = server_payload[0]["metadata"]
        self.data_shape = metadata.shape
        if hasattr(metadata, "mean"):
            self.dm = torch.as_tensor(metadata.mean, **self.setup)[None, :, None, None]
            self.ds = torch.as_tensor(metadata.std, **self.setup)[None, :, None, None]
        else:
            self.dm, self.ds = torch.tensor(0, **self.setup), torch.tensor(1, **self.setup)

        # Load server_payload into state:
        rec_models = self._construct_models_from_payload_and_buffers(server_payload, shared_data)
        shared_data = self._cast_shared_data(shared_data)
        if metadata.modality == "text":
            rec_models, shared_data = self._prepare_for_text_data(shared_data, rec_models)
        self._rec_models = rec_models
        # Consider label information
        if shared_data[0]["metadata"]["labels"] is None:
            labels = self._recover_label_information(shared_data, server_payload, rec_models)
        else:
            labels = shared_data[0]["metadata"]["labels"].clone()

        # Condition gradients?
        if self.cfg.normalize_gradients:
            shared_data = self._normalize_gradients(shared_data)
        return rec_models, labels, stats

    def _construct_models_from_payload_and_buffers(self, server_payload, shared_data):
        """Construct the model (or multiple) that is sent by the server and include user buffers if any."""

        # Load states into multiple models if necessary
        models = []
        for idx, payload in enumerate(server_payload):

            new_model = copy.deepcopy(self.model_template)
            new_model.to(**self.setup, memory_format=self.memory_format)

            # Load parameters
            parameters = payload["parameters"]
            if shared_data[idx]["buffers"] is not None:
                # User sends buffers. These should be used!
                buffers = shared_data[idx]["buffers"]
                new_model.eval()
            elif payload["buffers"] is not None:
                # The server has public buffers in any case
                buffers = payload["buffers"]
                new_model.eval()
            else:
                # The user sends no buffers and there are no public bufers
                # (i.e. the user in in training mode and does not send updates)
                new_model.train()
                for module in new_model.modules():
                    if hasattr(module, "track_running_stats"):
                        module.reset_parameters()
                        module.track_running_stats = False
                buffers = []

            with torch.no_grad():
                for param, server_state in zip(new_model.parameters(), parameters):
                    param.copy_(server_state.to(**self.setup))
                for buffer, server_state in zip(new_model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))

            if self.cfg.impl.JIT == "script":
                example_inputs = self._initialize_data((1, *self.data_shape))
                new_model = torch.jit.script(new_model, example_inputs=[(example_inputs,)])
            elif self.cfg.impl.JIT == "trace":
                example_inputs = self._initialize_data((1, *self.data_shape))
                new_model = torch.jit.trace(new_model, example_inputs=example_inputs)
            models.append(new_model)
        return models

    def _cast_shared_data(self, shared_data):
        """Cast user data to reconstruction data type."""
        for data in shared_data:
            data["gradients"] = [g.to(dtype=self.setup["dtype"]) for g in data["gradients"]]
            if data["buffers"] is not None:
                data["buffers"] = [b.to(dtype=self.setup["dtype"]) for b in data["buffers"]]
        return shared_data

    def _initialize_data(self, data_shape):
        """Note that data is initialized "inside" the network normalization."""
        init_type = self.cfg.init
        if init_type == "randn":
            candidate = torch.randn(data_shape, **self.setup)
        elif init_type == "randn-trunc":
            candidate = (torch.randn(data_shape, **self.setup) * 0.1).clamp(-0.1, 0.1)
        elif init_type == "rand":
            candidate = (torch.rand(data_shape, **self.setup) * 2) - 1.0
        elif init_type == "zeros":
            candidate = torch.zeros(data_shape, **self.setup)
        # Initializations from Wei et al, "A Framework for Evaluating Gradient Leakage
        #                                  Attacks in Federated Learning"
        elif any(c in init_type for c in ["red", "green", "blue", "dark", "light"]):  # init_types like 'red-true'
            candidate = torch.zeros(data_shape, **self.setup)
            if "light" in init_type:
                candidate = torch.ones(data_shape, **self.setup)
            else:
                nonzero_channel = 0 if "red" in init_type else 1 if "green" in init_type else 2
                candidate[:, nonzero_channel, :, :] = 1
            if "-true" in init_type:
                # Shift to be truly RGB, not just normalized RGB
                candidate = (candidate - self.dm) / self.ds
        elif "patterned" in init_type:  # Look for init_type=rand-patterned-4
            pattern_width = int("".join(filter(str.isdigit, init_type)))
            if "randn" in init_type:
                seed = torch.randn([data_shape[0], 3, pattern_width, pattern_width], **self.setup)
            elif "rand" in init_type:
                seed = (torch.rand([data_shape[0], 3, pattern_width, pattern_width], **self.setup) * 2) - 1
            else:  # default is also randn
                seed = torch.randn([data_shape[0], 3, pattern_width, pattern_width], **self.setup)
            # Shape expansion:
            x_factor, y_factor = (
                torch.as_tensor(data_shape[2] / pattern_width).ceil(),
                torch.as_tensor(data_shape[3] / pattern_width).ceil(),
            )
            candidate = (
                torch.tile(seed, (1, 1, int(x_factor), int(y_factor)))[:, :, : data_shape[2], : data_shape[3]]
                .contiguous()
                .clone()
            )
        elif "wei" in init_type:  # Look for init_type=rand-wei-4
            pattern_width = int("".join(filter(str.isdigit, init_type)))
            if "rand" in init_type:
                seed = (torch.rand([data_shape[0], 3, pattern_width, pattern_width], **self.setup) * 2) - 1
            else:
                seed = torch.randn([data_shape[0], 3, pattern_width, pattern_width], **self.setup)
            # Shape expansion:
            x_factor, y_factor = (
                torch.as_tensor(data_shape[2] / pattern_width).ceil(),
                torch.as_tensor(data_shape[3] / pattern_width).ceil(),
            )
            candidate = (
                torch.tile(seed, (1, 1, int(x_factor), int(y_factor)))[:, :, : data_shape[2], : data_shape[3]]
                .contiguous()
                .clone()
            )
        else:
            raise ValueError(f"Unknown initialization scheme {init_type} given.")

        candidate.to(memory_format=self.memory_format)
        candidate.requires_grad = True
        candidate.grad = torch.zeros_like(candidate)
        return candidate

    def _init_optimizer(self, candidate):
        optimizer, scheduler = optimizer_lookup(
            candidate,
            self.cfg.optim.optimizer,
            self.cfg.optim.step_size,
            scheduler=self.cfg.optim.step_size_decay,
            warmup=self.cfg.optim.warmup,
            max_iterations=self.cfg.optim.max_iterations,
        )
        return optimizer, scheduler

    def _normalize_gradients(self, shared_data, fudge_factor=1e-6):
        """Normalize gradients to have norm of 1. No guarantees that this would be a good idea for FL updates."""
        for data in shared_data:
            grad_norm = torch.stack([g.pow(2).sum() for g in data["gradients"]]).sum().sqrt()
            torch._foreach_div_(data["gradients"], max(grad_norm, fudge_factor))
        return shared_data

    def _recover_label_information(self, user_data, server_payload, rec_models):
        """Recover label information.

        This method runs under the assumption that the last two entries in the gradient vector
        correpond to the weight and bias of the last layer (mapping to num_classes).
        For non-classification tasks this has to be modified.

        The behavior with respect to multiple queries is work in progress and subject of debate.
        """
        num_data_points = user_data[0]["metadata"]["num_data_points"]
        if self.ds_name == "imagenet":
            num_classes = 1000
        else:
            num_classes = 100
        num_queries = len(user_data)

        if self.cfg.label_strategy is None:
            return None

        elif "wainakh" in self.cfg.label_strategy:
            # TODO: verify this!!
            if shared_data["gradients"].shape[1] == 1:
                weight_grad_loc = -2
            else:
                weight_grad_loc = -1
            if self.cfg.label_strategy == "wainakh-simple":
                # As seen in Weinakh et al., "User Label Leakage from Gradients in Federated Learning"
                m_impact = 0
                for query_id, shared_data in enumerate(user_data):
                    g_i = shared_data["gradients"][weight_grad_loc].sum(dim=1)
                    m_query = (
                        torch.where(g_i < 0, g_i, torch.zeros_like(g_i)).sum() * (1 + 1 / num_classes) / num_data_points
                    )
                    s_offset = 0
                    m_impact += m_query / num_queries
            elif self.cfg.label_strategy == "wainakh-whitebox":
                # Augment previous strategy with measurements of label impact for dummy data.
                m_impact = 0
                s_offset = torch.zeros(num_classes, **self.setup)

                print("Starting a white-box search for optimal labels. This will take some time.")
                for query_id, model in enumerate(rec_models):
                    # Estimate m:
                    weight_params = (list(rec_models[0].parameters())[weight_grad_loc],)
                    for class_idx in range(num_classes):
                        fake_data = torch.randn([num_data_points, *self.data_shape], **self.setup)
                        fake_labels = torch.as_tensor([class_idx] * num_data_points, **self.setup)
                        with torch.autocast(self.setup["device"].type, enabled=self.cfg.impl.mixed_precision):
                            loss = self.loss_fn(model(fake_data), fake_labels)
                        (W_cls,) = torch.autograd.grad(loss, weight_params)
                        g_i = W_cls.sum(dim=1)
                        m_impact += g_i.sum() * (1 + 1 / num_classes) / num_data_points / num_classes / num_queries

                    # Estimate s:
                    T = num_classes - 1
                    for class_idx in range(num_classes):
                        fake_data = torch.randn([T, *self.data_shape], **self.setup)
                        fake_labels = torch.arange(num_classes, **self.setup)
                        fake_labels = fake_labels[fake_labels != class_idx]
                        with torch.autocast(self.setup["device"].type, enabled=self.cfg.impl.mixed_precision):
                            loss = self.loss_fn(model(fake_data), fake_labels)
                        (W_cls,) = torch.autograd.grad(loss, (weight_params[0][class_idx],))
                        s_offset[class_idx] += W_cls.sum() / T / num_queries

            else:
                raise ValueError(f"Invalid Wainakh strategy {self.cfg.label_strategy}.")

            # After determining impact and offset, run the actual recovery algorithm
            label_list = []
            g_per_query = [shared_data["gradients"][weight_grad_loc].sum(dim=1) for shared_data in user_data]
            g_i = torch.stack(g_per_query).mean(dim=0)
            # Stage 1:
            negative_indices = torch.nonzero(g_i.squeeze() < 0).squeeze()
            if negative_indices.numel() == 0:
                certain_labels = []
            else:
                sorted_indices = torch.argsort(g_i[negative_indices].squeeze())
                certain_labels = negative_indices[sorted_indices[:min(num_data_points, negative_indices.numel())]].tolist()
            for lab in certain_labels:
                label_list.append(torch.as_tensor(lab, device=self.setup["device"]))

            # Stage 2:
            g_i = g_i - s_offset
            while len(label_list) < num_data_points:
                selected_idx = g_i.argmin()
                label_list.append(torch.as_tensor(selected_idx, device=self.setup["device"]))
                g_i[selected_idx] -= m_impact
            # Finalize labels:
            labels = torch.stack(label_list)

        elif self.cfg.label_strategy == "ebi":  
            # This is slightly modified analytic label recovery in the style of Wainakh
            bias_per_query = [shared_data["gradients"][-1] for shared_data in user_data]
            label_list = []
            # Stage 1
            average_bias = torch.stack(bias_per_query).mean(dim=0)
            valid_classes = (average_bias < 0).nonzero()
            sorted_indices = torch.argsort(average_bias[valid_classes.squeeze()]).view(-1)
            for i in range(min(num_data_points, len(valid_classes))):
                label_list.append(valid_classes[sorted_indices[i]].squeeze())
            m_impact = average_bias[valid_classes].sum() / num_data_points

            average_bias[valid_classes] = average_bias[valid_classes] - m_impact
            # Stage 2
            while len(label_list) < num_data_points:
                selected_idx = average_bias.argmin()
                label_list.append(selected_idx)
                average_bias[selected_idx] -= m_impact
            labels = torch.stack(label_list)

        elif self.cfg.label_strategy == "llbg":  
            # our variant of Wainakh's attack using bias gradient and theoretical value for impact
            bias_per_query = [shared_data["gradients"][-1] for shared_data in user_data]
            label_list = []
            # Stage 1
            average_bias = torch.stack(bias_per_query).mean(dim=0)
            valid_classes = (average_bias < 0).nonzero()
            label_list += [*valid_classes.squeeze(dim=-1)]
            # here is the difference from bias-corrected: impact is independant of gradient
            # this takes care to calculate impact correctly for fedavg
            data_per_batch = num_data_points
            num_steps = 1
            if user_data[0]["metadata"]["local_hyperparams"] is not None:
                # in fedavg, the batch size we need is the mini batch size and num of steps
                data_per_batch = user_data[0]["metadata"]["local_hyperparams"]["data_per_step"]
                num_steps = user_data[0]["metadata"]["local_hyperparams"]["steps"]
            m_impact = - 1 / data_per_batch

            for cls in valid_classes:
                model_conf = self.conf_stats[cls.item()]
                average_bias[cls] -= m_impact * (1 - model_conf)

            # if the model is untrained, we have a positive offset of number of local steps divided by number of classes
            average_bias -= num_steps / num_classes

            # Stage 2
            while len(label_list) < num_data_points:
                selected_idx = average_bias.argmin()
                model_conf = self.conf_stats[selected_idx.item()]
                label_list.append(selected_idx)
                average_bias[selected_idx] -= m_impact * (1 - model_conf)
            labels = torch.stack(label_list)

        elif self.cfg.label_strategy == "random":
            # A random baseline
            labels = torch.randint(0, num_classes, (num_data_points,), device=self.setup["device"])
        else:
            raise ValueError(f"Invalid label recovery strategy {self.cfg.label_strategy} given.")

        # Pad with random labels if too few were produced:
        if len(labels) < num_data_points:
            labels = torch.cat(
                [labels, torch.randint(0, num_classes, (num_data_points - len(labels),), device=self.setup["device"])]
            )

        # Always sort, order does not matter here:
        labels = labels.sort()[0]
        log.info(f"Recovered labels {labels.tolist()} through strategy {self.cfg.label_strategy}.")
        return labels
