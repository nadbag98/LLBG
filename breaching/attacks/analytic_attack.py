"""Simple analytic attack that works for (dumb) fully connected models."""

import torch

from .base_attack import _BaseAttacker


class AnalyticAttacker(_BaseAttacker):
    """Implements a sanity-check analytic inversion

        Only works for a torch.nn.Sequential model with input-sized FC layers."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device('cpu'))):
        super().__init__(model, loss_fn, cfg_attack, setup)

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        # Main reconstruction: loop starts here:
        inputs_from_queries = []
        for model, user_gradient in zip(rec_models, shared_data['gradients']):
            idx = len(user_gradient) - 1
            for layer in list(model)[::-1]:  # Only for torch.nn.Sequential
                if isinstance(layer, torch.nn.Linear):
                    bias_grad = user_gradient[idx]
                    weight_grad = user_gradient[idx - 1]
                    layer_inputs = self.invert_fc_layer(weight_grad, bias_grad, labels)
                    idx -= 2
                elif isinstance(layer, torch.nn.Flatten):
                    inputs = layer_inputs.reshape(shared_data['num_data_points'], *self.data_shape)
                else:
                    raise ValueError(f'Layer {layer} not supported for this sanity-check attack.')
            inputs_from_queries += [inputs]

        final_reconstruction = torch.stack(inputs_from_queries).mean(dim=0)
        reconstructed_data = dict(data=inputs, labels=labels)

        return reconstructed_data, stats


    def invert_fc_layer(self, weight_grad, bias_grad, image_positions):
        """The basic trick to invert a FC layer."""
        # By the way the labels are exactly at (bias_grad < 0).nonzero() if they are unique
        valid_classes = bias_grad != 0
        intermediates = (weight_grad[valid_classes, :] / bias_grad[valid_classes, None])
        if len(image_positions) == 0:
            reconstruction_data = intermediates
        elif len(image_positions) == 1:
            reconstruction_data = intermediates.mean(dim=0)
        else:
            reconstruction_data = intermediates[image_positions]
        return reconstruction_data


class ImprintAttacker(AnalyticAttacker):
    """Abuse imprint secret for near-perfect attack success."""

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        if 'ImprintBlock' in server_secrets.keys():
            weight_idx = server_secrets['ImprintBlock']['weight_idx']
            bias_idx = server_secrets['ImprintBlock']['bias_idx']
        else:
            raise ValueError(f'No imprint hidden in model {rec_models[0]} according to server.')

        bias_grad = shared_data['gradients'][0][bias_idx]
        weight_grad = shared_data['gradients'][0][weight_idx]

        image_positions = bias_grad.nonzero()
        layer_inputs = self.invert_fc_layer(weight_grad, bias_grad, [])
        inputs = layer_inputs.reshape(layer_inputs.shape[0], *self.data_shape)
        # Fill up with zero if not enough data can be found:
        missing_entries = torch.zeros(len(labels) - inputs.shape[0], *self.data_shape, **self.setup)
        inputs = torch.cat([inputs, missing_entries], dim=0)

        reconstructed_data = dict(data=inputs, labels=labels)
        return reconstructed_data, stats