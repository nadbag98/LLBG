"""Implementation for iterative gradient inversion attacks.

This covers optimization-based reconstruction attacks as in Wang et al. "Beyond Infer-
ring Class Representatives: User-Level Privacy Leakage From Federated Learning."
and convers subsequent developments such as
* Zhu et al., "Deep Leakage from gradients",
* Geiping et al., "Inverting Gradients - How easy is it to break privacy in FL"
* ?
"""

import torch
import time

from .optimization_based_attack import OptimizationBasedAttacker
from .auxiliaries.regularizers import regularizer_lookup, TotalVariation
from .auxiliaries.objectives import Euclidean, CosineSimilarity, objective_lookup
from .auxiliaries.augmentations import augmentation_lookup

import logging

log = logging.getLogger(__name__)


class IterativeOptimizationAttacker(OptimizationBasedAttacker):
    """Implements an iterative optimization-based attacks."""

    # def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
    #     super().__init__(model, loss_fn, cfg_attack, setup)
    #     objective_fn = objective_lookup.get(self.cfg.objective.type)
    #     if objective_fn is None:
    #         raise ValueError(f"Unknown objective type {self.cfg.objective.type} given.")
    #     else:
    #         self.objective = objective_fn(**self.cfg.objective)
    #     self.regularizers = []
    #     try:
    #         for key in self.cfg.regularization.keys():
    #             if self.cfg.regularization[key].scale > 0:
    #                 self.regularizers += [regularizer_lookup[key](self.setup, **self.cfg.regularization[key])]
    #     except AttributeError:
    #         pass  # No regularizers selected.
    #
    #     try:
    #         self.augmentations = []
    #         for key in self.cfg.augmentations.keys():
    #             self.augmentations += [augmentation_lookup[key](**self.cfg.augmentations[key])]
    #         self.augmentations = torch.nn.Sequential(*self.augmentations).to(**setup)
    #     except AttributeError:
    #         self.augmentations = torch.nn.Sequential()  # No augmentations selected.

    # def __repr__(self):
    #     n = "\n"
    #     return f"""Attacker (of type {self.__class__.__name__}) with settings:
    # Hyperparameter Template: {self.cfg.type}
    #
    # Objective: {repr(self.objective)}
    # Regularizers: {(n + ' '*18).join([repr(r) for r in self.regularizers])}
    # Augmentations: {(n + ' '*18).join([repr(r) for r in self.augmentations])}
    #
    # Optimization Setup:
    #     {(n + ' ' * 8).join([f'{key}: {val}' for key, val in self.cfg.optim.items()])}
    #     """

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False):
        # Initialize stats module for later usage:
        # set all labels to 0, so prepare attack func doesn't reconstruct them
        num_points = shared_data[0]["metadata"]["num_data_points"]
        shared_data[0]["metadata"]["labels"] = torch.zeros(num_points)
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        # Main reconstruction loop starts here:
        scores = torch.zeros(self.cfg.restarts.num_trials)
        candidate_solutions, candidate_labels = [], []
        try:
            for trial in range(self.cfg.restarts.num_trials):
                rec_data, rec_labels = self._run_trial(rec_models, shared_data, labels, stats, trial, initial_data,
                                                       dryrun)
                candidate_solutions.append(rec_data)
                candidate_labels.append(rec_labels)
                scores[trial] = self._score_trial(rec_data, rec_labels, rec_models, shared_data)
        except KeyboardInterrupt:
            print("Trial procedure manually interruped.")
            pass
        optimal_solution, opt_ind = self._select_optimal_reconstruction(candidate_solutions, scores, stats)
        reconstructed_data = dict(data=optimal_solution, labels=candidate_labels[opt_ind])
        if server_payload[0]["metadata"].modality == "text":
            reconstructed_data = self._postprocess_text_data(reconstructed_data)
        if "ClassAttack" in server_secrets:
            # Only a subset of images was actually reconstructed:
            true_num_data = server_secrets["ClassAttack"]["true_num_data"]
            reconstructed_data["data"] = torch.zeros([true_num_data, *self.data_shape], **self.setup)
            reconstructed_data["data"][server_secrets["ClassAttack"]["target_indx"]] = optimal_solution
            reconstructed_data["labels"] = server_secrets["ClassAttack"]["all_labels"]
        return reconstructed_data, stats

    def _run_trial(self, rec_model, shared_data, labels, stats, trial, initial_data=None, dryrun=False):
        """Run a single reconstruction trial."""
        # Initialize losses:
        # TODO: initialize without all labels
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)

        num_data_points = shared_data[0]["metadata"]["num_data_points"]
        last_bias_grad = shared_data[0]["gradients"][-1]
        log.info(f"Bias gradient before augmentation: {last_bias_grad}")
        rec_labels = (last_bias_grad < 0).nonzero().view(-1)

        try:
            # this is the iterative part
            while torch.numel(rec_labels) <= num_data_points:
                log.info(f"Recovered labels {sorted(rec_labels.tolist())} through iterative strategy.")
                # we init the optimization process each time we reconstruct more labels
                candidate = self._initialize_data([torch.numel(rec_labels), *self.data_shape])
                best_candidate = candidate.detach().clone()
                minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)

                self.objective.initialize(self.loss_fn, self.cfg.impl)
                optimizer, scheduler = self._init_optimizer([candidate])
                current_wallclock = time.time()

                # we have a different max iterations for the last iteration, where we reconstruct data
                curr_iterations_num = self.cfg.optim.max_iterations if torch.numel(rec_labels) == num_data_points else self.cfg.optim.max_iterative_iterations
                for iteration in range(curr_iterations_num):
                    closure = self._compute_objective(candidate, rec_labels, rec_model, optimizer, shared_data,
                                                      iteration)
                    objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                    scheduler.step()

                    with torch.no_grad():
                        # Project into image space
                        if self.cfg.optim.boxed:
                            candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds),
                                                       -self.dm / self.ds)
                        if objective_value < minimal_value_so_far:
                            minimal_value_so_far = objective_value.detach()
                            best_candidate = candidate.detach().clone()

                    if iteration + 1 == curr_iterations_num or iteration % self.cfg.optim.callback == 0:
                        timestamp = time.time()
                        log.info(
                            f"| It: {iteration + 1} for {torch.numel(rec_labels)}/{num_data_points} points "
                            f"| Rec. loss: {objective_value.item():2.4f} | "
                            f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                        )
                        current_wallclock = timestamp

                    if not torch.isfinite(objective_value):
                        log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                        break

                    stats[f"Trial_{trial}_Val"].append(objective_value.item())

                    if dryrun:
                        break

                # if we already reconstructed all labels and data, we exit reconstruction loop and return candidate
                if torch.numel(rec_labels) == num_data_points:
                    break
                else:
                    # if we don't have all the labels, we reconstruct more and then reconstruct data again
                    rec_labels = self._get_additional_labels(rec_model, best_candidate, rec_labels,
                                                             last_bias_grad, num_data_points)

        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass

        return best_candidate.detach(), rec_labels

    def _get_additional_labels(self, models, rec_data, rec_labels, last_bias_grad, num_data_points):
        # TODO: expand to support multiple models (in rec_model)
        attacked_model = models[0]
        best_cand_out = attacked_model(rec_data)
        best_cand_loss = self.loss_fn(best_cand_out, rec_labels)
        all_params = [param for param in attacked_model.parameters()]
        new_bias_grad = torch.autograd.grad(best_cand_loss, all_params[-1])
        # TODO: test with this normalization
        # we subtract the relative part of the gradients from the original
        corrected_bias_grad = last_bias_grad - \
                              (torch.numel(rec_labels) / num_data_points) * new_bias_grad[0]
        #log.info(f"Corrected bias gradient with {torch.numel(rec_labels)} known labels: {corrected_bias_grad}")

        # if corrected bias gradient has negative labels - we take them all
        # TODO: take most negative first, in case we got too many labels here
        new_labels = (corrected_bias_grad < 0).nonzero().view(-1)
        if torch.numel(new_labels) == 0:
            # if all indices of corrected gradient are positive - take argmin
            new_labels = torch.argmin(corrected_bias_grad)
        # try to fill reconstructed labels tensor, until reached num_data_points
        return torch.cat((rec_labels, new_labels[:num_data_points - torch.numel(rec_labels)]))

    # def _compute_objective(self, candidate, labels, rec_model, optimizer, shared_data, iteration):
    #     def closure():
    #         optimizer.zero_grad()
    #
    #         if self.cfg.differentiable_augmentations:
    #             candidate_augmented = self.augmentations(candidate)
    #         else:
    #             candidate_augmented = candidate
    #             candidate_augmented.data = self.augmentations(candidate.data)
    #
    #         total_objective = 0
    #         total_task_loss = 0
    #         for model, data in zip(rec_model, shared_data):
    #             objective, task_loss = self.objective(model, data["gradients"], candidate_augmented, labels)
    #             total_objective += objective
    #             total_task_loss += task_loss
    #         for regularizer in self.regularizers:
    #             total_objective += regularizer(candidate_augmented)
    #
    #         if total_objective.requires_grad:
    #             total_objective.backward(inputs=candidate, create_graph=False)
    #         with torch.no_grad():
    #             if self.cfg.optim.langevin_noise > 0:
    #                 step_size = optimizer.param_groups[0]["lr"]
    #                 noise_map = torch.randn_like(candidate.grad)
    #                 candidate.grad += self.cfg.optim.langevin_noise * step_size * noise_map
    #             if self.cfg.optim.grad_clip is not None:
    #                 grad_norm = candidate.grad.norm()
    #                 if grad_norm > self.cfg.optim.grad_clip:
    #                     candidate.grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
    #             if self.cfg.optim.signed is not None:
    #                 if self.cfg.optim.signed == "soft":
    #                     scaling_factor = (
    #                         1 - iteration / self.cfg.optim.max_iterations
    #                     )  # just a simple linear rule for now
    #                     candidate.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
    #                 elif self.cfg.optim.signed == "hard":
    #                     candidate.grad.sign_()
    #                 else:
    #                     pass
    #
    #         self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
    #         return total_objective
    #
    #     return closure

    # def _score_trial(self, candidate, labels, rec_model, shared_data):
    #     """Score candidate solutions based on some criterion."""
    #
    #     if self.cfg.restarts.scoring in ["euclidean", "cosine-similarity"]:
    #         objective = Euclidean() if self.cfg.restarts.scoring == "euclidean" else CosineSimilarity()
    #         objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])
    #         score = 0
    #         for model, data in zip(rec_model, shared_data):
    #             score += objective(model, data["gradients"], candidate, labels)[0]
    #     elif self.cfg.restarts.scoring in ["TV", "total-variation"]:
    #         score = TotalVariation(scale=1.0)(candidate)
    #     else:
    #         raise ValueError(f"Scoring mechanism {self.cfg.scoring} not implemented.")
    #     return score if score.isfinite() else float("inf")

    def _select_optimal_reconstruction(self, candidate_solutions, scores, stats):
        """Choose one of the candidate solutions based on their scores (for now).

        More complicated combinations are possible in the future."""
        optimal_val, optimal_index = torch.min(scores, dim=0)
        optimal_solution = candidate_solutions[optimal_index]
        stats["opt_value"] = optimal_val.item()
        if optimal_val.isfinite():
            log.info(f"Optimal candidate solution with rec. loss {optimal_val.item():2.4f} selected.")
            return optimal_solution, optimal_index
        else:
            log.info("No valid reconstruction could be found.")
            return torch.zeros_like(optimal_solution), 0
