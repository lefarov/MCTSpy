import torch


def q_loss(
    q_values,
    q_values_next,
    act,
    rew,
    done,
    q_values_next_estimate: torch.Tensor=None,
    discount: float=1.0,
) -> torch.Tensor:

    # Select Q values for chosen actions
    q_values_selected = q_values.gather(-1, act)

    # Select optimal values for the next time step
    q_opt_next = torch.max(q_values_next, dim=-1, keepdim=True)
    if q_values_next_estimate is not None:
        # Double Q idea: select the optimum action for the observation at t+1
        # using the trainable model, but compute it's Q value with target one
        q_opt_next = q_values_next.gather(
            -1, torch.argmax(q_values_next_estimate, dim=-1, keepdim=True)
        )

    # Target estimate for Cumulative Discounted Reward
    q_values_target = rew + discount * q_opt_next * (1. - done)

    # Compute TD error
    loss = torch.nn.functional.smooth_l1_loss(q_values_selected, q_values_target)

    return loss


class CombinedLoss(torch.nn.Module):

    def __init__(self,
        model: torch.nn.Module,
        model_target: torch.nn.Module,
        opponent_loss: torch.nn.Module,
        discount: float=1.0,
        double_q: bool=True,
        mask_q: bool=True,
        weights: tuple=(1.0, 1.0, 1.0),
    ):
        super().__init__()

        self.model = model
        self.model_target = model_target
        self.opponent_loss = opponent_loss
        self.discount = discount
        self.double_q = double_q
        self.mask_q = mask_q
        self.weights = weights

    def forward_move(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        done: torch.Tensor,
        obs_next: torch.Tensor,
        mask_next: torch.Tensor,
        act_opponent: torch.Tensor=None,
        info_dict: dict=None,
    ):
        # For the move data we train opponent action prediction head and Q head for move actions
        *_, q_values, pred_act_opponent = self.model(obs)
        opponent_loss = self.opponent_loss(pred_act_opponent, act_opponent.squeeze(-1))

        info_dict["opponent_loss"] = opponent_loss
        loss = self.weights[0] * opponent_loss

        with torch.no_grad():
            q_values_next = self.model_target.move(obs_next)
            # TODO: double check if we need grad-flow through the estimate for Double-Q
            q_values_next_estimate = None
            if self.double_q:
                # Compute Q estimate for the next time-step using trainable model (used in Double-Q)
                q_values_next_estimate = self.model.move(obs_next)

            # Mask Q-values if masking is available
            if self.mask_q:
                # Surprisingly that works for batches :)
                q_values_next[mask_next == 0.0] = torch.finfo(q_values.dtype).min
                if self.double_q:
                    q_values_next_estimate[mask_next == 0.0] = torch.finfo(q_values.dtype).min

        move_loss = q_loss(
            q_values,
            q_values_next,
            act,
            rew,
            done,
            q_values_next_estimate,
            self.discount,
        )

        info_dict["move_loss"] = opponent_loss
        loss += self.weights[1] * move_loss

        return loss

    def forward_sense(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        done: torch.Tensor,
        obs_next: torch.Tensor,
        info_dict: dict=None
    ):
        # For sense actions we train only Q head for sense actions
        q_values = self.model.sense(obs)

        with torch.no_grad():
            q_values_next = self.model_target.sense(obs_next)
            q_values_next_estimate = None
            if self.double_q:
                q_values_next_estimate = self.model.sense(obs_next)

        loss = q_loss(
            q_values,
            q_values_next,
            act,
            rew,
            done,
            q_values_next_estimate,
            self.discount,
        )

        info_dict["sense_loss"] = loss
        return loss * self.weights[-1]
