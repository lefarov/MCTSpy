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
    q_opt_next, _ = torch.max(q_values_next, dim=-1, keepdim=True)
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
        memory_length: int=50,
        obs_shape: tuple=(8, 8, 13),
        sense_num: int=64,
        move_num: int=64*64,
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

        # Indices for sense and move actions in the batch of data
        self.sense_ind = 0
        self.move_ind = 1

        # Dimensions for observation. We'll need them to cast batch to the right shape
        self.obs_shape = (memory_length, *obs_shape)
        self.sense_num = sense_num
        self.move_num = move_num

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        done: torch.Tensor,
        obs_next: torch.Tensor,
        act_next_mask: torch.Tensor,
        act_opponent: torch.Tensor=None,
        info_dict: dict=None,
    ):
        # Merge action type and batch dimension since torch.Conv3d
        # doesn't support "arbitrary-batched" semantics (WTF?).
        obs = obs.view(-1, *self.obs_shape)
        obs_next = obs_next.view(-1, *self.obs_shape)

        # Feedforward all data in batch through the trainable model.
        _, sense_q, move_q, pred_act_opponent = self.model(obs)
        sense_q = sense_q.view(2, -1, self.sense_num)
        move_q = move_q.view(2, -1, self.move_num)
        pred_act_opponent = pred_act_opponent.view(2, -1, self.move_num)

        # Compute opponent's move prediction loss.
        opponent_loss = self.opponent_loss(
            pred_act_opponent[self.move_ind], act_opponent[self.move_ind].squeeze(-1),
        )

        info_dict["opponent_loss"] = opponent_loss
        loss = self.weights[0] * opponent_loss

        with torch.no_grad():
            # Predict sense and move Q-values for the next time-step
            _, sense_q_next, move_q_next, _ = self.model_target(obs_next)
            sense_q_next = sense_q_next.view(2, -1, self.sense_num)
            move_q_next = move_q_next.view(2, -1, self.move_num)

            sense_q_next_estimate, move_q_next_estimate = None, None
            if self.double_q:
                # Compute Q estimate for the next time-step using trainable model (used in Double-Q)
                _, sense_q_next_estimate, move_q_next_estimate, _ = self.model(obs_next)
                sense_q_next_estimate = sense_q_next_estimate.view(2, -1, self.sense_num)
                move_q_next_estimate = move_q_next_estimate.view(2, -1, self.move_num)

            if self.mask_q:
                # Surprisingly that works for batches :)
                move_q_next[act_next_mask == 0.0] = torch.finfo(move_q_next.dtype).min
                if self.double_q:
                    move_q_next_estimate[act_next_mask == 0.0] = torch.finfo(move_q_next_estimate.dtype).min

        # Compute Q-loss for sense action
        sense_loss = q_loss(
            sense_q[self.sense_ind],
            move_q_next[self.sense_ind],
            act[self.sense_ind].unsqueeze(-1),
            rew[self.sense_ind].unsqueeze(-1),
            done[self.sense_ind].unsqueeze(-1),
            move_q_next_estimate[self.sense_ind] if self.double_q else None,
            self.discount,
        )

        info_dict["sense_loss"] = sense_loss
        loss += self.weights[1] * sense_loss

        # Compute Q-loss for move action
        move_loss = q_loss(
            move_q[self.move_ind],
            sense_q_next[self.move_ind],
            act[self.move_ind].unsqueeze(-1),
            rew[self.move_ind].unsqueeze(-1),
            done[self.move_ind].unsqueeze(-1),
            sense_q_next_estimate[self.move_ind] if self.double_q else None,
            self.discount,
        )

        info_dict["move_loss"] = move_loss
        loss += self.weights[2] * move_loss

        info_dict["total_loss"] = loss

        return loss
