import torch


def q_loss(
    model: torch.nn.Module,
    model_target: torch.nn.Module,
    obs: torch.Tensor,
    act: torch.Tensor,
    rew: torch.Tensor,
    obs_next: torch.Tensor,
    done: torch.Tensor,
    discount: int=0.9,
    double_q: bool=True,
) -> torch.Tensor:
    # Compute Q values for observation at t
    q_values = model(obs)
    # Select Q values for chosen actions
    q_values_selected = q_values.gather(-1, act)

    with torch.no_grad():
        # Compute Q values for next observation using target model
        q_values_next = model_target(obs_next)

    if double_q:
        # Double Q idea: select the optimum action for the observation at t+1
        # using the trainable model, but compute it's Q value with target one
        q_values_next_estimate = model(obs_next)
        q_opt_next = q_values_next.gather(
            -1, torch.argmax(q_values_next_estimate, dim=-1, keepdim=True)
        )

    else:
        # Select max Q value for the observation at t+1
        q_opt_next = torch.max(q_values_next, dim=-1, keepdim=True)

    # Target estimate for Cumulative Discounted Reward
    q_values_target = rew + discount * q_opt_next * (1. - done)

    # Compute TD error
    loss = torch.nn.functional.smooth_l1_loss(
        q_values_selected, q_values_target
    )

    return loss