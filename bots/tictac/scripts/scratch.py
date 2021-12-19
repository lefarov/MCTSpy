import os
import wandb
import torch


if __name__ == "__main__":
    run = wandb.init(project="recon_tictactoe", entity="not-working-solutions")
    model_dir = os.path.abspath(os.path.join(run.dir, "model_checkpoint"))
    os.makedirs(model_dir)

    trained_model_artifact = wandb.Artifact("full-state-conv", type="model")

    # Imitate training and checkpointing
    net = torch.nn.Linear(64, 64)
    torch.save(net.state_dict(), os.path.join(model_dir, "model_0.pt"))
    torch.save(net.state_dict(), os.path.join(model_dir, "model_10.pt"))

    trained_model_artifact.add_dir(model_dir)
    run.log_artifact(trained_model_artifact)
