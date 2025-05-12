from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from agent import Agent
from utils.utils import get_last_ckpt


@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="config",
)
def main(cfg: DictConfig):
    agent = Agent(cfg)

    if cfg.eval:
        last_ckpt = get_last_ckpt(Path(cfg.checkpoint_dir))
        if last_ckpt is None:
            print("No checkpoint found in the specified directory. Exiting evaluation.")
            return
        agent.load_model(last_ckpt)
        agent.eval()
        agent.env.load_env_specs(cfg.checkpoint_dir)
        agent.run_policy(100)
        return

    wandb.init(
        project=cfg.project,
        resume=cfg.resume_str is not None,
        id=cfg.resume_str,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
    )
    wandb.run.name = cfg.exp_name
    wandb.run.save()

    wandb.log({"config": OmegaConf.to_container(cfg, resolve=True)})

    agent.train(num_episodes=int(cfg.num_episodes))


if __name__ == "__main__":
    main()
