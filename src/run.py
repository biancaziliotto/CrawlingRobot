import time

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from agent import Agent


@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="config",
)
def main(cfg: DictConfig):

    agent = Agent(cfg)

    # wandb.init(
    #     project=cfg.project,
    #     resume=not cfg.resume_str is None,
    #     id=cfg.resume_str,
    #     config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
    # )
    # wandb.run.name = cfg.exp_name
    # wandb.run.save()

    # wandb.log({"config": OmegaConf.to_container(cfg, resolve=True)})

    # agent.train(num_episodes=int(cfg.num_episodes))

    agent.load_model("checkpoints/model_100.ckpt")
    agent.run_policy(100)

    # agent.save_model(
    #     cfg.save_path,
    # )


if __name__ == "__main__":
    main()
