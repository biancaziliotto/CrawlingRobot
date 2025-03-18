import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from agent import Agent


@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="config",
)
def main(cfg: DictConfig):

    wandb.init(
        project=cfg.project,
        resume=not cfg.resume_str is None,
        id=cfg.resume_str,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
    )
    wandb.run.name = cfg.exp_name
    wandb.run.save()

    wandb.log({"config": OmegaConf.to_container(cfg, resolve=True)})

    agent = Agent(cfg)


if __name__ == "__main__":
    main()
