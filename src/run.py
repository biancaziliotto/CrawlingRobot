import argparse

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
    parser = argparse.ArgumentParser(description="RL agent")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the agent instead of training",
    )
    args = parser.parse_args()

    agent = Agent(cfg)

    if args.eval:
        agent.load_model(cfg.load_path)
        agent.run_policy(10)
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
