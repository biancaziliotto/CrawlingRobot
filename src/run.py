import hydra
from omegaconf import DictConfig

from agent import Agent


@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="config",
)
def main(cfg: DictConfig):

    agent = Agent(cfg)


if __name__ == "__main__":
    main()
