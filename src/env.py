import mujoco


class Env:
    """
    Physical system.
    """

    def __init__(self, cfg):
        """
        Initialize physical system.
        """

        self.mj_model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self._create_renderer()

        self.action_dim = self.mj_model.nu
        self.state_dim = self.mj_model.njnt

        print("Environment initialized.")
        print(f"state_dim = {self.state_dim}")
        print(f"action_dim = {self.action_dim}")

        return

    def _create_renderer(self):
        self.renderer = mujoco.Renderer(self.mj_model)
        return

    def render(self):
        """
        Update the rendering scene.
        """
        pass

    def reset(self):
        """
        Initialize episode.
        """
        pass

    def step(self, action):
        """
        Execute action and update state.
        """
        pass

    def compute_reward(self):
        """
        Compute reward.
        """
        pass

    def compute_reset(self):
        """
        Check conditions to truncate episode.
        """
        pass
