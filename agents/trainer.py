class Trainer:
    def train(self, env_fn, num_steps, save_dir):
        raise NotImplementedError()

    def evaluate(self, num_episodes, eval_trainer=None):
        raise NotImplementedError()

    def calculate_eigenvalues(self, num_steps):
        raise NotImplementedError()
