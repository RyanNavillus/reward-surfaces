class Trainer:
    def train(self, env_fn, num_steps, save_dir):
        raise NotImplementedError()

    def evaluate(self, num_episodes, num_steps, eval_trainer=None):
        raise NotImplementedError()

    def calculate_eigenvalues(self, num_steps):
        raise NotImplementedError()

    def load_weights(self, fname):
        raise NotImplementedError()

    def save_weights(self, fname):
        raise NotImplementedError()

    def set_weights(self, params):
        raise NotImplementedError()

    def get_weights(self):
        raise NotImplementedError()
