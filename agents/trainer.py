class Trainer:
    def train(self, env_fn, num_steps, save_dir):
        raise NotImplementedError()

    def evaluate(self, env_fn, load_dir, get_stats, get_convexity):
        raise NotImplementedError()
