from Rainbow.agent import Agent
from Rainbow.main import main, get_parser

class RainbowTrainer:
    def __init__(self, atari_env_name, learning_starts=int(20e3)):
        self.atari_env_name = atari_env_name
        self.learning_starts = learning_starts

    def train(self, num_steps, save_dir, save_freq=1000):
        parser = get_parser()
        self.args = parser.parse_args([
            f"--game={self.atari_env_name}",
            f"--T-max={num_steps}",
            f"--results-dir={save_dir}",
            f"--learn-start={self.learning_starts}",
            f"--checkpoint-interval={save_freq}",
        ])
        saved_files = main(self.args)
        return saved_files
