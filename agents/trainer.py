class Evaluator:
    def _next_state(self):
        raise NotImplementedError()

    def _next_state_act(self):
        raise NotImplementedError()


class ActionEvaluator(Evaluator):
    def parameters(self):
        raise NotImplementedError()

    def eval_log_prob(self, states, actions):
        raise NotImplementedError()

    @property
    def device(self):
        raise NotImplementedError()

    @property
    def gamma(self):
        raise NotImplementedError()


class Estimator:
    def setup_buffer(self):
        raise NotImplementedError()

    def parameters(self):
        raise NotImplementedError()

    def generate_samples(self, batch_size, max_samples):
        raise NotImplementedError()

    def calulate_grad_from_buffer(self, vec, num_samples):
        raise NotImplementedError()

    @property
    def device(self):
        raise NotImplementedError()

    def cleanup_buffer(self):
        raise NotImplementedError()


class Trainer:
    def train(self, env_fn, num_steps, save_dir):
        raise NotImplementedError()

    def evaluator(self, eval_trainer=None):
        raise NotImplementedError()

    def parameterized_evaluator(self):
        raise NotImplementedError()

    def estimator(self):
        raise NotImplementedError()

    def evaluate(self, num_episodes, num_steps, eval_trainer=None):
        raise NotImplementedError()

    def load_weights(self, fname):
        raise NotImplementedError()

    def save_weights(self, fname):
        raise NotImplementedError()

    def set_weights(self, params):
        raise NotImplementedError()

    def get_weights(self):
        raise NotImplementedError()
