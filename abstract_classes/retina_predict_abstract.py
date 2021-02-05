class AbstractPredict(object):

    def __init__(self, config):
        self.config = config

    def analyze_name(self, path):
        raise NotImplementedError

    def load_model(self, name):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError
