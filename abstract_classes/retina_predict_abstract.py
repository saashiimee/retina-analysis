class AbstractPredict(object):

    def __init__(self, config):
        self.config = config

    def load_model(self, name):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError
