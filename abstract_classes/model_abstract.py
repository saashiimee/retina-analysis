class AbstractModelNN(object):

    def __init__(self, config):
        self.config = config
        self.model = None

    def save(self):
        if self.model is None:
            raise Exception("[Error] No Model Found. Please build the model first before proceeding.")

        print("[INFO] Saving Model...")
        json_string = self.model.to_json()
        open(self.config.hdf5_path + self.config.dataset_name + '_architecture.json', 'w').write(json_string)
        print("[INFO] Model Saved. Your model can be found in the dataset folder.")

    def load(self):
        if self.model is None:
            raise Exception("[Error] No Model Found. Please build the model first before proceeding.")

        print("[INFO] Loading Model Checkpoint ...\n")
        self.model.load_weights(self.config.hdf5_path + self.config.dataset_name + '_best_weights.h5')
        print("[INFO] Model Loaded. Ready to proceed.")

    def build_model(self):
        raise NotImplementedError
