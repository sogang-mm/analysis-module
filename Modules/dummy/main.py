class Dummy:
    def __init__(self):
        # TODO
        #   - initialize and load model here
        self.model = None
        self.result = None

    def inference_by_path(self, image_path):
        result = []
        # TODO
        #   - Inference using image path
        import time
        time.sleep(5)
        result = [[(0, 0, 0, 0), {'TEST': 0.95, 'DEBUG': 0.05}], [(100, 100, 100, 100), {'TEST': 0.95, 'DEBUG': 0.05}]]
        self.result = result

        return self.result
