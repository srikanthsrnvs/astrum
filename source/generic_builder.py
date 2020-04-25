from image_classifier import ImageClassifier
from object_detector import ObjectDetector

class GenericBuilder:

    def __init__(self, job, log_dir, finished_queue, cv):
        self.log_dir = log_dir
        self.job = job
        self.finished_queue = finished_queue

        if self.job.type == 'image_classification':
            self.network = ImageClassifier(self.job, self.log_dir, self.finished_queue, cv)
        elif self.job.type == 'object_detection':
            self.network = ObjectDetector(self.job, log_dir, self.finished_queue, cv)
        elif self.job.type == 'structured_prediction':
            # TODO: Structured data predictor should be built
            pass
        elif self.job.type == 'structured_classification':
            # TODO: Structured classifier should be built
            pass
        elif self.job.type == 'custom':
            # TODO: Custom built nn should be built
            pass
        else:
            raise Exception()

    def build(self):
        self.network.build()
        self.model = self.network.model