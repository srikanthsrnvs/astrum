from image_classifier import ImageClassifier


class GenericBuilder:

    def __init__(self, job_type, output_classes, job_id, log_dir, urls=[]):
        self.job_type = job_type
        self.log_dir = log_dir
        self.output_classes = output_classes
        self.urls = urls
        self.job_id = job_id
        if self.job_type == 'image_classification':
            self.network = ImageClassifier(
                urls, output_classes, job_id, log_dir)
        elif self.job_type == 'structured_prediction':
            # TODO: Structured data predictor should be built
            pass
        elif self.job_type == 'structured_classification':
            # TODO: Structured classifier should be built
            pass
        elif self.job_type == 'custom':
            # TODO: Custom built nn should be built
            pass
        else:
            raise Exception()

    def build(self):
        self.network.build()
        self.model = self.network.model