from augment_loader import DataLoader

class LabelAugmenter:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    


    def augment_labels(self, image, bboxes):
        pass