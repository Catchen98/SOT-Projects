from .registry import DATASETS
from .custom import CustomDataset


@DATASETS.register_module
class KittiDataset(CustomDataset):
    CLASSES = ('Car', 'Van', 'Truck', 'Pedestrian', 'Person', 'Cyclist', 'Tram', 'Misc')
    def __init__(self, **kwargs):
        super(KittiDataset, self).__init__(**kwargs)