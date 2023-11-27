from mmdet3d.registry import DATASETS
from mmdet3d.datasets.det3d_dataset import Det3DDataset


@DATASETS.register_module()
class MyDataset(Det3DDataset):