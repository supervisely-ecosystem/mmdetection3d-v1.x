from typing import Optional, Union
from mmdet3d.apis import LidarDet3DInferencer
from mmengine.dataset import Compose
from mmdet3d.utils import ConfigType


class PcdDet3DInferencer(LidarDet3DInferencer):

    def __init__(self,
                 model: Union[str, None] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: str = 'mmdet3d',
                 palette: str = 'none') -> None:
        super().__init__(
            model=model,
            weights=weights,
            device=device,
            scope=scope,
            palette=palette)

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        load_point_idx = self._get_transform_idx(pipeline_cfg,
                                                 'LoadPointsFromFile')
        if load_point_idx == -1:
            raise ValueError(
                'LoadPointsFromFile is not found in the test pipeline')

        load_cfg = pipeline_cfg[load_point_idx]
        self.coord_type, self.load_dim = load_cfg['coord_type'], load_cfg[
            'load_dim']
        self.use_dim = list(range(load_cfg['use_dim'])) if isinstance(
            load_cfg['use_dim'], int) else load_cfg['use_dim']

        pipeline_cfg[load_point_idx]['type'] = 'PCDLoader'
        return Compose(pipeline_cfg)
