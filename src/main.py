from src.custom_dataset import CustomDataset
from mmdet3d.apis.inference import inference_detector, init_model
from mmengine import Config
from mmdet3d.apis import LidarDet3DInferencer
LidarDet3DInferencer.list_models("mmdet3d")

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=3,  # replace with your point cloud data dimension
        use_dim=3),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# Instantiate the CustomDataset
dataset = CustomDataset(
    data_root='app_data/sly_project/',
    ann_file='app_data/sly_project/infos_train.pkl',
    pipeline=test_pipeline,
)

checkpoint_path = None
config_file = 'configs/baseline_model_config.py'  # Replace with the actual config file path
cfg = Config.fromfile(config_file)

# Build the detector
model = init_model(cfg, checkpoint_path, device='cuda:0')


# Inference through the dataset
for i in range(len(dataset)):
    data = dataset[i]
    result = inference_detector(model, data)
    # Process the result (e.g., visualization, saving, etc.)
