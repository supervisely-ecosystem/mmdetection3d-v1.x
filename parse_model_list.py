mmdetection3d_root_dir = "mmdetection3d"
import os
all_base_configs = os.listdir(f"{mmdetection3d_root_dir}/configs/_base_/models")

detection3d_models = '''SECOND (Sensor'2018)
    PointPillars (CVPR'2019)
    PointRCNN (CVPR'2019)
    SSN (ECCV'2020)
    3DSSD (CVPR'2020)
    SA-SSD (CVPR'2020)
    Part-A^2 (TPAMI'2020)
    PV-RCNN (CVPR'2020)
    CenterPoint (CVPR'2021)
    CenterFormer (ECCV'2022)
    BEVFusion (ICRA'2023)'''

def make_dict(s):
    # parse the string to list of dicts like [{'name': 'SECOND', 'paper': 'Sensor', 'year': '2018'}, ...]
    result = []
    s = s.split('\n')
    for line in s:
        line = line.strip()
        if line:
            name, paper_year = line.split(' ')
            paper, year = paper_year.strip("()").split('\'')
            result.append({'name': name.strip(), 'paper': paper.strip(), 'year': int(year.strip())})
    return result

res = make_dict(detection3d_models)

from mmdet3d.apis import Base3DInferencer
mim_dir = Base3DInferencer._get_repo_or_mim_dir('mmdet3d')
model_index = list(Base3DInferencer._get_models_from_metafile(mim_dir))
# model_index = [{'Name': '3dssd_4x4_kitti-3d-car', 'In Collection': '3DSSD', 'Config': 'configs/3dssd/3dssd_...-3d-car.py', 'Metadata': {'Training Memory (GB)': 4.7}, 'Results': [{...}], 'Weights': 'https://download.ope...9c8fc4.pth'}]
for x in res:
    model_name = x['name']
    base_model_name = None
    for model in model_index:
        if model['In Collection'] == model_name:
            assert model['Config'].startswith('configs')
            base_model_name = model['Config'].split('/')[1]
            x['model_name'] = base_model_name
            
            # find base_configs
            base_configs = []
            for base_config in all_base_configs:
                if base_config.startswith(base_model_name):
                    base_configs.append(f"configs/_base_/models/{base_config}")
            if len(base_configs) == 0:
                print(f"Base config not found for {model_name}")
            else:
                x['base_configs'] = base_configs
            
            # collect pre_trained_configs
            pre_trained_models = [model for model in model_index if model['Config'].split('/')[1] == base_model_name]
            pre_trained_configs = []
            for model in pre_trained_models:
                if model.get('Weights'):
                    metadata = model.get('Metadata') or model['metadata']
                    pre_trained_configs.append({
                        'config': model['Config'],
                        'weights': model['Weights'],
                        'metadata': metadata,
                        'results': model['Results']
                    })
                else:
                    print(f"weights not found for {model['Config']}")
            x['pre_trained_configs'] = pre_trained_configs

            # collect results and metadata
            break
    if base_model_name is None:
        print(f"Model {model_name} not found in model_index")

# filter out models that are not in model_index
res = [x for x in res if 'model_name' in x]

model_list = {"detection_3d": res}

# save the result to a json file
import json
with open('model_list.json', 'w') as f:
    json.dump(model_list, f, indent=4)