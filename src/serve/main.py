from src.serve.model import MMDetection3dModel

model = MMDetection3dModel(use_gui=True)
model.serve()
