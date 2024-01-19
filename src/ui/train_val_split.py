import supervisely as sly
from supervisely.app.widgets import TrainValSplits, Card, Container, Button
import src.globals as g
from supervisely.project import get_project_class

select_btn = Button("Select")
splits = TrainValSplits(project_id=g.PROJECT_ID)
content = Container(widgets=[splits, select_btn])

card = Card(
    title="Train / Validation splits",
    description="Define how to split your data to train/val subsets.",
    content=content,
)
card.lock("Confirm selected classes.")


def dump_train_val_splits(project_dir):
    project_class = get_project_class(g.PROJECT_INFO.type)
    splits._project_fs = project_class(project_dir, sly.OpenMode.READ)

    train_split, val_split = splits.get_splits()
    app_dir = g.app_dir
    sly.json.dump_json_file(train_split, f"{app_dir}/train_split.json")
    sly.json.dump_json_file(val_split, f"{app_dir}/val_split.json")

    return train_split, val_split
