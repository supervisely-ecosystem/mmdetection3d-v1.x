from supervisely.app.widgets import (
    Card,
    Container,
    RadioGroup,
    Button,
    NotificationBox,
    Field,
    Text,
)

from src.ui.utils import update_custom_button_params

msg = """
    <b>Detection 3D</b>: the model will predict Cuboids (3d bounding boxes) of objects.
    <br>
    <b>Segmentation 3D</b>: the model will predict a label for each point in a point cloud.
    """

info = NotificationBox(title="INFO: How to select the task?", description=msg, box_type="info")


task_selector = RadioGroup(
    items=[
        RadioGroup.Item(value="3D object detection", label="3D object detection"),
        # RadioGroup.Item(value="3D segmentation", label="3D instance segmentation (upcoming in the future updates)"),
    ],
    direction="vertical",
)


select_field = Field(title="Select the task you are going to solve:", content=task_selector)
select_btn = Button(text="Select task")
tmp_txt = Text("3D Segmentation is upcoming in the future updates.", status="info")

card = Card(
    title="Task",
    description="Select task from list below",
    content=Container(widgets=[info, select_field, select_btn, tmp_txt], direction="vertical"),
    lock_message="Please, select project and load data.",
)
