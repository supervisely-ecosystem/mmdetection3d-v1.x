from supervisely.app.widgets import Card, Container, RadioGroup, Button, NotificationBox, Field, Text

from src.ui.utils import update_custom_button_params

msg = """
    <b>3D object detection</b>: the model will predict bounding boxes of the objects.
    All annotations will be converted to Rectangles.
    <br>
    <b>3D instance Segmentation</b>: the model will predict bounding boxes and masks of the objects.
    Only Bitmap and Polygon annotations will be used."""

info = NotificationBox(title="INFO: How to select the task?", description=msg, box_type="info")


task_selector = RadioGroup(
    items=[
        RadioGroup.Item(value="3D object detection", label="3D object detection"),
        # RadioGroup.Item(value="3D instance segmentation", label="3D instance segmentation (upcoming in the future updates)"),
    ],
    direction="vertical",
)


select_field = Field(title="Select the task you are going to solve:", content=task_selector)
select_btn = Button(text="Select task")
tmp_txt = Text("The 3D instance segmentation is upcoming in the future updates", status='info')

card = Card(
    title="MMDetection task",
    description="Select task from list below",
    content=Container(widgets=[info, select_field, select_btn, tmp_txt], direction="vertical"),
    lock_message="Please, select project and load data.",
)
