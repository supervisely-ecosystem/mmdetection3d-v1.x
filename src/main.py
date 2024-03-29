import src.globals as g
import supervisely as sly
from supervisely.app.widgets import Container

import src.ui.handlers as handlers

layout = Container(widgets=[handlers.stepper])
app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)

g.app = app
