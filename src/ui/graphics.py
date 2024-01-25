from random import randint
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
from collections import OrderedDict
from supervisely.app.widgets import GridPlot, Container, Field, Empty, IFrame
from supervisely.app.content import StateJson, DataJson
import plotly.graph_objects as go
import src.globals as g

NumT = Union[int, float]


class StageMonitoring(object):
    def __init__(self, stage_id: str, title: str, description: str = None) -> None:
        self._name = stage_id
        self._metrics = OrderedDict()
        self._title = title
        self._description = description

    def create_metric(self, metric: str, series: Optional[List[str]] = None, **kwargs):
        if metric in self._metrics:
            raise ArithmeticError("Metric already exists.")

        if series is None:
            srs = []
        else:
            srs = [
                {
                    "name": ser,
                    "data": [],
                }
                for ser in series
            ]

        self._metrics[metric] = {
            "title": metric,
            "series": srs,
        }
        self._metrics[metric].update(kwargs)

    def create_series(self, metric: str, series: Union[List[str], str]):
        if isinstance(series, str):
            series = [series]
        new_series = [{"name": ser, "data": []} for ser in series]
        self._metrics[metric]["series"].extend(new_series)

    def compile_plot_field(
        self,
        make_right_indent=True,
        is_iframe=False,
    ) -> Tuple[Union[Field, Container], Union[GridPlot, IFrame]]:
        if make_right_indent is True:
            self.create_metric("right_indent_empty_plot", ["right_indent_empty_plot"])

        if is_iframe is True:
            plot = IFrame("static/point_cloud_visualization.html", height="500px", width="1100px")
        else:
            data = list(self._metrics.values())
            plot = GridPlot(data, columns=len(data))
            if make_right_indent is True:
                plot._widgets["right_indent_empty_plot"].hide()

        field = Field(plot, self._title, self._description)
        return field, plot

    @property
    def name(self):
        return self._name


class Monitoring(object):
    def __init__(self) -> None:
        self._stages = {}
        self.container = None

    def add_stage(self, stage: StageMonitoring, make_right_indent=True, is_iframe=False):
        field, plot = stage.compile_plot_field(make_right_indent, is_iframe)
        self._stages[stage.name] = {}
        self._stages[stage.name]["compiled"] = field
        self._stages[stage.name]["raw"] = plot

    def update_iframe(
        self,
        stage_id: str,
        xyz: np.ndarray,
    ):
        fig = go.Figure()

        scatter_trace = go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker=dict(
                size=5,  # Adjust the size of markers
                color="blue",  # You can also use an array for individual colors
                opacity=0.8,
            ),
        )

        fig.add_trace(scatter_trace)

        fig.update_layout(
            scene=dict(xaxis_title="X-axis", yaxis_title="Y-axis", zaxis_title="Z-axis")
        )
        fig.update_layout(
            title="Point Cloud Visualization",
            scene=dict(aspectmode="cube"),  # Keep the aspect ratio equal
            margin=dict(l=0, r=0, b=0, t=0),  # Adjust the margin for a clean layout
        )

        # fig.show()
        fig.write_html(g.STATIC_DIR.joinpath("point_cloud_visualization.html"))

        iframe: IFrame = self._stages[stage_id]["raw"]
        iframe.set("static/point_cloud_visualization.html", height="900px", width="300px")

    def add_scalar(
        self,
        stage_id: str,
        metric_name: str,
        series_name: str,
        x: NumT,
        y: NumT,
    ):
        gridplot: GridPlot = self._stages[stage_id]["raw"]
        gridplot.add_scalar(f"{metric_name}/{series_name}", y, x)

    def add_scalars(
        self,
        stage_id: str,
        metric_name: str,
        new_values: Dict[str, NumT],
        x: NumT,
    ):
        self._stages[stage_id]["raw"].add_scalars(
            metric_name,
            new_values,
            x,
        )

    def compile_monitoring_container(self, hide: bool = False) -> Container:
        if self.container is None:
            self.container = Container([stage["compiled"] for stage in self._stages.values()])
        if hide:
            self.container.hide()
        return self.container


train_stage = StageMonitoring("train", "Train")
train_stage.create_metric("Loss", ["loss"])
train_stage.create_metric("Learning Rate", ["lr"], decimals_in_float=6)

visualization = StageMonitoring("visual", "Visualization")


val_stage = StageMonitoring("val", "Validation")
val_stage.create_metric("Metrics", g.NUSCENES_METRIC_KEYS)
val_stage.create_metric("3D Errors")
val_stage.create_metric("Class-Wise AP")


def add_3d_errors_metric():
    gp: GridPlot = monitoring._stages["val"]["raw"]
    gp._widgets["3D Errors"].show()


def add_classwise_metric(selected_classes):
    gp: GridPlot = monitoring._stages["val"]["raw"]
    gp._widgets["Class-Wise AP"].show()


monitoring = Monitoring()
monitoring.add_stage(train_stage, True)
monitoring.add_stage(visualization, True, is_iframe=True)
monitoring.add_stage(val_stage, True)


# add_btn = Button("add")

gp: GridPlot = monitoring._stages["val"]["raw"]
gp._widgets["Class-Wise AP"].hide()
gp._widgets["3D Errors"].hide()
