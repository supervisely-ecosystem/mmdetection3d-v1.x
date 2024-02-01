from random import randint
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
from collections import OrderedDict
from supervisely.app.widgets import GridChart, Container, Field, Empty, IFrame, GridPlot
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

        new_kwargs = {"title": metric, "series": srs, **kwargs}
        self._metrics[metric] = new_kwargs

    def create_series(self, metric: str, series: Union[List[str], str]):
        if isinstance(series, str):
            series = [series]
        new_series = [{"name": ser, "data": []} for ser in series]
        self._metrics[metric]["series"].extend(new_series)

    def compile_plot_field(
        self,
        make_right_indent=True,
        is_iframe=False,
    ) -> Tuple[Union[Field, Container], Union[GridChart, IFrame]]:
        if make_right_indent is True:
            self.create_metric("right_indent_empty_plot", ["right_indent_empty_plot"])

        if is_iframe is True:
            # res = runner.val_evaluator.metrics[0].saved_results
            # pts_filename = g.PROJECT_DIR + "/" + res["pcd_path"]
            # pcd = o3d.io.read_point_cloud(pts_filename)
            # xyz = np.asarray(pcd.points, dtype=np.float32)
            # self.initialize_iframe
            # plot = IFrame("static/point_cloud_visualization.html", height="500px", width="1100px")
            plot = IFrame()
            pass
        else:
            data = list(self._metrics.values())
            plot = GridChart(data, columns=len(data))
            # plot = GridPlot(data, columns=len(data))
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

    def update_iframe():
        pass

    def add_scalar(
        self,
        stage_id: str,
        metric_name: str,
        series_name: str,
        x: NumT,
        y: NumT,
    ):
        grid_chart: GridChart = self._stages[stage_id]["raw"]
        grid_chart.add_scalar(f"{metric_name}/{series_name}", y, x)

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

    def initialize_iframe(self, stage_id: str, xyz: np.ndarray):
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
        iframe.set("static/point_cloud_visualization.html", height="500px", width="1000px")


train_stage = StageMonitoring("train", "Train")
train_stage.create_metric("Loss", ["loss"], stroke_curve="straight")
train_stage.create_metric("Learning Rate", ["lr"], decimalsInFloat=6, stroke_curve="straight")

visualization = StageMonitoring("visual", "Visualization")


val_stage = StageMonitoring("val", "Validation")
val_stage.create_metric("Metrics", g.NUSCENES_METRIC_KEYS, stroke_curve="straight")
val_stage.create_metric("3D Errors", stroke_curve="straight")
val_stage.create_metric("Class-Wise AP", stroke_curve="straight")


def add_3d_errors_metric():
    grid_chart: GridChart = monitoring._stages["val"]["raw"]
    grid_chart._widgets["3D Errors"].show()


def add_classwise_metric(selected_classes):
    grid_chart: GridChart = monitoring._stages["val"]["raw"]
    grid_chart._widgets["Class-Wise AP"].show()


monitoring = Monitoring()
monitoring.add_stage(train_stage, True)
monitoring.add_stage(visualization, True, is_iframe=True)
monitoring.add_stage(val_stage, True)


# add_btn = Button("add")

grid_chart: GridChart = monitoring._stages["val"]["raw"]
grid_chart._widgets["Class-Wise AP"].hide()
grid_chart._widgets["3D Errors"].hide()
