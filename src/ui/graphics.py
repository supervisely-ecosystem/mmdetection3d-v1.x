from random import randint
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
from collections import OrderedDict
from supervisely.app.widgets import GridChart, Container, Field, Empty, IFrame, GridPlot
from supervisely.app.content import StateJson, DataJson
import plotly.graph_objects as go
import src.globals as g
import supervisely as sly
import open3d as o3d
import numpy as np
import pickle, random
import copy

# from src.inference.functional import create_sly_annotation, up_bbox3d, filter_by_confidence

import src.tests.draw_test as drt

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

    def initialize_iframe(self, stage_id: str, pts_filepath: str, gt_bboxes_3d=[]) -> None:

        pcd = o3d.io.read_point_cloud(pts_filepath)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=1)

        xyz = np.asarray(pcd_downsampled.points, dtype=np.float32)
        center = np.mean(xyz, axis=0)

        distances = np.linalg.norm(xyz - center, axis=1)
        normalized_distances = (distances - np.min(distances)) / (
            np.max(distances) - np.min(distances)
        )

        fig = go.Figure()

        scatter_trace = go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker=dict(
                size=2,  # Adjust the size of markers
                color=normalized_distances,  # You can also use an array for individual colors
                opacity=0.5,
            ),
        )

        fig.add_trace(scatter_trace)

        fig.update_layout(
            scene=dict(
                aspectmode="data",
                camera=dict(
                    eye=dict(x=0.1, y=0.1, z=1.5),
                ),
                xaxis_title="X-axis",
                yaxis_title="Y-axis",
                zaxis_title="Z-axis",
            )
        )
        fig.update_layout(
            title="Point Cloud Visualization",
            scene=dict(aspectmode="cube"),  # Keep the aspect ratio equal
            margin=dict(l=0, r=0, b=0, t=0),  # Adjust the margin for a clean layout
        )

        if len(gt_bboxes_3d) != 0:
            for gt_bbox in gt_bboxes_3d:

                b, bbox_label_3d = gt_bbox["bbox_3d"], gt_bbox["bbox_label_3d"]

                if b[0] < -100:
                    continue

                center, dimensions, yaw = b[:3], b[3:6], b[6]
                box_corners = drt.get_box_corners(center, dimensions, yaw)
                lines_x, lines_y, lines_z = [], [], []

                for start, end in drt.lines:
                    lines_x.extend([box_corners[start][0], box_corners[end][0], None])
                    lines_y.extend([box_corners[start][1], box_corners[end][1], None])
                    lines_z.extend([box_corners[start][2], box_corners[end][2], None])

                fig.add_trace(
                    go.Scatter3d(
                        x=lines_x, y=lines_y, z=lines_z, mode="lines", line=dict(color="green")
                    )
                )

            fig.update_layout(scene=dict(aspectmode="data"))

        fig.write_html(g.STATIC_DIR.joinpath(f"point_cloud_visualization.html"))

        self._stages[stage_id]["fig"] = fig

        iframe: IFrame = self._stages[stage_id]["raw"]
        iframe.set(f"static/point_cloud_visualization.html", height="500px", width="1000px")

    def update_iframe(self, stage_id, pred_bboxes_3d, epoch):

        pcl_fig = self._stages[stage_id]["fig"]
        fig = copy.deepcopy(pcl_fig)
        for b in pred_bboxes_3d:

            center, dimensions, yaw = b[:3], b[3:6], b[6]

            box_corners = drt.get_box_corners(center, dimensions, yaw)

            lines_x, lines_y, lines_z = [], [], []

            for start, end in drt.lines:
                lines_x.extend([box_corners[start][0], box_corners[end][0], None])
                lines_y.extend([box_corners[start][1], box_corners[end][1], None])
                lines_z.extend([box_corners[start][2], box_corners[end][2], None])

            fig.add_trace(
                go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode="lines", line=dict(color="red"))
            )

            # for start, end in drt.lines:
            #     x_vals = [box_corners[start][0], box_corners[end][0]]
            #     y_vals = [box_corners[start][1], box_corners[end][1]]
            #     z_vals = [box_corners[start][2], box_corners[end][2]]

            #     # Adding a line trace for each pair of points
            #     fig.add_trace(
            #         go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, mode="lines", line=dict(color="red"))
            #     )

        fig.update_layout(scene=dict(aspectmode="data"))

        fname = "pred_bbox_visualization_" + str(epoch) + ".html"
        fig.write_html(g.STATIC_DIR.joinpath(fname))
        iframe: IFrame = self._stages[stage_id]["raw"]
        iframe.set(f"static/{fname}", height="500px", width="1000px")


train_stage = StageMonitoring("train", "Train")
train_stage.create_metric("Loss", ["loss"], stroke_curve="straight", data_type="tuple")
train_stage.create_metric(
    "Learning Rate", ["lr"], decimalsInFloat=6, stroke_curve="straight", data_type="tuple"
)

visualization = StageMonitoring("visual", "Visualization (updates every 5th epoch)")


val_stage = StageMonitoring("val", "Validation")
val_stage.create_metric(
    "Metrics", g.NUSCENES_METRIC_KEYS, stroke_curve="straight", data_type="tuple"
)
val_stage.create_metric("3D Errors", stroke_curve="straight", data_type="tuple")
val_stage.create_metric("Class-Wise AP", stroke_curve="straight", data_type="tuple")


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

grid_chart_val: GridChart = monitoring._stages["val"]["raw"]
grid_chart_val._widgets["3D Errors"].hide()
grid_chart_val._widgets["Class-Wise AP"].hide()
