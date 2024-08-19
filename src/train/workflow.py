# Description: This file contains versioning features and the Workflow class that is used to add input and output to the workflow.

import supervisely as sly
import src.ui.models as models_ui
from supervisely.api.file_api import FileInfo


def workflow_input(api: sly.Api, project_id: int):
    try:
        api.app.workflow.add_input_project(project_id)
        file_info = None
        if not models_ui.is_pretrained_model_radiotab_selected():
            remote_weights_path = models_ui.get_selected_custom_path()
            if remote_weights_path is None or remote_weights_path == "":
                sly.logger.debug(
                    "Workflow Input: weights file path is not specified. Cannot add input file to the workflow."
                )
            else:
                file_info = api.file.get_info_by_path(sly.env.team_id(), remote_weights_path)        
        if file_info is not None:
            api.app.workflow.add_input_file(file_info, model_weight=True)
        sly.logger.debug(
            f"Workflow Input: Project ID - {project_id}, Input File - {file_info if file_info else None}"
        )
    except Exception as e:
        sly.logger.debug(f"Failed to add input to the workflow: {repr(e)}")


def workflow_output(api: sly.Api, artefacts_path: str):
    try:
        file_infos_list = api.file.list(sly.env.team_id(), artefacts_path, recursive=False, return_type="fileinfo")
        all_checkpoints = []
        best_checkpoints = []
        for info in file_infos_list:
            if "best" in info.name:
                best_checkpoints.append(info)
            elif ".pth" in info.name:
                all_checkpoints.append(info)
        if len(best_checkpoints) > 1:
            best_file_info = sorted(best_checkpoints, key=lambda x: x.name, reverse=True)[0]
        elif len(best_checkpoints) == 1:
            best_file_info = best_checkpoints[0]
        else:
            best_file_info = None
        
        if len(all_checkpoints) > 1:
            last_file_info = sorted(all_checkpoints, key=lambda x: x.name, reverse=True)[0]
        elif len(all_checkpoints) == 1:
            last_file_info = all_checkpoints[0]
        else:
            last_file_info = None

        if best_file_info is None and last_file_info is not None:
            best_file_info = last_file_info
        elif best_file_info is None and last_file_info is None:
            sly.logger.debug(
                f"Workflow Output: No checkpoint files found in Team Files. Cannot set workflow output."
            )
            return
        
        module_id = api.task.get_info_by_id(api.task_id).get("meta", {}).get("app", {}).get("id")

        if not models_ui.is_pretrained_model_radiotab_selected():
            node_custom_title = "Train Custom Model"
        else:
            node_custom_title = None
        if best_file_info:
            node_settings = sly.WorkflowSettings(
                title=node_custom_title,
                url=(
                    f"/apps/{module_id}/sessions/{api.task_id}"
                    if module_id
                    else f"apps/sessions/{api.task_id}"
                ),
                url_title="Show Results",
            )
            relation_settings = sly.WorkflowSettings(
                title="Checkpoints",
                icon="folder",
                icon_color="#FFA500",
                icon_bg_color="#FFE8BE",
                url=f"/files/{best_file_info.id}/true",
                url_title="Open Folder",
            )
            meta = sly.WorkflowMeta(relation_settings, node_settings)
            api.app.workflow.add_output_file(best_file_info, model_weight=True, meta=meta)
            sly.logger.debug(
                f"Workflow Output: Node custom title - {node_custom_title}, Best filename - {best_file_info}"
            )
            sly.logger.debug(f"Workflow Output: Meta \n    {meta.as_dict}")
        else:
            sly.logger.debug(
                f"File {best_file_info} not found in Team Files. Cannot set workflow output."
            )
    except Exception as e:
        sly.logger.debug(f"Failed to add output to the workflow: {repr(e)}")
