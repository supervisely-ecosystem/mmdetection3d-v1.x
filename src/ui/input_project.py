from supervisely.app.widgets import Card, ProjectThumbnail

from src.globals import api, PROJECT_ID


project_info = api.project.get_info_by_id(PROJECT_ID)
project_th = ProjectThumbnail(project_info)


card = Card(
    title="Input project",
    description="Selected project from which images and annotations will be downloaded",
    content=project_th,
)
