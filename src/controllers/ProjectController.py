from .BaseController import BaseController
from fastapi import UploadFile
from models import ResponseSignal
import os

class ProjectController(BaseController):
    def __init__(self):
        super().__init__()

    def get_project_path(self, project_id):
        # Convert project_id to string if it's not already
        project_id_str = str(project_id)
        project_dir = os.path.join(self.file_dir,
                                   project_id_str)
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
            
        return project_dir
    