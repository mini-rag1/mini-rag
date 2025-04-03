from .BaseDataModel import BaseDataModel
from .db_schemas import Project
from .enums.DataBaseEnum import DataBaseEnum

class ProjectModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_PROJECT_NAME.value]

    #i need the func of init collection to create the indexes
    #since its async i need to await it
    #but the init cant be async 
    #so the solution is to make 3rd function that will call the init and await it
    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client=db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_PROJECT_NAME.value not in all_collections:
            self.collection = self.db_client[DataBaseEnum.COLLECTION_PROJECT_NAME.value]
            indexes = Project.get_indexes()
            for index in indexes:
                await self.collection.create_index(index["key"], 
                                                   name = index["name"], 
                                                   unique = index["unique"])

    async def create_project(self, project: Project):
        result = await self.collection.insert_one(project.dict(by_alias=True, exclude_unset=True))
        # inserting the doc into the db and converting the pydantic model to a dict using .model_dump()
        project.id = result.inserted_id
        # will return the id we have
        return project
    
    async def get_project_or_create_one(self, project_id: str):
        record = await self.collection.find_one({"project_id": project_id})

        if record is None:
            # create a new project
            project = Project(project_id=project_id)
            project_id = await self.create_project(project)
            project = await self.collection.find_one({"_id": project_id})
            return Project(**project)

        return Project(**record)  # returning the record as a pydantic model instead of a dict
    
    async def get_all_projects(self, page: int = 1, page_size: int = 10):
        # count total number of docs
        total_documents = await self.collection.count_documents({})

        # the documents calc
        total_pages = total_documents // page_size
        if total_documents % page_size > 0:
            total_pages += 1

        cursor = self.collection.find({}).skip((page-1) * page_size).limit(page_size)  # cursor is like a pointer that points on an array
        projects = []
        async for document in cursor:
            projects.append(Project(**document))

        return projects, total_pages
    