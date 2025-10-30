from .BaseModel import BaseModel


class ClusteringModel(BaseModel):

    def __init__(self, file_route, generate_file, head, columns,
                 info, describe, nulls, duplicated_sum, distribution):

        super().__init__(file_route, generate_file, head,
                         columns, info, describe, nulls, duplicated_sum, distribution)

    def train(self):
        pass
