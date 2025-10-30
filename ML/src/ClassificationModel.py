from . import BaseModel


class ClusteringModel(BaseModel):

    def __init__(self, file_route, head, columns,
                 info, describe, nulls, distribution):

        super().__init__(file_route, head,
                         columns, info, describe, nulls, distribution)

    def train(self):
        pass
