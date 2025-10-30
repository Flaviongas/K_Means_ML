from .ClusteringModel import ClusteringModel
from .ClassificationModel import ClassificationModel
from time import sleep


def generate_data_summary(config):
    classification = config['classification']
    clustering = config['clustering']

    if classification['explore']:
        _, file, head, dshape, generate_file, df_columns, is_null, duplicated_sum, value_counts, value_counts_column, describe, info = classification.values()
        classification = ClassificationModel(file_route=file, generate_file=generate_file,
                                             head=head, dshape=dshape, columns=df_columns, nulls=is_null, duplicated_sum=duplicated_sum,
                                             distribution=value_counts, distribution_column=value_counts_column, describe=describe,
                                             info=info, dataset_name="Objetos Estelares")
        classification.explore()

    sleep(1)  # Para que los nombres sean distintos
    if clustering['explore']:
        _, file, head, dshape, generate_file, df_columns, is_null, duplicated_sum, value_counts, value_counts_column, describe, info = clustering.values()
        clustering = ClusteringModel(file_route=file, generate_file=generate_file,
                                     head=head, dshape=dshape, columns=df_columns, nulls=is_null, duplicated_sum=duplicated_sum,
                                     distribution=value_counts, distribution_column=value_counts_column, describe=describe, info=info, dataset_name="Pokemon")
        clustering.explore()
