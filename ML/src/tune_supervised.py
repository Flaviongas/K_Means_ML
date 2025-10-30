from .ClassificationModel import ClassificationModel


def tune_supervised(config):
    classification = config['classification']

    if classification['explore']:
        _, file, head, dshape, generate_file, df_columns, is_null, duplicated_sum, value_counts, value_counts_column, describe, info = classification.values()
        classification = ClassificationModel(file_route=file, generate_file=generate_file,
                                             head=head, dshape=dshape, columns=df_columns, nulls=is_null, duplicated_sum=duplicated_sum,
                                             distribution=value_counts, distribution_column=value_counts_column, describe=describe,
                                             info=info, dataset_name="Objetos Estelares")
        classification.train()
        classification.tune()
