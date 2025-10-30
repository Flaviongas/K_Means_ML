from .ClusteringModel import ClusteringModel


def generate_data_summary(config):
    classification = config['classification']
    clustering = config['clustering']

    if classification['explore']:
        pass

    if clustering['explore']:
        print("help")
        _, file, head, generate_file, df_columns, is_null, duplicated_sum, value_counts, describe, info = clustering.values()
        print(f"{generate_file=}")
        clustering = ClusteringModel(file_route=file, generate_file=generate_file,
                                                head=head, columns=df_columns, nulls=is_null, duplicated_sum=duplicated_sum,
                                                distribution=value_counts, describe=describe, info=info)
        clustering.explore()
