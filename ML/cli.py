import argparse
import yaml
from .src.data_summary import generate_data_summary


def main():

    parser = argparse.ArgumentParser(description='ML CLI')
    subparsers = parser.add_subparsers(dest='command',
                                       help='Subcomandos disponibles')

    data_summary_parser = subparsers.add_parser('data-summary',
                                                help='Genera un resumen exploratorio de los datos')

    data_summary_parser.add_argument('--config', required=True,
                                     help='Ruta al archivo de configuración YAML')

    train_supervised_parser = subparsers.add_parser('train-supervised',
                                                    help='Entrena un modelo supervisado')

    train_supervised_parser.add_argument('--config', required=True,
                                         help='Ruta al archivo de configuración YAML')

    tune_supervised_parser = subparsers.add_parser('tune-supervised',
                                                   help='Realiza tuning de un modelo supervisado')

    tune_supervised_parser.add_argument('--config', required=True, help='Ruta al archivo de configuración YAML')

    cluster_kmeans_parser = subparsers.add_parser('cluster-kmeans',
                                                  help='Entrena un modelo de clustering KMeans')

    cluster_kmeans_parser.add_argument('--config', required=True,
                                       help='Ruta al archivo de configuración YAML')

    report_parser = subparsers.add_parser('report',
                                          help='Genera un reporte de resultados')

    report_parser.add_argument('--config', required=True,
                               help='Ruta al archivo de configuración YAML')

    args = parser.parse_args()

    match args.command:
        case 'data-summary':
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
                generate_data_summary(config)

        case 'train-supervised':
            raise NotImplementedError("train-supervised no está implementado aún.")

        case 'tune-supervised':
            raise NotImplementedError("tune-supervised no está implementado aún.")

        case 'cluster-kmeans':
            raise NotImplementedError("cluster-kmeans no está implementado aún.")

        case 'report':
            raise NotImplementedError("report no está implementado aún.")

        case _:
            parser.print_help()


if __name__ == '__main__':
    main()
