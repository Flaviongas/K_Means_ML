import pandas as pd


class BaseModel():

    def __init__(self, file_route, generate_file, head, columns, info, describe, nulls, duplicated_sum, distribution):
        self.df = pd.read_csv(file_route)
        self.generate_file = generate_file
        self.head = head
        self.columns = columns
        self.info = info
        self.describe = describe
        self.nulls = nulls
        self.duplicated_sum = duplicated_sum
        self.distribution = distribution

    def explore(self):
        if self.head:
            print(" " * 40)
            print("Primeras filas del DataFrame:")
            print(self.df.head())
            print(" " * 40)

        if self.columns:
            print(" " * 40)
            print("Columnas del DataFrame:")
            print(self.df.columns)
            print(" " * 40)

        if self.info:
            print(" " * 40)
            print("Información del DataFrame:")
            print(self.df.info())
            print(" " * 40)

        if self.describe:
            print(" " * 40)
            print("Estadísticas descriptivas del DataFrame:")
            print(self.df.describe())
            print(" " * 40)

        if self.nulls:
            print(" " * 40)
            print("Cantidad de valores nulos por columna:")
            print(self.df.isnull().sum())
            print(" " * 40)

        if self.duplicated_sum:
            print(" " * 40)
            print("Cantidad de filas duplicadas:")
            print(self.df.duplicated().sum())
            print(" " * 40)

        if self.distribution:
            print(" " * 40)
            print("Distribución de clases:")
            print(self.df.value_counts())
            print(" " * 40)

    def train(self):
        pass
