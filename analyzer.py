import pandas as pd
import numpy as np

class DataAnalyzer:

    def __init__(self, df):
        self.df = df

    # Clasificación de variables
    def classify_variables(self):

        numeric = self.df.select_dtypes(include=["int64","float64"]).columns.tolist()

        categorical = self.df.select_dtypes(include=["object"]).columns.tolist()

        return numeric, categorical


    # Estadísticas descriptivas
    def descriptive_stats(self):

        return self.df.describe()


    # Valores nulos
    def missing_values(self):

        missing = self.df.isnull().sum()

        percent = (missing / len(self.df)) * 100

        table = pd.DataFrame({
            "Valores Nulos": missing,
            "Porcentaje %": percent
        })

        return table


    # Moda
    def mode(self, column):

        moda = self.df[column].mode()

        if len(moda) > 0:
            return moda[0]
        else:
            return None


    # Media
    def mean(self, column):

        return self.df[column].mean()


    # Mediana
    def median(self, column):

        return self.df[column].median()