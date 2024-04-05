import os
from os import path
from marginModel import marketData


if __name__ == "__main__":

    project_folder = path.abspath(path.dirname(__file__))
    in_folder = path.join(project_folder, 'instruments')

    df = marketData(in_folder)
    print(df.head())