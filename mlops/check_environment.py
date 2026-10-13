
import sys

import imblearn
import mlflow
import numpy
import pandas
import sklearn

print("Python:", sys.version.split()[0])
print("MLflow:", mlflow.__version__)
print("scikit-learn:", sklearn.__version__)
print("imbalanced-learn:", imblearn.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
