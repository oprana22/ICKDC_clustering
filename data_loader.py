import urllib.request
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer,
    load_digits, make_moons, make_circles,
    fetch_openml #used to fetch the UCI datasets
)

warnings.filterwarnings("ignore", category=FutureWarning) #suppress OpenML parser warnings


def load_sipu_synthetic(name):
    """
    Downloads synthetic datasets from the SIPU repository.
    """

    url = f"https://cs.uef.fi/sipu/datasets/{name}.txt"
    print(f"Downloading {name}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        data = np.loadtxt(response)
        X, y = data[:, :2], data[:, 2]
        return X, y
    except Exception as e:
        print(f"Failed to download {name}: {e}")
        return None, None


def fetch_uci_openml(name):
    """
    Fetches real world datasets from OpenML, cleans NaNs, and label encodes targets.
    """
    print(f"Fetching {name} from OpenML...")
    try:
        data = fetch_openml(name=name, as_frame=True, parser='auto') #download as a pandas df for easier handeling
        X_df = data.data
        y_raw = data.target

        X_df = X_df.select_dtypes(include=[np.number]) #drop non-numeric columns

        #combine X and y temporarily to drop rows with missing values together
        df_combined = X_df.copy()
        df_combined['target'] = y_raw
        df_combined = df_combined.dropna()

        #separate back into X array and cleanly encoded y array
        X = df_combined.drop(columns=['target']).values
        y = LabelEncoder().fit_transform(df_combined['target'])
        return X, y

    except Exception as e:
        print(f"Failed to fetch {name}: {e}")
        return None, None


def get_all_datasets():
    """
    fetches, normalizes, and packages datasets into synthetic and real world dictionaries
    """
    synthetic_datasets = {}
    real_datasets = {}
    scaler = MinMaxScaler()

    sipu_names = ["Aggregation", "R15", "spiral", "flame", "Compound"] #synthetic datasets

    for name in sipu_names:
        X, y = load_sipu_synthetic(name)
        if X is not None:
            synthetic_datasets[name.capitalize()] = (scaler.fit_transform(X), y)

    #scikit-learn synthetic manifolds
    X_toy, y_toy = make_moons(n_samples=1480, noise=0.1, random_state=42)
    synthetic_datasets["Toy"] = (scaler.fit_transform(X_toy), y_toy)

    X_circ, y_circ = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)
    synthetic_datasets["Circles"] = (scaler.fit_transform(X_circ), y_circ)


    #REAL WORLD DATASETS
    print("\nloading scikit-learn real world datasets...")

    iris = load_iris()
    real_datasets["Iris"] = (scaler.fit_transform(iris.data), iris.target)

    wine = load_wine()
    real_datasets["Wine"] = (scaler.fit_transform(wine.data), wine.target)

    wdbc = load_breast_cancer()
    real_datasets["Wdbc"] = (scaler.fit_transform(wdbc.data), wdbc.target)

    digits = load_digits() #my own addition
    real_datasets["Digits"] = (scaler.fit_transform(digits.data), digits.target)

    #fetch the datasets from OpenML
    print("\nloading additional real world datasets via OpenML...")
    openml_datasets = {
        "Seeds": "seeds",
        "Ecoli": "ecoli",
        "Dermatology": "dermatology",
        "Ionosphere": "ionosphere",
        "Parkinsons": "parkinsons",
        "Segmentation": "segment"
    }

    for pretty_name, openml_name in openml_datasets.items():
        X, y = fetch_uci_openml(openml_name)
        if X is not None:
            real_datasets[pretty_name] = (scaler.fit_transform(X), y) # min-max normalization from the paper
    return synthetic_datasets, real_datasets

if __name__ == "__main__":
    #test to see if works
    syn, real = get_all_datasets()
    print("\nsuccessfully loaded the following real world datasets:")
    for name, (X, y) in real.items():
        print(f"- {name}: {X.shape[0]} samples, {X.shape[1]} dimensions, {len(np.unique(y))} classes")