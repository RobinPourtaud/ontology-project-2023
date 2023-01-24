import pandas as pd

def load_data(online : bool = True) -> pd.DataFrame:
    """Load data from a local file or from a URL
    # Equivalent to the result of step 4

    Args:
        online (bool, optional): Load data from a local file or from a URL. Defaults to True.

    Returns:
        pd.DataFrame: Dataframe with data
    """
    column = ["noun_phrase","frequency","Core concept"]
    if online: 
        try : 
            return pd.read_csv('https://docs.google.com/spreadsheets/d/16OlZEn2__3ALyECxYS00vlkECWGORTsfKs3ygWjaue8/export?gid=0&format=csv')[column]
        except Exception as e:
            print("Error: ", e)
            print("Loading data from local file")
    return pd.read_csv("data/NP.csv")[column]

