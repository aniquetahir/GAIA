import pandas as pd

def list_of_dict_to_csv(lod, save_location):
    df = pd.DataFrame(lod)
    df.to_csv(save_location)