import pandas as pd
import datasets

df = datasets.load_dataset('bentrevett/multi30k')
df.save_to_disk(dataset_dict_path='data')