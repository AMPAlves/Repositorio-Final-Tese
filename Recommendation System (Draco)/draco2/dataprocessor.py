import os
import pandas as pd
import json

class Aggregate:
    def __init__(self,type,attribute):
        self.type = type,
        self.attribute = attribute

desktop = os.path.normpath(os.path.expanduser(r"C:\Users\ideapad\Desktop\Repositório Final Tese\Data Repository"))
domain_id = "14"
domain_usecase = "141"
domain_question = "1411"
aggregate_task = {}
scale_dict = {}
data_dict = {}
field_dict = {}
task_at_hand = {}

def load_dataset(path):
    new_path = os.path.join(desktop, path, "Use Cases", "Use Case - "+ str(usecase_number), "Data", "Data Files")
    json_path = os.path.join(new_path,"dataset.json")
    dataset = None
    if (os.path.isfile(json_path)):
        dataset = pd.read_json(json_path,encoding="ISO-8859-1")
    csv_path = os.path.join(new_path,"dataset.csv")
    if (os.path.isfile(csv_path)):
        dataset = pd.read_csv(csv_path,encoding="ISO-8859-1")
    #print(dataset.head())
    datasetFields = list(dataset.columns.values)
    dataset = dataset[dataset[datasetFields].notnull().all(1)]
    return dataset

def countOfFiles(path):
    onlyfiles = os.listdir(path)
    return len(onlyfiles)



pathname = input()
usecase_number = input()
domainquestion_number = input()
featured_dataset = load_dataset(pathname)
print(len(featured_dataset.index))
featured_dataset = featured_dataset[featured_dataset['year'] == 80]
print(len(featured_dataset.index))
