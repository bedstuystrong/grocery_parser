import os

FOLDER_ROOT = os.path.dirname(os.path.abspath(__file__))
GROCERIES_PATH = os.path.join(FOLDER_ROOT, "groceries_list.csv")
TAXONOMY_PATH = os.path.join(FOLDER_ROOT, "taxonomy.csv")
OUTPUT_PATH = os.path.join(FOLDER_ROOT, "output.json")
UNIDENTIFIED_OUTPUT_PATH = os.path.join(FOLDER_ROOT, "unidentified.json")