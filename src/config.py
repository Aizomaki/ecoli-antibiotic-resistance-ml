from pathlib import Path

DATA_PATH = Path("archive/BVBRC_E.coli_Dataset.csv")

ID_COL = "Genome ID"
ANTIBIOTIC_COL = "Antibiotic"
LABEL_COL = "Resistant Phenotype"

RAW_COLUMNS = [
    ID_COL,
    ANTIBIOTIC_COL,
    LABEL_COL,
    "Laboratory Typing Method",
    "Testing Standard",
    "Testing Standard Year",
    "Evidence",
    "Laboratory Typing Platform",
    "Vendor",
    "Source",
    "PubMed",
]

CATEGORICAL_FEATURES = [
    "Laboratory Typing Method",
    "Testing Standard",
    "Evidence",
    "Laboratory Typing Platform",
    "Vendor",
]

NUMERIC_FEATURES = [
    "Testing Standard Year",
    "num_tests",
    "num_unique_antibiotics",
    "has_pubmed",
    "has_source",
]

LABEL_MAP = {
    "Resistant": 1,
    "Susceptible": 0,
    "Intermediate": 0,
    "Nonsusceptible": 1,
    "Susceptible-dose dependent": 0,
}

MIN_LABELS_PER_ANTIBIOTIC = 2000
MIN_POSITIVES_PER_ANTIBIOTIC = 100
MIN_NEGATIVES_PER_ANTIBIOTIC = 100
DATA_URL = "https://www.kaggle.com/datasets/valeriamaciel/e-coli-resistance-dataset"
MIN_COMPLETE_CASES = 200
MAX_ANTIBIOTICS = 10

LOGREG_MAX_ITER = 5000

TASK_B_MIN_OTHER_TESTS = 1

TEST_SIZE = 0.2
RANDOM_STATE = 42

OUTPUT_DIR = Path("reports")
