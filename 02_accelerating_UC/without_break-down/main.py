import os
import re
import numpy as np

# 1. File Reading
def list_lp_files(directory="data/uc/"):
    return [f for f in os.listdir(directory) if re.match(r'instance\d+\.lp', f)]

# 2. Parsing LP Format
def parse_lp(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    constraints = []
    objective = None

    # Extract constraints and objective function
    for line in lines:
        if line.startswith('Minimize') or line.startswith('Maximize'):
            objective = line.split()[1:]
        elif line.startswith('Subject To'):
            constraints = lines[lines.index(line)+1:]

    return objective, constraints

# 3. Feature Engineering
def convert_to_matrix(objective, constraints):
    # For this example, we'll use placeholder functions
    # In practice, you'd convert LP expressions to numerical matrices here
    objective_matrix = np.array(objective)  # Placeholder
    constraints_matrix = np.array(constraints)  # Placeholder

    return objective_matrix, constraints_matrix

# 4. Data Aggregation
def aggregate_data(directory="data/uc/"):
    files = list_lp_files(directory)
    objectives = []
    all_constraints = []

    for file in files:
        file_path = os.path.join(directory, file)
        objective, constraints = parse_lp(file_path)
        objective_matrix, constraints_matrix = convert_to_matrix(objective, constraints)

        objectives.append(objective_matrix)
        all_constraints.append(constraints_matrix)

    return objectives, all_constraints

# 5. Final Dataset Preparation
def prepare_dataset(directory="data/uc/"):
    objectives, constraints = aggregate_data(directory)
    # Here, for demonstration, we'll consider objectives as features and constraints as labels
    # Depending on the Neural Diving model setup, this might differ
    return objectives, constraints

X, y = prepare_dataset()
print("Sample features:", X[0])
print("Sample labels:", y[0])
