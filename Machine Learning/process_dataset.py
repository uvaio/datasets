import pandas as pd

# Read the original CSV file
input_file = 'original_dataset.csv'
df = pd.read_csv(input_file)

# Select the desired columns including "vondstnummer"
selected_columns = ["subcategorie", "object", "objectdeel", "vlak_min", "vlak_max", "begin_dat", "eind_dat", "niveau1", "vondstnummer"]
df_selected = df[selected_columns]

# Specify the name for the new CSV file
output_file = 'selected_data.csv'

# Write the selected columns to a new CSV file
df_selected.to_csv(output_file, index=False)

print(f'Selected columns saved to {output_file}')
