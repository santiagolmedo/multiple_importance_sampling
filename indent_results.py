import json

def format_json_file(file_path):
    try:
        # Read the original content of the file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Write the indented JSON back to the file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

        print("JSON file has been formatted and saved.")

    except json.JSONDecodeError:
        print("Error: The file does not contain valid JSON.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

format_json_file('results_mis_3_prod_multidimensional_aux.txt')
