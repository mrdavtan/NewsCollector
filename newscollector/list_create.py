import os
import json
import uuid
from datetime import datetime
import argparse
import logging

def sanitize_title(title):
    return title.replace("\n", " ").replace("/", "").strip()

def process_json_files(folder_path):
    data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                try:
                    json_data = json.load(file)

                    # Extract relevant information from the JSON
                    file_uuid = json_data.get("id", "")
                    title = json_data.get("title", "")
                    source = json_data.get("source", "")
                    url = json_data.get("url", "")
                    published_date = json_data.get("date", "")

                    # Sanitize the title
                    sanitized_title = sanitize_title(title)

                    # Append the extracted information to the data list
                    data.append({
                        "uuid": file_uuid,
                        "title": sanitized_title,
                        "source": source,
                        "url": url,
                        "published_date": published_date
                    })
                except (json.JSONDecodeError, KeyError):
                    logging.error(f"Error processing file: {filename}")

    # Get the current working directory
    current_dir = os.getcwd()

    # Create a new UUID for the output file
    output_uuid = str(uuid.uuid4())

    # Get the current date and time
    current_datetime = datetime.now()
    current_date = current_datetime.strftime("%Y%m%d")
    current_time = current_datetime.strftime("%H%M%S")

    # Create the output JSON structure
    output_data = {
        "uuid": output_uuid,
        "creation_date": current_date,
        "path": current_dir,
        "data": data
    }

    # Generate the output filename
    output_filename = f"title_list_{current_date}_{current_time}.json"

    # Write the output JSON to a file
    with open(output_filename, "w") as output_file:
        json.dump(output_data, output_file, indent=4)

    logging.info(f"Output file created: {output_filename}")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process JSON files and create a title list.")
parser.add_argument("folder_path", help="Path to the folder containing JSON files.")
args = parser.parse_args()

# Process the JSON files and create the output file
process_json_files(args.folder_path)
