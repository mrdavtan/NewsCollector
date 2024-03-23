import json
import uuid
from datetime import datetime
import argparse
import logging
from googlesearch import search

def sanitize_title(title):
    return title.replace("\n", " ").replace("/", "").strip()

def perform_google_search(title_data):
    search_results = []

    for item in title_data:
        if not item["has_content_or_description"]:
            query = sanitize_title(item["title"])
            search_results_item = {
                "uuid": item["uuid"],
                "title": item["title"],
                "links": []
            }

            try:
                for url in search(query, tld="co.in", num=10, stop=10, pause=2):
                    search_results_item["links"].append({
                        "url": url,
                        "visited": False
                    })
            except ImportError:
                logging.error("No module named 'google' found")

            search_results.append(search_results_item)

    return search_results

def save_search_results(search_results, output_file):
    search_uuid = str(uuid.uuid4())
    current_datetime = datetime.now()
    current_date = current_datetime.strftime("%Y%m%d")
    current_time = current_datetime.strftime("%H%M%S")

    search_results_data = {
        "uuid": search_uuid,
        "creation_date": current_date,
        "data": search_results
    }

    with open(output_file, "w") as file:
        json.dump(search_results_data, file, indent=4)

    logging.info(f"Search results saved to: {output_file}")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Perform Google search for titles without content.")
parser.add_argument("input_file", help="Path to the input title_list_YYYYMMDD_HHMMSS.json file.")
args = parser.parse_args()

# Read the input title_list_YYYYMMDD_HHMMSS.json file
with open(args.input_file, "r") as file:
    title_data = json.load(file)["data"]

# Perform Google search for titles without content or description
search_results = perform_google_search(title_data)

# Generate the output file name
current_datetime = datetime.now()
current_date = current_datetime.strftime("%Y%m%d")
current_time = current_datetime.strftime("%H%M%S")
output_file = f"search_title_list_{current_date}_{current_time}.json"

# Save the search results to the output file
save_search_results(search_results, output_file)
