# NewsCollector Retriever

This project is an adaptation of the **NewsCollector** package by elisemercury. It has been modified to run as a cron job on a daily basis to collect news articles and save them as JSON files. The collected data is intended to be used by other programs for automating the creation of datasets and running graph visualizations.

## Features

- Collects news articles from various sources
- Saves articles as JSON files
- Runs as a daily cron job
- Cleans the articles folder of duplicates and remaining articles before each run

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/NewsCollectorRetriever.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## USage

To run the newcollector with automation, use the provided 'run.sh' script for convenience.

   ```bash
   ./run.sh
   ```

By default, the script will prompt for confirmation before running. To skip the prompt and run automatically, use the '-y' argument:

./run.sh -y

The script will first clean the articles folder of duplicates and archive them before running the newscollector.py script.

## Output

The newscollecto.py will output JSON files in a folder named by the date. Each JSON file represents a collected news article or the meta data such as title, source and date from the RSS feed.

## Acknowledgements

This project is based on the NewsCollector package by elisemercury. Special thanks to elisemercury for creating a cool project that served as the foundation for this retriever.

For more information about how the NewsCollector algorithm works, please refer to the following resources:

    NewsCollector Usage Documentation
    Medium article by elisemercury

License

This project is licensed under the MIT License.






