#!/bin/bash

if [ "$1" != "-y" ]; then
  read -p "Continue to activate the virtual environment? (y/n): " choice
  if [ "$choice" != "y" ]; then
    echo "Script execution stopped."
    exit 1
  fi
fi

source .venv/bin/activate
echo "Virtual environment activated: $VIRTUAL_ENV"

echo "Current directory: $(pwd)"
echo "Navigating to: $(dirname "$0")/newscollector/articles"
cd "$(dirname "$0")/newscollector/articles"
echo "Moved to the 'articles' directory."
echo "Current directory: $(pwd)"

# Check if any JSON files exist in the 'articles' directory
if ls *.json >/dev/null 2>&1; then
  # Count the number of JSON files
  json_count=$(ls -1 *.json | wc -l)
  echo "Found $json_count JSON file(s) in the 'articles' directory."
  if [ "$json_count" -gt 20 ]; then
    if [ "$1" != "-y" ]; then
      read -p "More than 20 JSON files found. Continue to remove duplicates? (y/n): " choice
      if [ "$choice" != "y" ]; then
        echo "Script execution stopped."
        exit 1
      fi
    fi
    python3 remove_duplicate.py
    echo "remove_duplicate.py executed."
    if [ "$1" != "-y" ]; then
      read -p "Continue to archive articles? (y/n): " choice
      if [ "$choice" != "y" ]; then
        echo "Script execution stopped."
        exit 1
      fi
    fi
    python3 archive.py
    echo "archive.py executed."
  fi
  cd ../..
  echo "Moved back to the parent directory."
  echo "Current directory: $(pwd)"
else
  echo "No JSON files found in the 'articles' directory."
  echo "Skipping duplicate removal and archiving steps."
  cd ../..
  echo "Moved back to the parent directory."
  echo "Current directory: $(pwd)"
fi

if [ "$1" != "-y" ]; then
  read -p "Continue to run the 'newscollector.py' script? (y/n): " choice
  if [ "$choice" != "y" ]; then
    echo "Script execution stopped."
    exit 1
  fi
fi
python3 newscollector.py
echo "newscollector.py script executed."

echo "Script execution completed."
