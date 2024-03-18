#!/bin/bash

read -p "Continue to activate the virtual environment? (y/n): " choice
if [ "$choice" != "y" ]; then
  echo "Script execution stopped."
  exit 1
fi

source .venv/bin/activate
echo "Virtual environment activated: $VIRTUAL_ENV"

python3 -c "import os; os.chdir('./newscollector/articles')"
echo "Moved to the 'articles' directory."

# Check if any JSON files exist in the 'articles' directory
if ls *.json >/dev/null 2>&1; then
  # Count the number of JSON files
  json_count=$(ls -1 *.json | wc -l)
  echo "Found $json_count JSON file(s) in the 'articles' directory."

  if [ "$json_count" -gt 20 ]; then
    read -p "More than 20 JSON files found. Continue to remove duplicates? (y/n): " choice
    if [ "$choice" != "y" ]; then
      echo "Script execution stopped."
      exit 1
    fi

    python3 remove_duplicate.py
    echo "remove_duplicate.py executed."

    read -p "Continue to archive articles? (y/n): " choice
    if [ "$choice" != "y" ]; then
      echo "Script execution stopped."
      exit 1
    fi

    python3 archive.py
    echo "archive.py executed."

    read -p "Continue to process articles? (y/n): " choice
    if [ "$choice" != "y" ]; then
      echo "Script execution stopped."
      exit 1
    fi

    python3 -c "import os; os.chdir('..'); os.system('python3 process.py .')"
    echo "process.py executed."
  else
    read -p "No need to remove duplicates. Continue to archive articles? (y/n): " choice
    if [ "$choice" != "y" ]; then
      echo "Script execution stopped."
      exit 1
    fi

    python3 archive.py
    echo "archive.py executed."

    read -p "Continue to navigate back to the parent directory? (y/n): " choice
    if [ "$choice" != "y" ]; then
      echo "Script execution stopped."
      exit 1
    fi

    python3 -c "import os; os.chdir(os.path.dirname(os.getcwd()))"
    echo "Moved back to the parent directory."

    read -p "Continue to run the 'newscollector.py' script? (y/n): " choice
    if [ "$choice" != "y" ]; then
      echo "Script execution stopped."
      exit 1
    fi

    python3 -c "import os; os.chdir(os.path.join(os.path.dirname(os.getcwd()), 'newscollector', 'newscollector')); os.system('python3 newscollector.py')"
    echo "newscollector.py script executed."
  fi
else
  echo "No JSON files found in the 'articles' directory."
  echo "Skipping duplicate removal and archiving steps."

  read -p "Continue to navigate back to run newscollector.py? (y/n): " choice
  if [ "$choice" != "y" ]; then
    echo "Script execution stopped."
    exit 1
  fi

  python3 -c "import os; os.chdir(os.path.join(os.path.dirname(os.getcwd()), 'newscollector', 'newscollector')); os.system('python3 newscollector.py')"
  echo "newscollector.py script executed."
fi

echo "Script execution completed."
