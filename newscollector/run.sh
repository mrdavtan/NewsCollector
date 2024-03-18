#!/bin/bash

read -p "Continue to activate the virtual environment? (y/n): " choice
if [ "$choice" != "y" ]; then
  echo "Script execution stopped."
  exit 1
fi

source .venv/bin/activate
echo "Virtual environment activated."

read -p "Continue to navigate to the 'newscollector' directory? (y/n): " choice
if [ "$choice" != "y" ]; then
  echo "Script execution stopped."
  exit 1
fi

cd newscollector/
echo "Moved to the 'newscollector' directory."

read -p "Continue to navigate to the 'articles' directory? (y/n): " choice
if [ "$choice" != "y" ]; then
  echo "Script execution stopped."
  exit 1
fi

cd articles/
echo "Moved to the 'articles' directory."

read -p "Continue to remove duplicate articles? (y/n): " choice
if [ "$choice" != "y" ]; then
  echo "Script execution stopped."
  exit 1
fi

python3 remove_duplicate.py
echo "Duplicate articles removed."

read -p "Continue to archive articles? (y/n): " choice
if [ "$choice" != "y" ]; then
  echo "Script execution stopped."
  exit 1
fi

python3 archive.py
echo "Articles archived."

read -p "Continue to navigate back to the 'newscollector' directory? (y/n): " choice
if [ "$choice" != "y" ]; then
  echo "Script execution stopped."
  exit 1
fi

cd ..
echo "Moved back to the 'newscollector' directory."

read -p "Continue to run the 'newscollector.py' script? (y/n): " choice
if [ "$choice" != "y" ]; then
  echo "Script execution stopped."
  exit 1
fi

python3 newscollector.py
echo "newscollector.py script executed."

echo "Script execution completed."


