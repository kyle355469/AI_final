#!/bin/bash

echo "Running legal_builder.py..."
python3 data_builder/legal_builder.py
if [ $? -ne 0 ]; then
  echo "legal_builder.py failed"
  exit 1
fi
echo "Finished legal_builder.py"

echo "Running violation_builder.py..."
python3 data_builder/violation_builder.py
if [ $? -ne 0 ]; then
  echo "violation_builder.py failed"
  exit 1
fi
echo "Finished violation_builder.py"

echo "----------------------------------"
echo "All builders completed!"
