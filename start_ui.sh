#!/bin/bash

echo "🔬 Starting Microstructure Segmentation Web UI..."
echo ""

# Display network info
python get_network_info.py

echo "Launching web interface..."
echo ""

# Launch the app
python app.py
