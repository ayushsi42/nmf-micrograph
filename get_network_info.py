#!/usr/bin/env python3
"""
Helper script to display network information for accessing the web UI
"""

import socket

def get_local_ip():
    """Get the local IP address."""
    try:
        # Create a socket to find local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "Unable to determine"

if __name__ == "__main__":
    local_ip = get_local_ip()
    
    print("=" * 60)
    print("🌐 Microstructure Segmentation Web UI")
    print("=" * 60)
    print()
    print("Access the application at:")
    print()
    print(f"  📱 From this computer:")
    print(f"     http://localhost:7861")
    print()
    print(f"  🌍 From other devices on your WiFi network:")
    print(f"     http://{local_ip}:7861")
    print()
    print("=" * 60)
    print()
    print("To start the web UI, run:")
    print("  python app.py")
    print()
