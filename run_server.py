#!/usr/bin/env python3
"""
Run the Chandigarh Autonomous Driving Simulator server.
"""

import os
import sys
import subprocess
import webbrowser
import time

def main():
    """Start the server and open browser"""
    print("=" * 60)
    print("🛣️  Chandigarh Autonomous Driving Simulator")
    print("=" * 60)
    print()
    print("Starting server...")
    
    # Start the server
    server_process = subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for server to start
    print("Waiting for server to initialize...")
    time.sleep(3)
    
    # Open browser
    print("🌐 Opening browser at http://localhost:8000")
    webbrowser.open("http://localhost:8000")
    
    print()
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        # Show server output
        for line in server_process.stdout:
            print(line, end='')
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server_process.terminate()
        server_process.wait()
        print("Server stopped.")

if __name__ == "__main__":
    main()