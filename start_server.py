#!/usr/bin/env python3
"""
Fast startup script for IShowSpeed Analytics Server

This script uses Redis caching to speed up server startup and performance.
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Start IShowSpeed Analytics Server with Redis caching')
    
    # Use PORT environment variable if available (for Render deployment)
    default_port = int(os.environ.get('PORT', 5000))
    
    parser.add_argument('--port', type=int, default=default_port,
                        help=f'Port to run server on (default: {default_port})')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')

    args = parser.parse_args()

    # Set environment variables
    os.environ['FLASK_PORT'] = str(args.port)
    os.environ['FLASK_HOST'] = args.host
    if args.debug:
        os.environ['FLASK_DEBUG'] = '1'

    # Start server with Redis caching
    print("Starting server with Redis caching...")
    try:
        import redis
        from app_redis_cache import main as run_app
    except ImportError:
        print("Redis not installed. Install with: pip install redis")
        sys.exit(1)

    print(f"Server will start on {args.host}:{args.port}")
    print("Data processing will happen on-demand when endpoints are accessed.")

    try:
        run_app()
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    main()
