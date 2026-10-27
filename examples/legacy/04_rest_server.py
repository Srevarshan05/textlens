"""
TextLens REST Server Example
============================
Demonstrates spinning up a full REST API server in 1 line of code.
"""

import textlens

if __name__ == "__main__":
    print("Starting TextLens REST API Server on port 8000...")
    print("Open http://localhost:8000/docs in your browser for Swagger API Documentation")
    textlens.serve(host="0.0.0.0", port=8000)
