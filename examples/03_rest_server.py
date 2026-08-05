"""
TextLens Example 03: 1-Line REST Server Launch
"""

import textlens

def main():
    print("Launching TextLens REST API Microservice...")
    # Launches Uvicorn + FastAPI with automatic Swagger UI documentation at http://localhost:8000/docs
    textlens.serve(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
