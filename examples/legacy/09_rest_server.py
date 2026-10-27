"""Start the TextLens REST API and open http://127.0.0.1:8000/docs."""

import textlens


if __name__ == "__main__":
    textlens.serve(host="127.0.0.1", port=8000)
