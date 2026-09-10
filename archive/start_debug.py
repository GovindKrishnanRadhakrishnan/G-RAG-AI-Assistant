import uvicorn
from main import app
import logging
import sys

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="127.0.0.1", port=9000, log_level="debug")
    except Exception as e:
        print("EXCEPTION CAUGHT:", e)
