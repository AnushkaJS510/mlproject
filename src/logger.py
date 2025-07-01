import logging
import os
from datetime import datetime

# Define the unique log file name using the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the path to the logs directory. This is the directory where logs will be stored.
# os.getcwd() gets the current working directory (D:\projectss\mlproject)
# "logs" is the subfolder name
LOGS_DIR = os.path.join(os.getcwd(), "logs")

# Create the logs directory if it doesn't already exist.
# The exist_ok=True argument prevents an error if the directory already exists.
os.makedirs(LOGS_DIR, exist_ok=True)

# Define the full path for the log file, combining the directory and the filename.
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

# Configure the basic settings for the logging module.
# filename: Specifies the file to which log messages will be written.
# format: Defines the layout of log messages.
#         - %(asctime)s: Human-readable time when the LogRecord was created.
#         - %(lineno)d: Line number where the logging call was made.
#         - %(name)s: Name of the logger (default is 'root').
#         - %(levelname)s: Text logging level for the message (e.g., INFO, WARNING).
#         - %(message)s: The logged message itself.
# level: Sets the minimum logging level. Messages at or above this level will be processed.
#        logging.INFO means INFO, WARNING, ERROR, CRITICAL messages will be logged.
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# This block ensures that the logging.info message is only executed when the script
# is run directly, not when it's imported as a module into another script.
if __name__ == "__main__":
    # Log an informational message to the configured log file.
    logging.info("Logging has started.")
    # You can add more logging calls here to test
    # logging.warning("This is a warning message.")
    # logging.error("This is an error message.")
