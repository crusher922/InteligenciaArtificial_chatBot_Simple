import json
import logging
from pathlib import Path
from datetime import datetime
import shutil


BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
CHECKPOINTS_DIR = LOGS_DIR / "checkpoints"

TRAINING_HISTORY_LOG = LOGS_DIR / "training_history.log"
EVENTS_LOG = LOGS_DIR / "events.log"
ERRORS_LOG = LOGS_DIR / "errors.log"

HYPERPARAMETERS_FILE = LOGS_DIR / "hyperparameters.json"
TRAIN_IO_FILE = LOGS_DIR / "train_io.json"
MODEL_STATUS_FILE = LOGS_DIR / "model_status.json"
VALIDATION_RESULTS_FILE = LOGS_DIR / "validation_results.json"


def ensure_log_dirs():
    LOGS_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)


def setup_logger(name, log_file, level=logging.INFO):
    ensure_log_dirs()
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


training_logger = setup_logger("training_logger", TRAINING_HISTORY_LOG)
events_logger = setup_logger("events_logger", EVENTS_LOG)
errors_logger = setup_logger("errors_logger", ERRORS_LOG, level=logging.ERROR)


def save_json(path, data):
    ensure_log_dirs()
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def append_training_history(message):
    training_logger.info(message)


def log_event(message):
    events_logger.info(message)


def log_error(message):
    errors_logger.error(message)


def update_model_status(status, model_path=None, checkpoint_available=False):
    data = {
        "status": status,
        "last_update": datetime.now().isoformat(),
        "model_path": str(model_path) if model_path else None,
        "checkpoint_available": checkpoint_available
    }
    save_json(MODEL_STATUS_FILE, data)


def save_hyperparameters(data):
    save_json(HYPERPARAMETERS_FILE, data)


def save_train_io(data):
    save_json(TRAIN_IO_FILE, data)


def save_validation_results(data):
    save_json(VALIDATION_RESULTS_FILE, data)


def create_checkpoint(source_file, checkpoint_name):
    ensure_log_dirs()
    destination = CHECKPOINTS_DIR / checkpoint_name
    shutil.copy2(source_file, destination)
    return destination