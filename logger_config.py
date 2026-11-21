
import logging
import uuid
from pathlib import Path
from datetime import datetime, timezone
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler


# Store request ID
request_id_var = ContextVar("request_id", default="SYSTEM")


# Add request ID to each log record
class RequestIDFilter(logging.Filter):

    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


# Format log records as readable text
class TextFormatter(logging.Formatter):

    def format(self, record):

        timestamp = datetime.fromtimestamp(
            record.created, tz=timezone.utc
        ).isoformat()

        component = getattr(record, "component", record.name)
        event = getattr(record, "event", "general")
        details = getattr(record, "details", {})

        log_message = (
            f"{timestamp} | "
            f"{record.levelname} | "
            f"Request ID: {record.request_id} | "
            f"Component: {component} | "
            f"Event: {event} | "
            f"Message: {record.getMessage()} | "
            f"Details: {details}"
        )

        if record.exc_info:
            log_message += "\n" + self.formatException(record.exc_info)

        return log_message


# Configure logging
def setup_logging():

    # Create logs directory
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)

    formatter = TextFormatter()
    request_filter = RequestIDFilter()

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(formatter)
    console.addFilter(request_filter)

    # Application log handler
    app_file = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8"
    )
    app_file.setLevel(logging.DEBUG)
    app_file.setFormatter(formatter)
    app_file.addFilter(request_filter)

    # Error log handler
    error_file = RotatingFileHandler(
        log_dir / "error.log",
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8"
    )
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(formatter)
    error_file.addFilter(request_filter)

    # Configure application logger
    logger = logging.getLogger("adaptive_rag")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Clear existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # Add handlers
    logger.addHandler(console)
    logger.addHandler(app_file)
    logger.addHandler(error_file)

    return logger


# Get logger for the project
def get_logger(name="adaptive_rag"):
    return logging.getLogger(name)


# Start tracking a new question
def start_request():
    request_id = uuid.uuid4().hex[:12]
    return request_id_var.set(request_id)


# Finish tracking the question
def end_request(token):
    request_id_var.reset(token)


# Record a detailed pipeline event
def log_event(level, component, event, message, **details):

    logger.log(
        level,
        message,
        extra={
            "component": component,
            "event": event,
            "details": details
        }
    )


# Initialize logging
logger = setup_logging()
