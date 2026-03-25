"""
Data validation for pipeline inputs and outputs using pandera.
Validates YouTube URL CSV and processing metadata CSVs before use.
"""
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check
from pathlib import Path
from typing import Optional

from src.utils.helpers import get_logger

logger = get_logger(__name__)


# Schema for data/youtube_urls.csv
YOUTUBE_URL_SCHEMA = DataFrameSchema(
    {
        "category": Column(
            str,
            checks=Check(lambda s: s.str.strip().ne(""), element_wise=False),
            nullable=False,
        ),
        "url": Column(
            str,
            checks=Check(
                lambda s: s.str.contains(r"youtube\.com|youtu\.be", regex=True),
                element_wise=False,
                error="All URLs must be YouTube links",
            ),
            nullable=False,
        ),
        "description": Column(str, nullable=True, required=False),
    },
    coerce=True,
)

# Schema for ml_data/processing_metadata.csv
PROCESSING_METADATA_SCHEMA = DataFrameSchema(
    {
        "original_file": Column(str, nullable=False),
        "segment_file": Column(str, nullable=False),
        "category": Column(str, nullable=False),
        "segment_number": Column(int, checks=Check.ge(0), nullable=False),
        "duration_s": Column(
            float,
            checks=Check(lambda x: (x - 10.0).abs() < 0.5, element_wise=False),
            nullable=False,
        ),
        "dbfs": Column(float, checks=Check.ge(-100), nullable=False),
        "speech_percentage": Column(
            float, checks=[Check.ge(0), Check.le(100)], nullable=False
        ),
    },
    coerce=True,
)


def validate_youtube_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Validate the YouTube URL CSV against the expected schema.

    Args:
        csv_path: Path to youtube_urls.csv.

    Returns:
        Validated DataFrame, or None on validation failure.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.error("YouTube CSV not found: %s", csv_path)
        return None

    try:
        df = pd.read_csv(path)
        validated = YOUTUBE_URL_SCHEMA.validate(df)
        logger.info("YouTube CSV validated: %d rows, %d unique categories",
                    len(validated), validated["category"].nunique())
        return validated
    except pa.errors.SchemaError as exc:
        logger.error("YouTube CSV validation failed:\n%s", exc)
        return None
    except Exception as exc:
        logger.error("Could not read YouTube CSV: %s", exc)
        return None


def validate_processing_metadata(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Validate the processing metadata CSV after a pipeline run.

    Args:
        csv_path: Path to processing_metadata.csv.

    Returns:
        Validated DataFrame, or None on validation failure.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.error("Processing metadata not found: %s", csv_path)
        return None

    try:
        df = pd.read_csv(path)
        validated = PROCESSING_METADATA_SCHEMA.validate(df)
        logger.info(
            "Processing metadata validated: %d segments, %d categories",
            len(validated), validated["category"].nunique(),
        )
        return validated
    except pa.errors.SchemaError as exc:
        logger.error("Processing metadata validation failed:\n%s", exc)
        return None
    except Exception as exc:
        logger.error("Could not read processing metadata: %s", exc)
        return None
