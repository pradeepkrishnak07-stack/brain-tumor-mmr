"""
Test suite for Streamlit Web Application (app.py)
Validates headless execution, initial state, and sample phantom pipeline execution.
"""

import os
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))


def test_streamlit_app_initial_load():
    """Verify that app.py loads cleanly without any unhandled exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"App threw unexpected exception: {at.exception}"
    assert len(at.info) >= 1  # "Please upload a brain MRI..." prompt is shown


def test_streamlit_app_sample_phantom_execution():
    """Verify that selecting a sample phantom executes the full MMR pipeline without error."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Select 'Load Verified Sample Phantom'
    at.radio[0].set_value("Load Verified Sample Phantom").run()
    assert not at.exception, f"App threw exception when processing sample phantom: {at.exception}"

    # Verify that tabs and metrics were rendered
    assert len(at.tabs) == 5
