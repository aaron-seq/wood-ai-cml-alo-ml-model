import requests
from typing import Optional, Dict, Any
import streamlit as st

API_BASE_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def score_cml_data(uploaded_file) -> Optional[Dict[str, Any]]:
    """Score CML data via API.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        API response as dict or None on error
    """
    try:
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        response = requests.post(
            f"{API_BASE_URL}/score-cml-data", files=files, timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to API. Make sure the API server is running on http://localhost:8000"
        )
        return None
    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None
