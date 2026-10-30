from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")

SAMPLE_CSV = b"""country,meas_a,meas_b,meas_c
CountryA,10,20,30
CountryB,40,50,60
CountryC,70,80,90
CountryD,15,25,35
CountryE,85,95,105
"""

SAMPLE_CSV_2 = b"""country,meas_a,meas_b,meas_c
CountryX,100,200,300
CountryY,400,500,600
CountryZ,700,800,900
"""


def test_clustering_result_persistence_on_rerun():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.file_uploader[0].upload("data.csv", SAMPLE_CSV).run()
    assert len(at.button) >= 1
    # Click button — after file upload, selectbox[0] is the algorithm selector
    at.button[0].click().run()
    # Now there are 2 selectboxes: algorithm_choice (0) + selected_k_solution (1)
    assert len(at.selectbox) == 2
    assert len(at.download_button) == 2

    # Rerun app without re-clicking button
    at.run()
    assert len(at.selectbox) == 2
    assert len(at.download_button) == 2
    assert "clustering_result" in at.session_state


def test_change_selected_cluster_k():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.file_uploader[0].upload("data.csv", SAMPLE_CSV).run()
    at.button[0].click().run()

    # selectbox[0] = algorithm_choice, selectbox[1] = selected_k_solution
    # Initial k is 2
    assert at.selectbox[1].value == 2
    # Switch to k = 3
    at.selectbox[1].select(3).run()
    assert at.selectbox[1].value == 3
    assert len(at.download_button) == 2
    assert at.session_state["selected_k_solution"] == 3


def test_download_buttons_persist_and_contain_data():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.file_uploader[0].upload("data.csv", SAMPLE_CSV).run()
    at.button[0].click().run()

    # Click first download button (cluster assignments)
    at.download_button[0].click().run()
    assert len(at.selectbox) == 2  # algorithm_choice + selected_k_solution
    assert len(at.download_button) == 2

    # Click second download button (metrics)
    at.download_button[1].click().run()
    assert len(at.selectbox) == 2
    assert len(at.download_button) == 2


def test_feature_change_invalidates_stale_results():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.file_uploader[0].upload("data.csv", SAMPLE_CSV).run()
    at.button[0].click().run()
    assert len(at.selectbox) == 2  # algorithm_choice + selected_k_solution

    # Modify selected features: unselect 'meas_c'
    at.multiselect[0].unselect("meas_c").run()

    # Stale results must be invalidated — selected_k_solution selectbox disappears
    # algorithm_choice selectbox (index 0) remains
    assert len(at.selectbox) == 1
    assert len(at.download_button) == 0
    assert "clustering_result" not in at.session_state


def test_uploading_new_file_invalidates_stale_results():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.file_uploader[0].upload("data.csv", SAMPLE_CSV).run()
    at.button[0].click().run()
    assert len(at.selectbox) == 2  # algorithm_choice + selected_k_solution

    # Upload different dataset
    at.file_uploader[0].upload("data_new.csv", SAMPLE_CSV_2).run()

    # Previous results must be cleared — selected_k_solution selectbox disappears
    # algorithm_choice selectbox (index 0) remains
    assert len(at.selectbox) == 1
    assert len(at.download_button) == 0
    assert "clustering_result" not in at.session_state


def test_country_to_cluster_correspondence_preserved():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.file_uploader[0].upload("data.csv", SAMPLE_CSV).run()
    at.button[0].click().run()

    cached = at.session_state["clustering_result"]
    expected_countries = ["CountryA", "CountryB", "CountryC", "CountryD", "CountryE"]
    assert cached["country_names"] == expected_countries
    assert len(cached["labels_by_k"][2]) == len(expected_countries)
    assert len(cached["labels_by_k"][3]) == len(expected_countries)
    assert list(cached["analysis_raw"]["country"]) == expected_countries


def test_missing_or_corrupted_session_state_handled_safely():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.file_uploader[0].upload("data.csv", SAMPLE_CSV).run()
    at.button[0].click().run()
    assert "clustering_result" in at.session_state

    # Corrupt or delete state key manually
    del at.session_state["clustering_result"]
    at.run()

    # Application should remain runnable and not crash
    # Only the algorithm_choice selectbox should remain (no cluster-k selector)
    assert len(at.selectbox) == 1
    assert len(at.download_button) == 0
    assert len(at.button) >= 1


def test_missingness_slider_and_exclusion_audit_display():
    """Verify that rows exceeding missingness threshold produce an exclusion warning and audit download."""
    csv_with_sparse = b"""country,meas_a,meas_b,meas_c
CountryA,10,20,30
CountryB,40,50,60
CountryC,70,80,90
CountrySparse,None,None,35
CountryD,15,25,35
"""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.file_uploader[0].upload("data_sparse.csv", csv_with_sparse).run()

    # Default slider is 30% -> CountrySparse has 2/3 missing (66.7%) -> excluded
    assert len(at.warning) >= 2  # Methodological warning + exclusion warning
    assert "exceeded the 30% missingness threshold" in at.warning[1].value
    # Exclusion audit download button should be present
    assert len(at.download_button) == 1
    assert at.download_button[0].key == "dl_exclusions"

    # Now click run clustering
    at.button[0].click().run()
    # Results should be displayed, and total download buttons = 3 (1 exclusion audit + 2 result downloads)
    assert len(at.download_button) == 3
    cached = at.session_state["clustering_result"]
    assert "CountrySparse" not in cached["country_names"]
    assert len(cached["country_names"]) == 4
    assert cached["algorithm_choice"] == "K-Means (Euclidean)"


def test_missingness_slider_change_invalidates_state():
    """Verify that changing the missingness slider invalidates stale clustering results."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.file_uploader[0].upload("data.csv", SAMPLE_CSV).run()
    at.button[0].click().run()
    assert "clustering_result" in at.session_state
    assert len(at.selectbox) == 2  # algorithm_choice + selected_k_solution

    # Adjust missingness slider from 30 to 50
    at.slider[0].set_value(50).run()

    # Stale clustering results must be cleared — selected_k_solution selectbox disappears
    # algorithm_choice selectbox (index 0) remains
    assert "clustering_result" not in at.session_state
    assert len(at.selectbox) == 1

