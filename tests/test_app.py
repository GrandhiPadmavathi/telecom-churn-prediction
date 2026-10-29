import pytest
from streamlit.testing.v1 import AppTest

from churn.config import ROOT, SETTINGS


@pytest.mark.skipif(not SETTINGS.paths.bundle.exists(), reason="needs a trained model (python -m churn.train)")
def test_app_renders_without_errors():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception
    assert at.metric[0].label == "Customers worth calling"
    at.sidebar.slider[0].set_value(0.6).run()
    assert not at.exception
