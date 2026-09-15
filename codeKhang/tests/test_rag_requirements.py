from pathlib import Path


def test_requirements_pin_google_generativeai_package_used_by_runtime_modules():
    requirements_text = (
        Path(__file__).resolve().parents[1] / "requirements.txt"
    ).read_text(encoding="utf-8")

    assert "google-generativeai" in requirements_text
    assert "google-genai" not in requirements_text
