from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).parent


setup(
    name="thermal-viewer",
    version="1.0.0",
    description="Open-source thermal camera viewer — FLIR Boson, Lepton, InfiRay and other UVC cameras",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.9",
    py_modules=["thermal_viewer", "fps_bare"],
    install_requires=[
        "opencv-python>=4.7",
        "numpy>=1.24",
    ],
    entry_points={
        "console_scripts": [
            "thermal-viewer=thermal_viewer:main",
        ],
    },
    url="https://github.com/ibmua/thermal-viewer",
    project_urls={
        "Homepage": "https://github.com/ibmua/thermal-viewer",
        "Issues": "https://github.com/ibmua/thermal-viewer/issues",
    },
)
