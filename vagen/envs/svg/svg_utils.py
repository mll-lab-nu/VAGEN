from PIL import Image
import numpy as np
from bs4 import BeautifulSoup
import re
from svgpathtools import svgstr2paths
import cairosvg
import io
from io import BytesIO
import os
from datasets import Dataset, load_dataset


def is_valid_svg(svg_text):
    try:
        svgstr2paths(svg_text)
        return True
    except Exception:
        return False


def clean_svg(svg_text, output_width=None, output_height=None):
    soup = BeautifulSoup(svg_text, 'xml')
    svg_bs4 = soup.prettify()

    import signal
    original_handler = signal.getsignal(signal.SIGALRM)

    try:
        def timeout_handler(signum, frame):
            raise TimeoutError("SVG processing timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)

        svg_cairo = cairosvg.svg2svg(
            svg_bs4, output_width=output_width, output_height=output_height
        ).decode()

    except TimeoutError:
        svg_cairo = """<svg></svg>"""
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

    svg_clean = "\n".join(
        [line for line in svg_cairo.split("\n") if not line.strip().startswith("<?xml")]
    )
    return svg_clean


def process_and_rasterize_svg(svg_string, resolution=256, dpi=128, scale=2):
    try:
        svgstr2paths(svg_string)
        out_svg = svg_string
    except Exception:
        try:
            svg = clean_svg(svg_string)
            svgstr2paths(svg)
            out_svg = svg
        except Exception as e:
            out_svg = (
                '<svg width="{0}" height="{0}" xmlns="http://www.w3.org/2000/svg">'
                '<rect width="100%" height="100%" fill="white"/></svg>'
            ).format(resolution)

    raster_image = rasterize_svg(out_svg, resolution, dpi, scale)
    return out_svg, raster_image


def rasterize_svg(svg_string, resolution=224, dpi=128, scale=2):
    try:
        svg_raster_bytes = cairosvg.svg2png(
            bytestring=svg_string,
            background_color='white',
            output_width=resolution,
            output_height=resolution,
            dpi=dpi,
            scale=scale,
        )
        svg_raster = Image.open(BytesIO(svg_raster_bytes))
    except Exception:
        try:
            svg = clean_svg(svg_string)
            svg_raster_bytes = cairosvg.svg2png(
                bytestring=svg,
                background_color='white',
                output_width=resolution,
                output_height=resolution,
                dpi=dpi,
                scale=scale,
            )
            svg_raster = Image.open(BytesIO(svg_raster_bytes))
        except Exception:
            svg_raster = Image.new('RGB', (resolution, resolution), color='white')
    return svg_raster


def load_svg_dataset(data_dir, dataset_name, split):
    """Load the SVG dataset from local files or HuggingFace."""
    dataset_folder = os.path.join(data_dir, f"{dataset_name.replace('/', '-')}")
    local_path = os.path.join(dataset_folder, split)

    if os.path.exists(local_path):
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(local_path)
            return dataset
        except Exception as e:
            print(f"Error loading from local path: {e}")

    try:
        print(f"Downloading dataset from HuggingFace: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)

        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            dataset.save_to_disk(local_path)
            print(f"Saved dataset to {local_path} for future use")
        except Exception as e:
            print(f"Failed to save to local path: {e}")

        return dataset
    except Exception as e:
        raise ValueError(f"Failed to load dataset from HuggingFace: {e}")
