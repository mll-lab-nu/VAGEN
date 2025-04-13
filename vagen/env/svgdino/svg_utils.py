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



# -------------- convert raw svg code into image --------------

def is_valid_svg(svg_text):
    try:
        svgstr2paths(svg_text)
        return True
    except Exception as e:
        print(f"Invalid SVG: {str(e)}")
        return False

def clean_svg(svg_text, output_width=None, output_height=None):
    soup = BeautifulSoup(svg_text, 'xml') # Read as soup to parse as xml
    svg_bs4 = soup.prettify() # Prettify to get a string

    # Store the original signal handler
    import signal
    original_handler = signal.getsignal(signal.SIGALRM)
    
    try:
        # Set a timeout to prevent hanging
        def timeout_handler(signum, frame):
            raise TimeoutError("SVG processing timed out")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        
        # Try direct conversion without BeautifulSoup
        svg_cairo = cairosvg.svg2svg(svg_bs4, output_width=output_width, output_height=output_height).decode()
        
    except TimeoutError:
        print("SVG conversion timed out, using fallback method")
        svg_cairo = """<svg></svg>"""
    finally:
        # Always cancel the alarm and restore original handler, regardless of success or failure
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)
        
    svg_clean = "\n".join([line for line in svg_cairo.split("\n") if not line.strip().startswith("<?xml")]) # Remove xml header
    return svg_clean


def use_placeholder():
    VOID_SVF = "Exception Placeholder"
    return VOID_SVF
 
def process_and_rasterize_svg(svg_string, resolution=256, dpi = 128, scale=2):
    try:
        svgstr2paths(svg_string) # This will raise an exception if the svg is still not valid
        out_svg = svg_string
    except:
        try:
            svg = clean_svg(svg_string)
            svgstr2paths(svg) # This will raise an exception if the svg is still not valid
            out_svg = svg
        except Exception as e:
            out_svg = use_placeholder()

    raster_image = rasterize_svg(out_svg, resolution, dpi, scale)
    return out_svg, raster_image

def rasterize_svg(svg_string, resolution=224, dpi = 128, scale=2):
    try:
        svg_raster_bytes = cairosvg.svg2png(
            bytestring=svg_string,
            background_color='white',
            output_width=resolution, 
            output_height=resolution,
            dpi=dpi,
            scale=scale) 
        svg_raster = Image.open(BytesIO(svg_raster_bytes))
    except: 
        try:
            svg = clean_svg(svg_string)
            svg_raster_bytes = cairosvg.svg2png(
                bytestring=svg,
                background_color='white',
                output_width=resolution, 
                output_height=resolution,
                dpi=dpi,
                scale=scale) 
            svg_raster = Image.open(BytesIO(svg_raster_bytes))
        except:
            svg_raster = Image.new('RGB', (resolution, resolution), color = 'white')
    return svg_raster




def load_svg_dataset(data_dir, dataset_name, split):
    """Load the SVG dataset from local files or HuggingFace"""
    dataset_folder = os.path.join(data_dir, f"{dataset_name.replace('/', '-')}")
    local_path = os.path.join(dataset_folder, split)
    
    if os.path.exists(local_path):
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(local_path)
            return dataset
        except Exception as e:
            print(f"Error loading from simplified path: {e}")
    
    try:
        print(f"Downloading dataset from HuggingFace: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            dataset.save_to_disk(local_path)
            print(f"Saved dataset to {local_path} for future use")
        except Exception as e:
            print(f"Failed to save to simplified path: {e}")
            
        return dataset
    except Exception as e:
        raise ValueError(f"Failed to load dataset from HuggingFace: {e}")