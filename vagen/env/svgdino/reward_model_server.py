# simple_reward_server.py
# A simple local server that handles SVG scoring requests

import json
import time
import logging
import argparse
from flask import Flask, request, jsonify

# Import SVG scoring related modules
# Note: Assuming these modules are installed in your environment
from vagen.env.svg.score import calculate_total_score
from vagen.env.svg.svg_utils import process_and_rasterize_svg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Global variable
score_config = None

def compute_svg_score(gt_svg_code, gen_svg_code):
    """Compute SVG score"""
    try:
        # Process the SVG code and generate images
        _, gt_image = process_and_rasterize_svg(gt_svg_code)
        _, gen_image = process_and_rasterize_svg(gen_svg_code)
        
        # Calculate score
        scores = calculate_total_score(
            gt_im=gt_image,
            gen_im=gen_image,
            gt_code=gt_svg_code,
            gen_code=gen_svg_code,
            score_config=score_config
        )
        
        return scores
    
    except Exception as e:
        logger.error(f"Error occurred while computing SVG score: {str(e)}")
        return {"error": str(e)}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "SVG scoring service is running normally"}), 200

@app.route('/compute_score', methods=['POST'])
def score_endpoint():
    """SVG scoring endpoint"""
    start_time = time.time()
    
    # Retrieve request data
    data = request.json
    if not data:
        return jsonify({"error": "Invalid request data"}), 400
    
    # Extract SVG code
    gt_svg_code = data.get('gt_svg_code')
    gen_svg_code = data.get('gen_svg_code')
    
    if not gt_svg_code or not gen_svg_code:
        return jsonify({"error": "Missing required parameters: 'gt_svg_code' and 'gen_svg_code'"}), 400
    
    # Compute score
    result = compute_svg_score(gt_svg_code, gen_svg_code)
    
    # Log processing time
    process_time = time.time() - start_time
    logger.info(f"Request processed in: {process_time:.4f} seconds")
    
    # Add processing time to the response
    if isinstance(result, dict) and "error" not in result:
        result["process_time"] = process_time
    
    return jsonify(result)

def main():
    """Main function"""
    global score_config
    
    parser = argparse.ArgumentParser(description='Local SVG scoring server')
    parser.add_argument('--port', type=int, default=5000, help='Local server port (default: 5000)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Local server host (default: 127.0.0.1)')
    parser.add_argument('--model-size', type=str, default='small', help='Model size (default: small)')
    parser.add_argument('--dino-only', action='store_true', help='Use DINO scoring only')
    parser.add_argument('--dino-weight', type=float, help='DINO scoring weight')
    parser.add_argument('--structural-weight', type=float, help='Structural scoring weight')
    parser.add_argument('--color-weight', type=float, help='Color scoring weight')
    parser.add_argument('--code-weight', type=float, help='Code scoring weight')
    
    args = parser.parse_args()
    
    # Set score configuration
    score_config = {
        "model_size": args.model_size,
        "dino_only": args.dino_only,
    }
    
    # Add optional weights
    if args.dino_weight is not None:
        score_config["dino_weight"] = args.dino_weight
    if args.structural_weight is not None:
        score_config["structural_weight"] = args.structural_weight
    if args.color_weight is not None:
        score_config["color_weight"] = args.color_weight
    if args.code_weight is not None:
        score_config["code_weight"] = args.code_weight
    
    logger.info(f"Score configuration: {score_config}")
    
    # Start Flask application
    logger.info(f"Starting Flask server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down server...")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
