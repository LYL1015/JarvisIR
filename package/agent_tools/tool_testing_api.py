import os
import time
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify
# Import model loaders and predictors
from .RIDCP.inference import load_ridcp_model, ridcp_predict
from .SCUNet.inference import load_scu_model, scu_predict
from .Retinexformer.inference import load_retinexformer_model, retinexformer_predict
from .img2img_turbo.inference import load_turbo_model, turbo_predict
from .ESRGAN.inference import load_esrgan_model, esrgan_predict
from .IDT.inference import load_idt_model, idt_predict
from .iqa_reward import IQAScore
# Configure environment variables
os.environ["BASICSR_JIT"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Initialize Flask application
app = Flask(__name__)

# Global variables
models = {}
iqa = IQAScore()

class ModelTester:
    """
    Model testing service for image restoration models.
    
    This class manages model loading, image processing, and quality assessment.
    """
    
    def __init__(self, output_base_dir="datasets/tmp_result"):
        """
        Initialize the model tester.
        
        Args:
            output_base_dir (str): Base directory for storing results.
        """
        self.output_base_dir = output_base_dir
        self.models = {}
        self.iqa = IQAScore()
        self.model_loaders = {
            'scunet': (load_scu_model, scu_predict),
            'retinexformer_lolv2': (lambda: load_retinexformer_model('LOLV2'), retinexformer_predict),
            'retinexformer_fivek': (lambda: load_retinexformer_model('FiveK'), retinexformer_predict),
            'turbo_night': (lambda: load_turbo_model('night'), turbo_predict),
            'turbo_rain': (lambda: load_turbo_model('rain'), turbo_predict),
            'turbo_snow': (lambda: load_turbo_model('snow'), turbo_predict),
            'real_esrgan': (load_esrgan_model, esrgan_predict),
            'ridcp': (load_ridcp_model, ridcp_predict),
            'idt': (load_idt_model, idt_predict)
        }
    
    def load_models(self, model_names):
        """
        Load specified models into memory.
        
        Args:
            model_names (list): List of model names to load.
        """
        print(f"Loading models: {', '.join(model_names)}")
        self.models = {}
        
        for model_name in model_names:
            if model_name in self.model_loaders:
                loader_fn = self.model_loaders[model_name][0]
                self.models[model_name] = loader_fn()
                print(f"Loaded {model_name}")
            else:
                print(f"Unknown model: {model_name}")
        
        print(f"Finished loading {len(self.models)} models")
        
    def resize_image(self, img_path, output_dir, target_size=(256, 256)):
        """
        Resize input image to a standard size.
        
        Args:
            img_path (str): Path to the input image.
            output_dir (str): Directory to save the resized image.
            target_size (tuple): Target resolution (width, height).
            
        Returns:
            str: Path to the resized image.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        with Image.open(img_path) as img:
            # Ensure consistent color mode
            img = img.convert('RGB')
            # Use high-quality resampling
            img = img.resize(target_size, Image.LANCZOS)
            
            # Generate output filename
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, f"{img_name}.png")
            
            # Save the resized image
            img.save(save_path, format='PNG')
        
        return save_path
    
    def process_image_with_models(self, model_list, img_path, output_dir):
        """
        Process an image with a sequence of models.
        
        Args:
            model_list (list): List of model names to apply in sequence.
            img_path (str): Path to the input image.
            output_dir (str): Directory to save the processed images.
            
        Returns:
            str: Path to the final processed image.
        """
        # Resize input image
        img_path = self.resize_image(img_path, output_dir)
        
        # Apply each model in sequence
        for model_name in model_list:
            if model_name not in self.models:
                print(f"Model {model_name} not loaded, skipping")
                continue
                
            # Get the predict function for this model
            _, predict_fn = self.model_loaders[model_name]
            
            # Process the image with the current model
            img_path = predict_fn(self.models[model_name], img_path, output_dir)
            print(f"Applied {model_name}, saved result to {img_path}")
        
        return img_path
    
    def create_output_dir(self):
        """
        Create a unique output directory based on current timestamp.
        
        Returns:
            str: Path to the created output directory.
        """
        timestamp = int(time.time())
        output_dir = os.path.join(self.output_base_dir, f"{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def process_request(self, img_path, model_list):
        """
        Process an image with the specified models and evaluate the result.
        
        Args:
            img_path (str): Path to the input image.
            model_list (list): List of model names to apply.
            
        Returns:
            dict: Dictionary with output path and quality score.
            
        Raises:
            FileNotFoundError: If the input image doesn't exist.
        """
        # Verify the image path
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        # Create a unique output directory
        output_dir = self.create_output_dir()
        
        # Process the image
        final_output = self.process_image_with_models(model_list, img_path, output_dir)
        
        # Evaluate the result
        score = self.iqa.get_iqa_score(final_output)
        
        return {
            "output_path": final_output,
            "score": score
        }

# Initialize the model tester
model_tester = None

@app.route('/process_image', methods=['POST'])
def process_image():
    """
    API endpoint for processing an image with specified models.
    
    Expects a JSON payload with:
    - img_path: Path to the input image
    - models: List of model names to apply
    
    Returns:
    - JSON with output_path and score
    """
    global model_tester
    
    # Parse request data
    data = request.get_json()
    img_path = data.get('img_path')
    models_to_use = data.get('models', [])
    
    # Validate input
    if not img_path:
        return jsonify({"error": "Missing image path"}), 400
    
    if not models_to_use:
        return jsonify({"error": "No models specified"}), 400
    
    try:
        # Process the image
        result = model_tester.process_request(img_path, models_to_use)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

def start_server(host='0.0.0.0', port=5010, model_names=None):
    """
    Start the API server with specified models.
    
    Args:
        host (str): Host address to bind the server.
        port (int): Port to listen on.
        model_names (list): List of model names to load. If None, loads a default set.
    """
    global model_tester
    
    # Initialize the model tester
    model_tester = ModelTester()
    
    # Define default models if none specified
    if model_names is None:
        model_names = [
            'scunet', 'real_esrgan', 'ridcp', 'idt',
            'turbo_rain', 'turbo_night',
            'retinexformer_lolv2', 'retinexformer_fivek'
        ]
    
    # Load the models
    model_tester.load_models(model_names)
    
    # Start the Flask application
    print(f"Starting API server at http://{host}:{port}")
    app.run(host=host, port=port)

if __name__ == '__main__':
    # Start the server with default settings
    start_server() 