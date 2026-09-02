
import os
import io
import torch
import pyiqa
from PIL import Image
from transformers import AutoModelForCausalLM
import math

# Environment variable pointing to the local Q-Align (q-future/one-align) weights.
# The transformers cache location is left to the user's environment (HF_HOME / TRANSFORMERS_CACHE).
QALIGN_PATH_ENV = 'QALIGN_PATH'

class IQAScore:
    """
    Image Quality Assessment class that implements multiple IQA metrics.
    
    This class provides methods to evaluate image quality using state-of-the-art
    IQA models like QAlign, MANIQA, MUSIQ, CLIPIQA, and NIQE.
    """
    
    def __init__(self, device='cuda', score_weight=None, qalign_path=None):
        """
        Initialize the IQA Score calculator.

        Args:
            device (str): Computing device ('cuda' or 'cpu').
            score_weight (list, optional): Weight for each metric [qalign, maniqa, musiq, clipiqa, niqe].
                                          Defaults to [1,1,1,1,1].
            qalign_path (str, optional): Local directory of the Q-Align weights. Falls back to the
                                         QALIGN_PATH environment variable.
        """
        self.device = device
        self.score_weight = score_weight if score_weight is not None else [1, 1, 1, 1, 1]

        # Path to the QAlign model
        qalign_path = qalign_path or os.environ.get(QALIGN_PATH_ENV)
        if not qalign_path:
            raise ValueError(
                f"Q-Align weights path is not set. Pass qalign_path=... or set the {QALIGN_PATH_ENV} "
                "environment variable to the local directory of the q-future/one-align weights "
                "(with the config files from dependences/qalign applied, see docs/mrrhf_training.md).")

        # Initialize metrics based on weights
        print("Loading IQA metrics...")
        
        # Only load models with non-zero weights to save memory
        if self.score_weight:
            self.qalign_metric = AutoModelForCausalLM.from_pretrained(qalign_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")   
            self.maniqa_metric = pyiqa.create_metric('maniqa', device=self.device)
            self.musiq_metric = pyiqa.create_metric('musiq', device=self.device)
            self.clipiqa_metric = pyiqa.create_metric('clipiqa', device=self.device)
            self.niqe_metric = pyiqa.create_metric('niqe', device=self.device)
        
        # Mean and standard deviation for score normalization
        self.mean_std = {
            'maniqa': {'mean': 0.2092878860158957, 'std': 0.1245655639250264}, 
            'musiq': {'mean': 40.525839667740804, 'std': 16.94055156979329}, 
            'qalign': {'mean': 0.25762742254801686, 'std': 0.1830440687605635}
        }
        
        print("IQA metrics loaded successfully")
    
    def imread2tensor(self, img_source, rgb=False):
        """
        Convert various image sources to PIL Image.
        
        Args:
            img_source: Can be bytes, file path string, or PIL Image.
            rgb (bool): Whether to convert to RGB.
            
        Returns:
            PIL.Image: Loaded image.
            
        Raises:
            Exception: If the source type is not supported.
        """
        if isinstance(img_source, bytes):
            img = Image.open(io.BytesIO(img_source))
        elif isinstance(img_source, str):
            img = Image.open(img_source)
        elif isinstance(img_source, Image.Image):
            img = img_source
        else:
            raise Exception("Unsupported source type")
            
        if rgb:
            img = img.convert('RGB')
            
        return img

    def preprocess_image(self, source_image_path):
        """
        Preprocess the image for quality assessment.
        
        Args:
            source_image_path (str or bytes or PIL.Image): Image to preprocess.
            
        Returns:
            PIL.Image: Preprocessed image.
        """
        return self.imread2tensor(source_image_path)

    def normalize_iqa_score(self, score, metric):
        """
        Normalize an IQA score using pre-computed mean and standard deviation.
        
        Args:
            score (float): Raw score.
            metric (str): Metric name ('maniqa', 'musiq', or 'qalign').
            
        Returns:
            float: Normalized score (z-score).
        """
        mean = self.mean_std[metric]['mean']
        std = self.mean_std[metric]['std']
        return (score - mean) / std

    def get_iqa_score(self, source_image_path):
        """
        Calculate quality scores for an image using multiple metrics.
        
        Args:
            source_image_path (str): Path to the image.
            eval (bool): Whether in evaluation mode.
            is_score_weight (bool): Whether to use weighted scoring.
            
        Returns:
            list or dict: List of scores [qalign, maniqa, musiq, clipiqa, niqe] or
                         dict with combined score and individual scores if is_score_weight=True.
        """
        with torch.no_grad():
            source_image = self.preprocess_image(source_image_path)
            
            # Calculate scores using each metric
            qalign_score = self.qalign_metric.score([source_image], task_="quality", input_="image").item()
            maniqa_score = self.maniqa_metric(source_image).item()
            musiq_score = self.musiq_metric(source_image).item()
            clipiqa_score = self.clipiqa_metric(source_image).item()
            niqe_score = self.niqe_metric(source_image).item()
            
            return [qalign_score, maniqa_score, musiq_score, clipiqa_score, math.exp(-niqe_score / 10.0)]