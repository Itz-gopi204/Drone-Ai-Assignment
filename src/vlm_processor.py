"""
Vision Language Model (VLM) Processor for Drone Security Agent.

This module handles:
1. Video frame extraction using OpenCV
2. Image captioning using BLIP-2 VLM
3. Integration with the security analysis pipeline

VLM Choice: BLIP-2 over CLIP
- BLIP-2 generates detailed descriptive captions (e.g., "Blue Ford F150 entering gate")
- CLIP only does classification/matching (e.g., "truck: 95% confidence")
- For security analysis, we need rich descriptions, not just labels
"""

import os
import cv2
import base64
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass
import json

# Check for GPU availability
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

# BLIP-2 imports (optional - requires transformers and GPU)
BLIP_AVAILABLE = False
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from PIL import Image
    BLIP_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    pass

# For API-based VLM (GPT-4 Vision)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class VideoFrame:
    """Represents a single extracted video frame with metadata."""
    frame_id: int
    timestamp: datetime
    frame_data: any  # numpy array or PIL Image
    description: str = ""
    location: Dict = None
    telemetry: Dict = None


@dataclass
class VLMConfig:
    """Configuration for VLM processing."""
    provider: str = "simulated"  # "blip2", "gpt4v", "direct", "simulated"
    model_name: str = "Salesforce/blip2-opt-2.7b"
    frame_interval_seconds: int = 5  # Extract 1 frame every N seconds
    max_frames: int = 100  # Maximum frames to process
    direct_analysis: bool = False  # If True, send image directly to Vision LLM for full analysis


class VideoProcessor:
    """
    Handles video file processing and frame extraction using OpenCV.
    """

    def __init__(self, frame_interval_seconds: int = 5):
        self.frame_interval = frame_interval_seconds

    def get_video_info(self, video_path: str) -> Dict:
        """Get video metadata."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1))
        }
        cap.release()
        return info

    def extract_frames(self, video_path: str, max_frames: int = 100) -> Generator[VideoFrame, None, None]:
        """
        Extract frames from video at specified intervals.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract

        Yields:
            VideoFrame objects with frame data and metadata
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = max(cap.get(cv2.CAP_PROP_FPS), 1)
        frame_skip = int(fps * self.frame_interval)

        frame_count = 0
        extracted_count = 0
        start_time = datetime.now()

        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract frame at interval
            if frame_count % max(frame_skip, 1) == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Calculate timestamp
                timestamp = start_time + timedelta(seconds=frame_count / fps)

                yield VideoFrame(
                    frame_id=extracted_count + 1,
                    timestamp=timestamp,
                    frame_data=frame_rgb,
                    location={"name": "Unknown", "zone": "unknown"}
                )
                extracted_count += 1

            frame_count += 1

        cap.release()

    def frame_to_base64(self, frame_data) -> str:
        """Convert frame to base64 for API calls."""
        try:
            from PIL import Image as PILImage
            if isinstance(frame_data, PILImage.Image):
                import io
                buffer = io.BytesIO()
                frame_data.save(buffer, format="JPEG")
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except ImportError:
            pass

        # numpy array
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode("utf-8")


class BLIP2Captioner:
    """
    BLIP-2 Vision Language Model for image captioning.

    Why BLIP-2 over CLIP:
    - BLIP-2 generates natural language descriptions
    - CLIP only provides similarity scores between image and text
    - For security analysis, we need: "Person in dark hoodie near fence"
    - Not just: "person: 92%, fence: 87%"
    """

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        if not BLIP_AVAILABLE:
            raise ImportError(
                "BLIP-2 requires: pip install transformers torch Pillow\n"
                "Also requires GPU with sufficient VRAM"
            )

        self.model_name = model_name
        self.processor = None
        self.model = None
        self._loaded = False

    def load_model(self):
        """Load BLIP-2 model (lazy loading to save memory)."""
        if self._loaded:
            return

        print(f"Loading BLIP-2 model: {self.model_name}...")
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        self.model.to(DEVICE)
        self._loaded = True
        print(f"BLIP-2 loaded on {DEVICE}")

    def caption_frame(self, frame_data, prompt: str = None) -> str:
        """Generate caption for a single frame."""
        self.load_model()

        # Convert to PIL Image if needed
        from PIL import Image
        if not isinstance(frame_data, Image.Image):
            image = Image.fromarray(frame_data)
        else:
            image = frame_data

        # Security-focused prompt
        security_prompt = prompt or (
            "Describe this security camera image focusing on people, "
            "vehicles, and any suspicious activities:"
        )

        inputs = self.processor(images=image, text=security_prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                early_stopping=True
            )

        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption.strip()


class GPT4VisionCaptioner:
    """
    GPT-4 Vision API for image captioning.
    Most accurate but requires API costs.
    """

    def __init__(self, api_key: str = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package required: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required for GPT-4 Vision")

        self.client = openai.OpenAI(api_key=self.api_key)

    def caption_frame(self, frame_data, prompt: str = None) -> str:
        """Generate caption using GPT-4 Vision."""
        processor = VideoProcessor()
        base64_image = processor.frame_to_base64(frame_data)

        security_prompt = prompt or (
            "You are a security analyst reviewing drone surveillance footage. "
            "Describe what you see in this frame. Focus on:\n"
            "- People (count, clothing, behavior, suspicious activities)\n"
            "- Vehicles (type, color, make if visible)\n"
            "- Objects (bags, packages, tools)\n"
            "- Location details (gate, fence, building)\n"
            "Be specific and concise."
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": security_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )

        return response.choices[0].message.content


class DirectVisionAnalyzer:
    """
    Direct Vision LLM Analyzer - sends image directly to Vision LLM for complete analysis.

    This is the BETTER approach:
    - Sends actual image to GPT-4 Vision
    - Gets complete security analysis in ONE call
    - No information loss from intermediate captioning

    Flow: Image → Vision LLM → {objects, alerts, threat_level, analysis}
    """

    def __init__(self, api_key: str = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package required: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required for Vision analysis")

        self.client = openai.OpenAI(api_key=self.api_key)

    def analyze_frame(self, frame_data, location: dict = None, timestamp: str = None) -> dict:
        """
        Analyze frame directly with Vision LLM - complete security analysis in one call.

        Returns:
            dict: {
                "description": "...",
                "objects": [...],
                "alerts": [...],
                "threat_level": "...",
                "analysis": "..."
            }
        """
        processor = VideoProcessor()
        base64_image = processor.frame_to_base64(frame_data)

        location_info = location or {"name": "Unknown", "zone": "unknown"}
        time_info = timestamp or "Unknown"

        security_prompt = f"""You are a security analyst for a drone surveillance system.
Analyze this security camera image and provide a complete security assessment.

CONTEXT:
- Location: {location_info.get('name', 'Unknown')} (Zone: {location_info.get('zone', 'unknown')})
- Timestamp: {time_info}

SECURITY ALERT RULES TO CHECK:
- R001 Night Activity (HIGH): Person detected between 00:00-05:00
- R002 Loitering Detection (HIGH): Person staying in same area for extended period
- R003 Perimeter Activity (MEDIUM): Any activity in perimeter zone
- R004 Repeat Vehicle (LOW): Same vehicle seen multiple times
- R005 Unknown Vehicle (MEDIUM): Unrecognized vehicle in restricted area
- R006 Suspicious Behavior (HIGH): Face covering, hiding, suspicious actions

INSTRUCTIONS:
1. Describe what you see in the image
2. Identify ALL objects (people, vehicles, animals, items)
3. Check if ANY security alert rules should be triggered
4. Assess the overall threat level

Respond in this EXACT JSON format:
{{
    "description": "Brief description of the scene",
    "objects": [
        {{"type": "person/vehicle/animal/object", "description": "detailed description with attributes"}}
    ],
    "alerts": [
        {{"rule_id": "R00X", "name": "Rule Name", "priority": "HIGH/MEDIUM/LOW", "reason": "why triggered"}}
    ],
    "analysis": "Security analysis of the scene",
    "threat_level": "NONE/LOW/MEDIUM/HIGH/CRITICAL"
}}

If nothing suspicious is detected, return empty alerts array and threat_level "NONE"."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": security_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )

            response_text = response.choices[0].message.content

            # Parse JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                return {
                    "description": response_text,
                    "objects": [],
                    "alerts": [],
                    "analysis": response_text,
                    "threat_level": "UNKNOWN"
                }

        except Exception as e:
            return {
                "description": f"Error analyzing image: {str(e)}",
                "objects": [],
                "alerts": [],
                "analysis": f"Error: {str(e)}",
                "threat_level": "UNKNOWN"
            }

    def caption_frame(self, frame_data, prompt: str = None) -> str:
        """Compatibility method - returns just the description."""
        result = self.analyze_frame(frame_data)
        return result.get("description", "Unable to analyze image")


class SimulatedVLM:
    """
    Simulated VLM for testing without GPU/API.
    Generates realistic security-focused descriptions for demo purposes.
    """

    def __init__(self):
        self.scenarios = [
            "Blue Ford F150 pickup truck entering through main gate",
            "Person in dark hoodie walking near perimeter fence",
            "White delivery van parked at loading dock",
            "Two workers in safety vests near warehouse entrance",
            "Empty parking lot, no activity detected",
            "Person carrying large bag near back fence, face partially covered",
            "Red sedan exiting through main gate",
            "Security guard on patrol near office building",
            "Unknown person loitering near restricted area",
            "Motorcycle parked near garage entrance",
        ]
        self._index = 0

    def caption_frame(self, frame_data, prompt: str = None) -> str:
        """Generate simulated caption."""
        caption = self.scenarios[self._index % len(self.scenarios)]
        self._index += 1
        return caption


class VLMProcessor:
    """
    Main VLM Processor class that orchestrates the full pipeline.

    Pipeline:
    1. Video Input → OpenCV Frame Extraction
    2. Frames → VLM (BLIP-2/GPT-4V/Simulated) → Text Descriptions
    3. Text Descriptions → LLM Security Analysis
    4. Analysis → Alerts + Database Storage
    """

    def __init__(self, config: VLMConfig = None):
        self.config = config or VLMConfig()
        self.video_processor = VideoProcessor(self.config.frame_interval_seconds)
        self.captioner = self._init_captioner()

    def _init_captioner(self):
        """Initialize the appropriate VLM based on config."""
        if self.config.provider == "blip2":
            if not BLIP_AVAILABLE:
                print("BLIP-2 not available, falling back to simulated VLM")
                return SimulatedVLM()
            return BLIP2Captioner(self.config.model_name)

        elif self.config.provider == "gpt4v":
            if not OPENAI_AVAILABLE:
                print("OpenAI not available, falling back to simulated VLM")
                return SimulatedVLM()
            try:
                return GPT4VisionCaptioner()
            except ValueError:
                print("OpenAI API key not found, falling back to simulated VLM")
                return SimulatedVLM()

        elif self.config.provider == "direct":
            # Direct Vision Analysis - sends image directly to Vision LLM for complete analysis
            if not OPENAI_AVAILABLE:
                print("OpenAI not available for direct analysis, falling back to simulated VLM")
                return SimulatedVLM()
            try:
                return DirectVisionAnalyzer()
            except ValueError:
                print("OpenAI API key not found, falling back to simulated VLM")
                return SimulatedVLM()

        else:
            return SimulatedVLM()

    def process_video(self, video_path: str, location_zones: List[Dict] = None) -> List[VideoFrame]:
        """
        Process entire video and generate descriptions for each frame.

        Args:
            video_path: Path to video file
            location_zones: Optional list of location data to assign to frames

        Returns:
            List of VideoFrame objects with descriptions
        """
        if not location_zones:
            location_zones = [
                {"name": "Main Gate", "zone": "perimeter"},
                {"name": "Parking Lot", "zone": "parking"},
                {"name": "Warehouse", "zone": "storage"},
                {"name": "Loading Dock", "zone": "operations"},
                {"name": "Back Fence", "zone": "perimeter"},
            ]

        processed_frames = []

        print(f"Processing video: {video_path}")
        video_info = self.video_processor.get_video_info(video_path)
        print(f"Video info: {video_info['duration_seconds']}s, {video_info['fps']:.1f} fps")

        for frame in self.video_processor.extract_frames(video_path, self.config.max_frames):
            # Assign location (cycle through zones)
            frame.location = location_zones[frame.frame_id % len(location_zones)]

            # Generate description using VLM
            frame.description = self.captioner.caption_frame(frame.frame_data)

            # Generate telemetry data
            frame.telemetry = {
                "drone_id": "DRONE-001",
                "timestamp": frame.timestamp.isoformat(),
                "latitude": 37.7749 + (frame.frame_id * 0.0001),
                "longitude": -122.4194 + (frame.frame_id * 0.0001),
                "altitude": 50,
                "battery": max(100 - frame.frame_id, 20),
            }

            processed_frames.append(frame)
            print(f"Frame {frame.frame_id}: {frame.description[:60]}...")

        return processed_frames

    def process_single_frame(self, frame_data, location: Dict = None) -> VideoFrame:
        """Process a single frame/image."""
        description = self.captioner.caption_frame(frame_data)

        return VideoFrame(
            frame_id=1,
            timestamp=datetime.now(),
            frame_data=frame_data,
            description=description,
            location=location or {"name": "Unknown", "zone": "unknown"}
        )

    def analyze_frame_directly(self, frame_data, location: Dict = None, timestamp: str = None) -> dict:
        """
        Analyze frame directly with Vision LLM - complete security analysis in one API call.

        This is the BETTER approach when using GPT-4 Vision:
        - Sends actual image to Vision LLM
        - Gets objects, alerts, and threat level in ONE call
        - No information loss from intermediate captioning

        Args:
            frame_data: Image data (PIL Image or numpy array)
            location: Location info dict
            timestamp: Timestamp string

        Returns:
            dict: Complete analysis with objects, alerts, threat_level, etc.
        """
        if isinstance(self.captioner, DirectVisionAnalyzer):
            return self.captioner.analyze_frame(frame_data, location, timestamp)
        elif isinstance(self.captioner, GPT4VisionCaptioner):
            # Create DirectVisionAnalyzer on the fly
            analyzer = DirectVisionAnalyzer()
            return analyzer.analyze_frame(frame_data, location, timestamp)
        else:
            # Fallback: use captioner for description, then return basic structure
            description = self.captioner.caption_frame(frame_data)
            return {
                "description": description,
                "objects": [],
                "alerts": [],
                "analysis": "Simulated analysis - use 'direct' provider for full Vision LLM analysis",
                "threat_level": "UNKNOWN"
            }


def get_vlm_status() -> Dict:
    """Get status of available VLM options."""
    has_openai_key = OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY"))
    return {
        "blip2_available": BLIP_AVAILABLE,
        "gpt4v_available": has_openai_key,
        "direct_available": has_openai_key,  # Direct Vision Analysis (recommended)
        "torch_available": TORCH_AVAILABLE,
        "device": DEVICE,
        "recommended": "direct" if has_openai_key else ("blip2" if BLIP_AVAILABLE else "simulated")
    }


def process_uploaded_video(video_file, provider: str = "simulated") -> List[Dict]:
    """
    Process an uploaded video file from Streamlit.

    Args:
        video_file: Streamlit UploadedFile object
        provider: VLM provider to use

    Returns:
        List of processed frames with descriptions
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name

    try:
        config = VLMConfig(provider=provider)
        processor = VLMProcessor(config)
        frames = processor.process_video(tmp_path)

        return [
            {
                "frame_id": f.frame_id,
                "timestamp": f.timestamp.isoformat(),
                "description": f.description,
                "location": f.location,
                "telemetry": f.telemetry
            }
            for f in frames
        ]
    finally:
        os.unlink(tmp_path)


def process_uploaded_image(image_file, provider: str = "simulated") -> Dict:
    """
    Process an uploaded image file from Streamlit.

    Args:
        image_file: Streamlit UploadedFile object or PIL Image
        provider: VLM provider to use

    Returns:
        Dict with frame description and metadata
    """
    try:
        from PIL import Image

        if hasattr(image_file, 'read'):
            image = Image.open(image_file)
        else:
            image = image_file

        config = VLMConfig(provider=provider)
        processor = VLMProcessor(config)
        frame = processor.process_single_frame(image)

        return {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp.isoformat(),
            "description": frame.description,
            "location": frame.location,
        }
    except Exception as e:
        return {
            "frame_id": 1,
            "timestamp": datetime.now().isoformat(),
            "description": f"Error processing image: {str(e)}",
            "location": {"name": "Unknown", "zone": "unknown"},
        }


if __name__ == "__main__":
    print("VLM Status:", json.dumps(get_vlm_status(), indent=2))

    print("\nSimulated VLM test:")
    sim_vlm = SimulatedVLM()
    for i in range(5):
        caption = sim_vlm.caption_frame(None)
        print(f"  Frame {i+1}: {caption}")
