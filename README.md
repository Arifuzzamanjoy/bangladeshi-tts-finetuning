# 🇧🇩 Bangladeshi TTS Fine-tuning Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

> **Fine-tune XTTS v2 for authentic Bangladeshi Bangla speech synthesis with constraint-first design for free-tier GPU environments.**

## 🎯 Project Overview

This repository provides a **complete MLOps pipeline** for fine-tuning neural text-to-speech models specifically for **Bangladeshi Bangla accent**. The project is designed for **resource-constrained environments** (Google Colab T4 GPU) while maintaining production-ready code quality and reproducible results.

### Key Features
- 🎤 **XTTS v2 Fine-tuning** optimized for Bangladeshi accent
- 🔧 **T4 GPU Optimization** with mixed precision and gradient accumulation
- 📊 **Comprehensive Evaluation** using MCD and speaker similarity metrics
- 🎯 **Accent-based Curation** with unsupervised clustering
- 🚀 **Production Deployment** with Docker and ONNX optimization
- 📚 **Complete Documentation** with reproducible workflows

## 🏗️ Project Structure

```
bangladeshi-tts-finetuning/
├── 📚 notebooks/                    # Core implementation notebooks
│   ├── 01_Setup_and_Baseline.ipynb         # Environment & baseline
│   ├── 02_Data_Curation.ipynb              # Data processing pipeline
│   ├── 03_Accent_Evaluation_Framework.ipynb # Evaluation metrics
│   ├── 04_Finetuning_Experiments.ipynb     # Training experiments
│   ├── 05_Final_Evaluation.ipynb           # Model comparison
│   └── 06_Deployment_Demo.ipynb            # Interactive demo
├── 🔧 src/                         # Reusable Python modules
│   ├── data_processing/             # Data pipeline components
│   ├── evaluation/                  # Accent evaluation framework
│   ├── training/                    # Training utilities
│   └── deployment/                  # Production deployment
├── 📁 data/                         # Dataset storage
│   ├── raw/                         # Original datasets
│   ├── processed/                   # Preprocessed audio
│   ├── manifests/                   # Training metadata
│   └── validation/                  # Hold-out validation sets
├── 🤖 models/                       # Model storage
│   ├── baseline/                    # Original checkpoints
│   ├── experiments/                 # Fine-tuned models
│   └── production/                  # Final deployment models
├── 📊 experiments/                  # Experiment tracking
├── 🐳 deployment/                   # Production deployment
├── requirements.txt                 # Python dependencies
├── DEPLOYMENT.md                   # Deployment strategy
├── PROJECT_SUMMARY.md              # Technical documentation
└── README.md                       # This file
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
```python
# 1. Open Google Colab and create new notebook
# 2. Clone the repository
!git clone https://github.com/your-username/bangladeshi-tts-finetuning.git
%cd bangladeshi-tts-finetuning

# 3. Mount Google Drive for persistence
from google.colab import drive
drive.mount('/content/drive')

# 4. Install dependencies
!pip install -r requirements.txt

# 5. Initialize experiment tracking
import wandb
wandb.login()  # Enter your W&B API key

# 6. Start with notebook sequence
# Run notebooks/01_Setup_and_Baseline.ipynb first
```

### Option 2: Local Development
```bash
# Prerequisites: Python 3.8+, CUDA-capable GPU (16GB+ VRAM recommended)
git clone https://github.com/your-username/bangladeshi-tts-finetuning.git
cd bangladeshi-tts-finetuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export TTS_PROJECT_ROOT="$(pwd)"
export TTS_DATA_DIR="$TTS_PROJECT_ROOT/data"
export TTS_MODEL_DIR="$TTS_PROJECT_ROOT/models"

# Start Jupyter Lab
jupyter lab
```

## 📋 Suggested Workflows

### 🎯 **Workflow 1: Complete Pipeline Execution**
*For users wanting to run the full project from scratch*

```bash
# Phase 1: Environment & Baseline (30 minutes)
notebooks/01_Setup_and_Baseline.ipynb
├── Environment setup with W&B integration
├── XTTS v2 model loading and inspection
├── Baseline synthesis with diagnostic tests
└── Structured qualitative analysis framework

# Phase 2: Data Processing (1-2 hours)
notebooks/02_Data_Curation.ipynb
├── Automated dataset download (OpenSLR 53/54, Common Voice)
├── Audio preprocessing (resample, trim, filter)
├── Bengali text normalization
└── Training manifest generation with statistics

# Phase 3: Evaluation Framework (1 hour)
notebooks/03_Accent_Evaluation_Framework.ipynb
├── AccentEvaluator class implementation
├── MCD and speaker similarity metrics
├── Accent-based data curation with clustering
└── Validation set creation

# Phase 4: Model Training (2-4 hours)
notebooks/04_Finetuning_Experiments.ipynb
├── T4 GPU optimized training configurations
├── Experiment A: Conservative layer-specific fine-tuning
├── Experiment B: Full model fine-tuning with low LR
└── Checkpoint management and W&B logging

# Phase 5: Evaluation & Selection (1 hour)
notebooks/05_Final_Evaluation.ipynb
├── Comparative model evaluation
├── Objective metrics computation
├── Data-driven model recommendation
└── Technical report generation

# Phase 6: Deployment Demo (30 minutes)
notebooks/06_Deployment_Demo.ipynb
├── Interactive Gradio web interface
├── Real-time synthesis demonstration
└── Production deployment preparation
```

### 🔬 **Workflow 2: Research & Development**
*For researchers focusing on specific components*

```bash
# Option 2A: Accent Evaluation Focus
notebooks/03_Accent_Evaluation_Framework.ipynb
├── Develop new evaluation metrics
├── Experiment with clustering algorithms
├── Create custom accent datasets
└── Validate evaluation approaches

# Option 2B: Training Strategy Focus
notebooks/04_Finetuning_Experiments.ipynb
├── Design new fine-tuning strategies
├── Experiment with layer freezing patterns
├── Optimize for different GPU configurations
└── Implement curriculum learning approaches

# Option 2C: Data Processing Focus
notebooks/02_Data_Curation.ipynb
├── Add new dataset sources
├── Implement advanced text normalization
├── Create speaker diarization pipeline
└── Develop quality filtering heuristics
```

### 🏭 **Workflow 3: Production Deployment**
*For teams deploying models to production*

```bash
# Phase 1: Model Selection & Optimization
notebooks/05_Final_Evaluation.ipynb
├── Select best performing model
├── Export to ONNX format
├── Apply quantization for efficiency
└── Benchmark inference performance

# Phase 2: Containerization
deployment/docker/
├── Build optimized Docker images
├── Configure multi-stage builds
├── Set up health checks
└── Implement security best practices

# Phase 3: Service Deployment
deployment/kubernetes/
├── Deploy with Kubernetes manifests
├── Configure autoscaling policies
├── Set up monitoring and logging
└── Implement A/B testing framework

# Phase 4: Quality Monitoring
src/deployment/monitoring.py
├── Real-time quality assessment
├── Performance metrics tracking
├── Automated alerting system
└── Model drift detection
```

### 📊 **Workflow 4: Experiment Tracking & Analysis**
*For systematic experimentation and hyperparameter optimization*

```bash
# Experiment Management Workflow
├── Design experiment matrix
│   ├── Learning rates: [1e-5, 5e-5, 1e-4]
│   ├── Layer freezing: [encoder, decoder, full]
│   ├── Data sources: [OpenSLR, Common Voice, combined]
│   └── Batch configurations: [1+16, 2+8, 1+32]
├── Automated experiment execution
│   ├── W&B sweep configuration
│   ├── Distributed training setup
│   └── Resource allocation management
├── Results analysis & visualization
│   ├── Performance comparison dashboards
│   ├── Statistical significance testing
│   └── Ablation study reports
└── Model selection & deployment
    ├── Multi-criteria decision analysis
    ├── Production readiness assessment
    └── Rollout strategy planning
```

## ⚙️ Configuration Management

### Environment Configuration
```python
# config/environment.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProjectConfig:
    # Paths
    project_root: str = os.getenv("TTS_PROJECT_ROOT", "./")
    data_dir: str = os.getenv("TTS_DATA_DIR", "./data")
    model_dir: str = os.getenv("TTS_MODEL_DIR", "./models")
    
    # Training
    batch_size: int = int(os.getenv("TTS_BATCH_SIZE", "1"))
    grad_accum_steps: int = int(os.getenv("TTS_GRAD_ACCUM", "16"))
    mixed_precision: bool = os.getenv("TTS_MIXED_PRECISION", "true").lower() == "true"
    
    # Evaluation
    validation_size: int = int(os.getenv("TTS_VAL_SIZE", "15"))
    evaluation_frequency: int = int(os.getenv("TTS_EVAL_FREQ", "200"))
    
    # Deployment
    inference_device: str = os.getenv("TTS_DEVICE", "auto")
    max_concurrent_requests: int = int(os.getenv("TTS_MAX_REQUESTS", "4"))
```

### Flexible Training Configuration
```python
# config/training.py
def create_training_config(
    experiment_name: str,
    strategy: str = "conservative",  # conservative, full, custom
    gpu_memory_gb: int = 16,
    session_length_hours: float = 2.0
) -> dict:
    """
    Create adaptive training configuration based on resources and strategy.
    
    Strategies:
    - conservative: Freeze encoder, train decoder + attention
    - full: Full model fine-tuning with ultra-low LR
    - custom: User-defined layer selection
    """
    
    base_config = {
        "experiment_name": experiment_name,
        "mixed_precision": True,
        "gradient_checkpointing": gpu_memory_gb < 24,
        "max_steps": int(session_length_hours * 1800),  # ~1800 steps/hour
    }
    
    if strategy == "conservative":
        base_config.update({
            "learning_rate": 1e-4,
            "batch_size": 2 if gpu_memory_gb >= 16 else 1,
            "grad_accum_steps": 8 if gpu_memory_gb >= 16 else 16,
            "freeze_encoder": True,
            "freeze_embeddings": True,
        })
    elif strategy == "full":
        base_config.update({
            "learning_rate": 5e-5,
            "batch_size": 1,
            "grad_accum_steps": 32,
            "freeze_encoder": False,
            "weight_decay": 0.01,
        })
    
    return base_config
```

## 🔄 Scalability Features

### 1. **Modular Architecture**
```python
# Extensible component system
from src.data_processing import AudioProcessor, TextNormalizer
from src.evaluation import AccentEvaluator, QualityMetrics
from src.training import ExperimentManager, CheckpointHandler

# Easy component swapping
audio_processor = AudioProcessor(strategy="aggressive_trimming")
text_normalizer = TextNormalizer(language="bengali", dialect="bangladeshi")
evaluator = AccentEvaluator(metrics=["mcd", "speaker_sim", "prosody"])
```

### 2. **Multi-GPU Training Support**
```python
# config/distributed.py
def setup_distributed_training(world_size: int, rank: int):
    """Configure distributed training for multi-GPU setups."""
    import torch.distributed as dist
    
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank
    )
    
    return {
        "batch_size_per_gpu": 1,
        "effective_batch_size": world_size * 16,  # with grad accumulation
        "learning_rate": 1e-4 * world_size,     # linear scaling
        "sync_batch_norm": True,
        "find_unused_parameters": True,
    }
```

### 3. **Dataset Expansion Framework**
```python
# src/data_processing/dataset_registry.py
class DatasetRegistry:
    """Extensible dataset management system."""
    
    def __init__(self):
        self.datasets = {
            "openslr_53": OpenSLRDataset,
            "openslr_54": OpenSLRDataset, 
            "common_voice": CommonVoiceDataset,
            "bengali_ai": BengaliAIDataset,
        }
    
    def register_dataset(self, name: str, dataset_class):
        """Add new dataset type."""
        self.datasets[name] = dataset_class
    
    def create_combined_manifest(self, dataset_configs: list):
        """Merge multiple datasets with provenance tracking."""
        # Implementation for flexible dataset combination
        pass
```

### 4. **Evaluation Metrics Extension**
```python
# src/evaluation/metrics_registry.py
class MetricsRegistry:
    """Pluggable evaluation metrics system."""
    
    def __init__(self):
        self.metrics = {
            "mcd": MelCepstralDistortion,
            "speaker_similarity": SpeakerSimilarity,
            "prosody_correlation": ProsodyCorrelation,
            "phoneme_accuracy": PhonemeAccuracy,
        }
    
    def register_metric(self, name: str, metric_class):
        """Add custom evaluation metric."""
        self.metrics[name] = metric_class
    
    def compute_all_metrics(self, synth_audio, ref_audio):
        """Compute all registered metrics."""
        results = {}
        for name, metric_class in self.metrics.items():
            metric = metric_class()
            results[name] = metric.compute(synth_audio, ref_audio)
        return results
```

## 🎛️ Flexibility Features

### 1. **Multi-Language Support**
```python
# config/language_configs.py
LANGUAGE_CONFIGS = {
    "bengali": {
        "phonemizer": "espeak",
        "language_code": "bn",
        "text_normalizer": "BengaliNormalizer",
        "digit_mapping": "bengali_digits",
        "punctuation_rules": "bengali_punct",
    },
    "hindi": {
        "phonemizer": "espeak", 
        "language_code": "hi",
        "text_normalizer": "HindiNormalizer",
        "digit_mapping": "devanagari_digits",
        "punctuation_rules": "hindi_punct",
    }
}

def get_language_config(language: str):
    """Get language-specific configuration."""
    return LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["bengali"])
```

### 2. **Model Architecture Abstraction**
```python
# src/training/model_registry.py
class ModelRegistry:
    """Support multiple TTS architectures."""
    
    def __init__(self):
        self.models = {
            "xtts_v2": XTTSv2Model,
            "xtts_v1": XTTSv1Model, 
            "vits": VITSModel,
            "tacotron2": Tacotron2Model,
        }
    
    def create_model(self, model_type: str, config: dict):
        """Factory method for model creation."""
        model_class = self.models[model_type]
        return model_class(config)
    
    def get_training_strategy(self, model_type: str):
        """Get model-specific training recommendations."""
        strategies = {
            "xtts_v2": {
                "recommended_batch_size": 1,
                "gradient_accumulation": 16,
                "mixed_precision": True,
                "layer_freezing": ["encoder"],
            },
            "vits": {
                "recommended_batch_size": 4,
                "gradient_accumulation": 4,
                "mixed_precision": True,
                "layer_freezing": ["text_encoder"],
            }
        }
        return strategies.get(model_type, strategies["xtts_v2"])
```

### 3. **Deployment Target Abstraction**
```python
# deployment/target_configs.py
DEPLOYMENT_TARGETS = {
    "colab": {
        "max_memory_gb": 16,
        "optimization": "memory_efficient",
        "checkpoint_frequency": 100,
        "storage_backend": "google_drive",
    },
    "kaggle": {
        "max_memory_gb": 16,
        "optimization": "compute_efficient", 
        "checkpoint_frequency": 200,
        "storage_backend": "kaggle_datasets",
    },
    "local_gpu": {
        "max_memory_gb": 24,
        "optimization": "speed_optimized",
        "checkpoint_frequency": 500,
        "storage_backend": "local_disk",
    },
    "cloud_gpu": {
        "max_memory_gb": 80,
        "optimization": "distributed",
        "checkpoint_frequency": 1000,
        "storage_backend": "cloud_storage",
    }
}
```

## 📚 Advanced Usage Examples

### Custom Dataset Integration
```python
# Add your own dataset
from src.data_processing.dataset_registry import DatasetRegistry

class MyCustomDataset:
    def __init__(self, config):
        self.config = config
    
    def download_and_process(self):
        # Custom dataset processing logic
        pass
    
    def create_manifest(self):
        # Generate training manifest
        pass

# Register and use
registry = DatasetRegistry()
registry.register_dataset("my_dataset", MyCustomDataset)

# Use in pipeline
config = {"path": "/path/to/data", "language": "bengali"}
dataset = registry.create_dataset("my_dataset", config)
```

### Custom Evaluation Metric
```python
# Add custom evaluation metric
from src.evaluation.metrics_registry import MetricsRegistry

class MyCustomMetric:
    def compute(self, synth_audio, ref_audio):
        # Custom metric computation
        return score

registry = MetricsRegistry()
registry.register_metric("my_metric", MyCustomMetric)
```

### Production Monitoring
```python
# Real-time quality monitoring
from src.deployment.monitoring import QualityMonitor

monitor = QualityMonitor(
    metrics=["mcd", "speaker_similarity", "latency"],
    alert_thresholds={"mcd": 2.5, "latency": 5.0},
    notification_webhook="https://your-webhook-url"
)

# Integrate with inference API
@app.route("/synthesize")
def synthesize_text():
    audio = tts_model.synthesize(text, speaker_audio)
    quality_scores = monitor.evaluate(audio, reference_audio)
    monitor.log_metrics(quality_scores)
    return audio
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow
```bash
# 1. Fork the repository
# 2. Create feature branch
git checkout -b feature/amazing-feature

# 3. Set up development environment
pip install -r requirements-dev.txt
pre-commit install

# 4. Make changes and test
pytest tests/
black src/ notebooks/
flake8 src/

# 5. Submit pull request
```

### Code Quality Standards
- **Type Hints**: All functions must include proper type annotations
- **Docstrings**: Google-style docstrings for all public methods
- **Testing**: Unit tests for critical functions (target: >90% coverage)
- **Formatting**: Black code formatting, line length 100
- **Documentation**: Keep README and docstrings current

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) - Excellent open-source TTS framework
- [OpenSLR](http://www.openslr.org/) - High-quality Bengali speech datasets
- [SpeechBrain](https://speechbrain.github.io/) - Speaker recognition models
- [Google Colab](https://colab.research.google.com/) - Free GPU access
- [Weights & Biases](https://wandb.ai/) - Experiment tracking platform

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/your-username/bangladeshi-tts-finetuning/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/bangladeshi-tts-finetuning/discussions)
- **Documentation**: [Project Wiki](https://github.com/your-username/bangladeshi-tts-finetuning/wiki)
- **Email**: your.email@domain.com

---

<p align="center">
  <b>Made with ❤️ for the Bengali-speaking community worldwide</b><br>
  <i>Empowering authentic Bangladeshi voice in AI technology</i>
</p>

---

## 📈 Roadmap

### Short-term (1-3 months)
- [ ] Multi-speaker support with speaker embedding clustering
- [ ] Real-time streaming synthesis for long texts
- [ ] Mobile app with on-device quantized model
- [ ] Advanced text preprocessing with NER integration

### Long-term (6-12 months)
- [ ] Emotion-controlled speech synthesis
- [ ] Regional dialect support (Chittagong, Sylhet, etc.)
- [ ] Code-switching for Bengali-English mixed text
- [ ] Few-shot speaker adaptation framework
- [ ] Integration with popular Bengali NLP libraries

### Community Goals
- [ ] Open dataset contribution platform
- [ ] Multilingual model supporting South Asian languages
- [ ] Educational resources and tutorials
- [ ] Research collaboration opportunities