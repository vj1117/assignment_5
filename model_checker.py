import torch
import torch.nn as nn
import sys
import json
from pathlib import Path

def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameter count details:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,}")
    return total

def check_model_requirements(model):
    """Check if model meets requirements"""
    total_params = count_parameters(model)

    # Check for Batch Normalization
    has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())

    # Check for Dropout
    has_dropout = any(isinstance(m, (nn.Dropout, nn.Dropout2d)) for m in model.modules())

    # Check for Global Average Pooling
    has_gap = any(isinstance(m, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)) for m in model.modules())

    # Check for Fully Connected layers
    has_fc = any(isinstance(m, nn.Linear) for m in model.modules())

    return {
        'total_params': total_params,
        'under_20k': total_params < 20000,
        'has_bn': has_bn,
        'has_dropout': has_dropout,
        'has_gap': has_gap,
        'has_fc': has_fc
    }

def print_requirements(requirements):
    """Print model requirements in a formatted way"""
    print(f"Model Requirements Check:")
    print(f"  Total Parameters: {requirements['total_params']:,}")
    print(f"  Under 20k params: {'✓' if requirements['under_20k'] else '✗'}")
    print(f"  Has Batch Normalization: {'✓' if requirements['has_bn'] else '✗'}")
    print(f"  Has Dropout: {'✓' if requirements['has_dropout'] else '✗'}")
    print(f"  Has Global Average Pooling: {'✓' if requirements['has_gap'] else '✗'}")
    print(f"  Has Fully Connected Layer: {'✓' if requirements['has_fc'] else '✗'}")
    
    return requirements

def check_model(model_path):
    """Load model and check requirements"""
    try:
        # Dynamically import the model
        sys.path.append('.')
        model_file = Path(model_path).stem
        module_name = model_file.replace('/', '.')
        
        # Import the module
        module = __import__(module_name, fromlist=['*'])
        
        # Find model classes in the module
        model_classes = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
                model_classes.append(attr)
        
        if not model_classes:
            print(f"No model classes found in {model_path}")
            return False
        
        # Check each model class
        results = {}
        for model_class in model_classes:
            print(f"\nChecking model: {model_class.__name__}")
            try:
                # Initialize without parameters
                model = model_class()
                
                # Create a dummy input to ensure the model is properly initialized
                dummy_input = torch.randn(1, 1, 28, 28)  # MNIST input size
                try:
                    # Try a forward pass to initialize lazy modules if any
                    with torch.no_grad():
                        _ = model(dummy_input)
                    print(f"Successfully ran forward pass with dummy input")
                except Exception as e:
                    print(f"Warning: Forward pass failed: {e}")
            except Exception as e:
                print(f"Error initializing {model_class.__name__}: {e}")
                continue
            
            # Check requirements
            print(f"Counting parameters for {model_class.__name__}...")
            requirements = check_model_requirements(model)
            print_requirements(requirements)
            results[model_class.__name__] = requirements
        
        # Save results to JSON
        with open('model_requirements.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return True
    
    except Exception as e:
        print(f"Error checking model: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model_checker.py <model_file_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    success = check_model(model_path)
    sys.exit(0 if success else 1)