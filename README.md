# MNIST CNN Optimization Project

## Project Overview and Objective

This project focuses on developing an efficient Convolutional Neural Network (CNN) for the MNIST handwritten digit classification task. The primary objective is to create a model that achieves high accuracy while maintaining a small parameter footprint, specifically:

- **Target Accuracy**: > 99.4% on the test set
- **Parameter Constraint**: < 20,000 parameters
- **Epoch Constraint**: < 20 epochs

The approach involves iterative experimentation with various CNN architectures, systematically applying techniques like batch normalization, dropout, global average pooling, and depthwise separable convolutions to optimize the model's performance while keeping the parameter count low.

## Experiments Summary

The project consists of 13 experiments, each building upon the previous one with architectural modifications and hyperparameter tuning to improve performance while maintaining parameter efficiency.

### Detailed Experiments Table

| Exp# | Architecture Description | Parameter Count | Batch Size | # Epochs | Training Accuracy | Test Accuracy | Criteria Satisfied |
|------|--------------------------|----------------|------------|----------|------------------|---------------|-------------------|
| Base | Initial large model with multiple conv layers. Base architecture to check parameter count | 6,379,786 | 64 | - | - | - | Model Requirements Check:<br>  Total Parameters: 6,379,786<br>  Under 20k params: ✗ |
| 1 | Basic CNN with 8 initial filters. Reduced filter counts (32→8) to decrease parameters | 20,346 | 64 | 15 | 99.95% | 98.98% | Model Requirements Check:<br>  Total Parameters: 20,346<br>  Under 20k params: ✗<br>  Has Batch Normalization: ✗<br>  Has Dropout: ✗<br>  Has Global Average Pooling: ✗<br>  Has Fully Connected Layer: ✗ |
| 2 | Use 1x1 convolutions to reduce parameters | 15,226 | 64 | 15 | 98.47% | 97.27% | Total Parameters: 15,226<br>  Under 20k params: ✓<br>  Has Batch Normalization: ✗<br>  Has Dropout: ✗<br>  Has Global Average Pooling: ✗<br>  Has Fully Connected Layer: ✗ |
| 3 | Add batch normalization for better training and check the impact on validation accuracy | 15,370 | 64 | 18 | 98.94% | 97.05% | Total Parameters: 15,370<br>  Under 20k params: ✓<br>  Has Batch Normalization: ✓<br>  Has Dropout: ✗<br>  Has Global Average Pooling: ✗<br>  Has Fully Connected Layer: ✗ |
| 4 | Applied Global Average Pooling (GAP) to reduce parameters | 15,370 | 64 | 18 | 99.88% | 99.28% | Total Parameters: 15,370<br>  Under 20k params: ✓<br>  Has Batch Normalization: ✓<br>  Has Dropout: ✗<br>  Has Global Average Pooling: ✓<br>  Has Fully Connected Layer: ✗ |
| 5 | Added Dropout for regularization | 15,370 | 64 | 18 | 99.45% | 99.36% | Total Parameters: 15,370<br>  Under 20k params: ✓<br>  Has Batch Normalization: ✓<br>  Has Dropout: ✓<br>  Has Global Average Pooling: ✓<br>  Has Fully Connected Layer: ✗ |
| 6 | Changed schedule to 'onecycle', used augmentation, changed dropout rate to 0.15 from 0.10 | 15,370 | 64 | 18 | 99.35% | 99.24% | Total Parameters: 15,370<br>  Under 20k params: ✓<br>  Has Batch Normalization: ✓<br>  Has Dropout: ✓<br>  Has Global Average Pooling: ✓<br>  Has Fully Connected Layer: ✗ |
| 7 | Added one more epoch to improve convergence | 15,370 | 64 | 19 | 99.33% | 99.37% | Total Parameters: 15,370<br>  Under 20k params: ✓<br>  Has Batch Normalization: ✓<br>  Has Dropout: ✓<br>  Has Global Average Pooling: ✓<br>  Has Fully Connected Layer: ✗ |
| 8 | Changed shape of point-wise convolution from 16, 32 to 16, 16. Added conv of 64 out_channels | 15,642 | 64 | 19 | 98.78% | 99.28% | Total Parameters: 15,642<br>  Under 20k params: ✓<br>  Has Batch Normalization: ✓<br>  Has Dropout: ✓<br>  Has Global Average Pooling: ✓<br>  Has Fully Connected Layer: ✗ |
| 9 | Implemented groups (depth-wise and point-wise) architecture | 17,786 | 64 | 19 | 99.2% | 99.36% | Total Parameters: 17,786<br>  Under 20k params: ✓<br>  Has Batch Normalization: ✓<br>  Has Dropout: ✓<br>  Has Global Average Pooling: ✓<br>  Has Fully Connected Layer: ✗ |
| 10 | Changed batch size to 128 for better generalization | 17,786 | 128 | 19 | 99.25% | 99.51% | Total Parameters: 17,786<br>  Under 20k params: ✓<br>  Has Batch Normalization: ✓<br>  Has Dropout: ✓<br>  Has Global Average Pooling: ✓<br>  Has Fully Connected Layer: ✗ |
| 11 | Further optimized groups architecture, keeping batch size 128 | 14,986 | 128 | 19 | 99.10% | 99.46% | Total Parameters: 14,986<br>  Under 20k params: ✓<br>  Has Batch Normalization: ✓<br>  Has Dropout: ✓<br>  Has Global Average Pooling: ✓<br>  Has Fully Connected Layer: ✗ |
| 12 | Optimized padding and bias settings for better parameter efficiency | 14,874 | 64 | 19 | 99.03% | 99.49% | Total Parameters: 14,874<br>  Under 20k params: ✓<br>  Has Batch Normalization: ✓<br>  Has Dropout: ✓<br>  Has Global Average Pooling: ✓<br>  Has Fully Connected Layer: ✗ |


## Experiment Progression Analysis

The experiments show a clear progression in model development:

1. **Parameter Reduction**: Starting from a large model with over 6M parameters, we quickly reduced to under 20K parameters by using smaller filter counts and 1x1 convolutions.

2. **Architecture Optimization**: 
   - Introduced batch normalization to improve training stability (Exp 3)
   - Added Global Average Pooling to eliminate fully connected layers (Exp 4)
   - Implemented dropout for regularization (Exp 5)
   - Utilized depthwise separable convolutions for parameter efficiency (Exp 9-12)

3. **Training Optimization**:
   - Implemented OneCycle learning rate schedule (Exp 6)
   - Added data augmentation to improve generalization (Exp 6)
   - Experimented with different batch sizes (Exp 10-11)
   - Fine-tuned dropout rates for optimal regularization

4. **Performance Improvement**:
   - Test accuracy improved from 98.98% to 99.49%, exceeding the target of 99.4%
   - Parameter count reduced from over 6M to under 15K, well below the 20K constraint
   - All training completed within 19 epochs, meeting the <20 epochs constraint

## Final Model Architecture

The final model (Small_MNIST13) successfully meets all the project requirements with the following characteristics:

### Architecture Overview

```python
class Small_MNIST13(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(Small_MNIST13, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Pooling and regularization
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Depthwise separable convolution
        self.conv3_dw = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
        self.conv3_pw = nn.Conv2d(32, 32, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 64, 1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout_rate * 1.5)
        
        # Another depthwise separable convolution
        self.conv5_dw = nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False)
        self.conv5_pw = nn.Conv2d(64, 64, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        
        # Final classification layer
        self.conv6 = nn.Conv2d(64, 10, 3, padding=1, bias=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
```

### Key Architecture Features

- **Total Parameters**: 14,874 (well under the 20k limit)
- **Batch Normalization**: Used after each convolutional layer (except the final one)
- **Dropout**: Two dropout layers with rates of 0.1 and 0.15*1.5
- **Global Average Pooling**: Used instead of fully connected layers to reduce parameters
- **Depthwise Separable Convolutions**: Used to reduce parameters while maintaining expressiveness
- **1x1 Convolutions**: Used for channel-wise dimensionality reduction
- **Bias Optimization**: Removed bias from most convolutional layers to reduce parameters

### Receptive Field Analysis

The model was designed with careful consideration of the receptive field, ensuring it's large enough to capture the essential features of MNIST digits. The final receptive field is sufficient to cover the critical parts of the 28x28 input images.

## Training Details

### Training Configuration

- **Number of Epochs**: 19 (under the constraint of less than 20)
- **Optimizer**: Adam with learning rate of 0.001
- **Learning Rate Scheduler**: OneCycleLR (implemented in Exp 6)
- **Loss Function**: Negative Log Likelihood (NLL) Loss
- **Batch Size**: 64 (increased to 128 in Exp 10-11)
- **Data Augmentation**: Applied from Exp 6 onwards
- **Regularization**: Dropout (rates 0.10-0.15) and Batch Normalization

### Training Approach

The training approach involved:

1. **Iterative Experimentation**: Starting with a simple model and gradually refining it
2. **Parameter Efficiency**: Focusing on techniques to reduce parameters while maintaining performance
3. **Regularization**: Adding dropout and batch normalization to prevent overfitting
4. **Architecture Optimization**: Using depthwise separable convolutions and 1x1 convolutions
5. **Hyperparameter Tuning**: Adjusting learning rates, dropout rates, and batch sizes

## Results

### Final Performance

- **Test Accuracy**: 99.49% (exceeding the target of 99.4%)
- **Parameter Count**: 14,874 (well below the 20k limit)
- **Training Epochs**: 19 (within the constraint of less than 20)

### Key Insights

1. **Batch Normalization Impact**: Adding batch normalization improved training stability and convergence
2. **Dropout Effectiveness**: Strategic placement of dropout layers helped prevent overfitting
3. **Global Average Pooling**: Replacing fully connected layers with GAP significantly reduced parameters
4. **Depthwise Separable Convolutions**: These proved highly effective for parameter efficiency
5. **Parameter Efficiency**: The final model achieved high accuracy with only ~14.9k parameters, demonstrating that well-designed smaller models can perform excellently on the MNIST task
6. **Batch Size Effect**: Increasing batch size from 64 to 128 in later experiments improved generalization
7. **Bias Removal**: Removing bias from most convolutional layers helped reduce parameters without sacrificing performance

### Training Progression

The training process showed consistent improvement across experiments, with the final model achieving the target accuracy while maintaining parameter efficiency. The use of OneCycleLR scheduler helped in faster convergence within the epoch constraint.

## Conclusion

This project successfully demonstrates that a carefully designed CNN with modern architectural techniques can achieve high accuracy on the MNIST dataset while maintaining a small parameter footprint. The final model meets all the specified constraints and provides a good example of efficient deep learning model design.

The key takeaway is that through systematic experimentation and application of techniques like batch normalization, dropout, global average pooling, and depthwise separable convolutions, it's possible to create highly efficient models without sacrificing performance.

The experiment progression clearly shows how each architectural decision contributed to the final model's success:
1. Starting with basic parameter reduction techniques
2. Adding regularization methods to prevent overfitting
3. Implementing advanced convolution techniques for parameter efficiency
4. Fine-tuning hyperparameters for optimal performance

This methodical approach resulted in a model that exceeds the target accuracy of 99.4% while using only 14,874 parameters, well below the 20,000 parameter constraint.

## Continuous Integration with GitHub Actions

This project includes a GitHub Actions workflow that automatically verifies model requirements whenever code is pushed or a pull request is created. The workflow checks:

1. **Total Parameter Count**: Ensures the model has fewer than 20,000 parameters
2. **Architectural Requirements**: Verifies the model uses:
   - Batch Normalization
   - Dropout
   - Global Average Pooling or Fully Connected Layer

### How It Works

The GitHub Actions workflow:

1. Extracts model classes from the notebook
2. Runs automated checks on each model
3. Reports which models meet all requirements
4. Fails the build if no model meets all requirements

This ensures that any changes to the model architecture still meet the project's constraints, providing continuous validation of the model requirements.

### Testing in GitHub

You can test the model requirements directly in GitHub by:

1. **Manually Triggering the Workflow**:
   - Go to your repository on GitHub
   - Click on the "Actions" tab
   - Select the "Model Requirements Tests" workflow
   - Click "Run workflow" button on the right side
   - Select the branch you want to run the workflow on
   - Click "Run workflow" to start the process

2. **Viewing Results**:
   - The workflow will run and show a green checkmark if all tests pass
   - Click on the completed workflow run to see detailed results
   - Under the "Check results" step, you'll see which models meet all requirements
   - The workflow will fail (red X) if no model meets all requirements

3. **Workflow Triggers**:
   - Automatically runs when code is pushed to main/master branches
   - Automatically runs on pull requests to main/master branches
   - Can be manually triggered as described above

### Running Tests Locally

You can also test model requirements locally using:

```bash
python model_checker.py <path_to_model_file>
```

This will output a detailed report of the model's parameters and architectural features.