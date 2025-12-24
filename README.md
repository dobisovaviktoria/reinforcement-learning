# Reinforcement Learning Project

## Authors

- Viktória Dobišová
- Nang Cherry Naw

---

## Project description

This project implements and evaluates reinforcement learning agents on the `MountainCarContinuous-v0` environment using **Stable Baselines3**. Three implementations are included:

- **Baseline** – standard reward function  
- **Custom** – custom reward wrapper  
- **Extension** – advanced/custom modifications  

---

## Prerequisites

- Python 3.12
- Stable Baselines3
- Gymnasium
- PyTorch
- NumPy
- TensorBoard
- PyYAML

---

## Training 
**Note:** You can skip this whole section as we have already run and copied all the results into the local project for you!

### Option 1: Using Google Colab

The training takes too much time even on a GPU, on a local CPU is would be way worse.

1. Open Google Colab
```
https://colab.research.google.com/
```

2. Start a new Notebook
```
Click File -> New Notebook.
```

3. Turn on GPU
```
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

4. Make sure you have correct dependencies
In the code cell run:
```
!pip install stable-baselines3==2.3.2 gymnasium==0.29.1
```

5. Upload your project as a zip file

Upload your saved project zip file
```
from google.colab import files
uploaded = files.upload()
```
Extract the zip file
```
import zipfile

with zipfile.ZipFile("reinforcement_learning.zip", "r") as zip_ref:
    zip_ref.extractall("project")
```
Navigate to the project directory
```
%cd /content/project/reinforcement_learning
```

6. Run the training
```
!python src/train_baseline.py
!python src/train_custom_reward.py
!python src/train_extension.py
```
**Note:** Consider running each training in a new notebook so the training runs in parallel. It will save you time.

7. Download logs and results and paste them into your local project

```
import shutil
from google.colab import files

# Create ZIP archives
print("Creating ZIP files...")
shutil.make_archive('/content/results', 'zip', '/content/project/reinforcement_learning/results')
shutil.make_archive('/content/logs', 'zip', '/content/project/reinforcement_learning/logs')

# Download both ZIPs
print("Downloading results.zip...")
files.download('/content/results.zip')

print("Downloading logs.zip...")
files.download('/content/logs.zip')

print("Downloads complete!")
```

### Option 2: Run locally

You can run the individual training files in your terminal, but it will take significantly longer.

```
python .\src\train_baseline.py
python .\src\train_custom_reward.py
python .\src\train_extension.py
```

**Note:** We went for this option.

## Tensorboard

Now, you can see resulting graphs in tensorboard by using: 

```
tensorboard --logdir logs
```

In case you don't have tensorboard installed on your device, first run:

```
pip install tensorboard
```

## Evaluation
To see the model comparison (outside Tensorboard results) run this command: 

```
python .\src\evaluate.py   
```