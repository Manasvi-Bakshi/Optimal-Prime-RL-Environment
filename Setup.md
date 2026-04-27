# OpenEnv Project Setup Guide

This guide will help you set up and run the OpenEnv evaluation environment locally.


## 1. Install Anaconda & Create Environment

Make sure Anaconda is installed.

Verify:

```
conda --version
```

Create and activate environment:

```
conda create -n openenv python=3.10 -y
conda activate openenv

(C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1) ; (conda activate openenv)
```

You should see:

```
(openenv)
```

---

## 2. Install Dependencies

Install required Python packages:

```
pip install fastapi uvicorn requests
```

---

## 3. VS Code Setup (Important)

Open the project folder in VS Code.

Then:

* Press `Ctrl + Shift + P`
* Search: `Python: Select Interpreter`
* Select:

  ```
  Python 3.10 (openenv)
  ```
---

## 4. Install Docker

Install Docker Desktop and restart your system.

Verify installation:

```
docker --version
docker run hello-world
```

---

## 5. Build Docker Image

From the project root directory:

```
docker build -t openenv .
```

---

## 6. Run Container

```
docker run -p 7860:7860 openenv
```

---

## 7. Validate the Setup

In a new terminal:

```
python validator.py
```

Expected:

```
All checks passed
```
### Baseline Agent

```
python baseline/run.py
```

---

## Notes

* Always activate the environment before running anything:

  ```
  conda activate openenv
  ```

* Do not use Git Bash for setup; prefer:

  * Command Prompt
  * PowerShell

* Ensure Docker is running before building.

