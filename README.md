1. Install Anaconda and kernel environment
conda --version
conda create -n openenv python=3.10 -y
conda activate openenv

"(openenv) C:\Users\manas>"

2. Install Dependencies
 pip install fastapi uvicorn
 pip install requests

3. VS Code Setup
In VS Code:

Press Ctrl + Shift + P
Search → Python: Select Interpreter
Choose:
Python 3.10 (openenv)