@echo off

python -m venv venv
call venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo Virtual environment created and activated.
pip install -r requirements.txt
echo Dependencies installed from requirements.txt.
python goobworkshop.py