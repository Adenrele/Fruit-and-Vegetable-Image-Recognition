#!/bin/bash
python3.10 -m venv myenv
source myenv/bin/activate #source is not recognised by sh so line 4 shoudl work. 
. myenv/bin/activate #neither of these work in the sh script.
pip install --upgrade pip
pip install kaggle
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
pip install python-dotenv
