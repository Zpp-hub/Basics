####Requirements
### 独立环境
## linux
python -m venv .env  
source .env/bin/activate  
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

## windows
python -m venv .env  
cd .env/Scripts  
activate.bat  
cd ../..  
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

#### 生成requirements.txt
pip freeze > requirements.txt
