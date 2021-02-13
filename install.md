[![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)](https://www.python.org/downloads/release/python-360/)
![](https://img.shields.io/badge/Invo--AI-Final%20Presentation-green)
![](https://img.shields.io/badge/Invo--AI-install--instruction-yellowgreen)

# DOC SCANNER
A python command line application to convert scanned invoice pdf to spreadsheet

## Installation

Clone the git repository:

```
git clone https://github.com/Gauravism2017/Invo-AI.git
cd Invo-AI
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```
Using Conda 
```
conda install --file requirements.txt
```
#### Note:
We need to install tesseract and pass executable as argument or in ```{root}/cfg/cofing.py``` as ```TESS_EXEC``` \
For Windows we need to installer poppler , we can install using conda
```
conda install -c conda-forge poppler
```
#### Traindata used in this project. Use LSTM based model for better result [eng.traindata](https://github.com/tesseract-ocr/tessdata_best/raw/master/eng.traineddata). Place it in tessdata folder in tesseract-ocr installation folder.   
#### tesseract version used : v5.0.0

### To use default UI
Install nodejs and yarn(instruction for each OS can be found on their website).
```
cd UI
yarn 
yarn start
```



## Usage with terminal

```python
python invoice.py -i [pdf file name] -o [output folder]
```



