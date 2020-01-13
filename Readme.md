# Project Overview

One Paragraph of project description goes here

This project trains classifier model 'LogisticRegression()' on training data by extracted 3 main featuers,  
1-word Frequency.  
2-TF-IDF.  
3-HashingVectorizer.  
All used data were cleaned and pre-proccing by 'Review_preproccess()' method.  
Also SelectKBest method has been applied as a Feature selection technique to minimse the length of featuer vector.  
Then validate the model by predict with development set.  

Dataset used provided by CMT307 instructores as followes:  
"The core dataset contains 25,000 reviews split into train, development  
and test sets. The overall distribution of labels is roughly balanced."  

# Files 
1 folder containes all data sets.(train , development , and test set)  
1 python file (.py) contains all methods and functionality to train, validate, and test the classifier model  
1 README file  

# Prerequisites

Version of dependcies were used

IPython >=4.0     :  7.8.0 (OK)  
cython >=0.21     :  0.29.13 (OK)  
jedi >=0.9.0      :  0.15.1 (OK)  
matplotlib >=2.0.0:  3.1.1 (OK)  
nbconvert >=4.0   :  5.6.0 (OK)  
numpy >=1.7       :  1.17.2 (OK)  
pandas >=0.13.1   :  0.25.1 (OK)  
psutil >=0.3      :  5.6.3 (OK)  
pycodestyle >=2.3 :  2.5.0 (OK)  
pyflakes >=0.6.0  :  2.1.1 (OK)  
pygments >=2.0    :  2.4.2 (OK)  
pylint >=0.25     :  2.4.2 (OK)  
qtconsole >=4.2.0 :  4.5.5 (OK)  
rope >=0.9.4      :  0.14.0 (OK)  
sphinx >=0.6.6    :  2.2.0 (OK)  
sympy >=0.7.3     :  1.4 (OK)  

Project was coded by Spyder3.3.6
Sikit-learn version 0.21.3.
nltk version 3.4.5.

### Running instructions
Place the whole folder on a desired directory,(for example Document folder)  
Open ubuntu terminal, then change directory to folder location (for example $ cd IMDB)  
Mke sure all data set in folder called 'IMDb' and the structure of dataset folder as provided in coursework
then run  code py : $ python c1888251.py   
results of training, development and training will appear on terminal.  
