# Movie-Recommender-System
A Web Base user-item Movie Recommendation Engine using Collaborative Filtering By matrix factorizations algorithm and
The recommendation based on the underlying idea that is if two persons both liked certian common movies,then the movies that one person has liked that the other person has not yet watched can be recommended to him.   
### Screenshot

###### Home page
![home](https://user-images.githubusercontent.com/20842692/45380125-941d7500-b61f-11e8-852d-c09e9586b35b.png)

###### Rating page
![rate](https://user-images.githubusercontent.com/20842692/45380186-be6f3280-b61f-11e8-8ad6-8b967d1cba1a.png)

### Technologies Used

#### Web Technologies
Html , Css , JavaScript , Bootstrap , Django

#### Machine Learning Library In Python3
Numpy , Pandas , Scipy

#### Database
SQLite

##### Requirements
```
python 3.6
pip3
virtualenv
```
##### Setup to run

Extract zip file in your computer

Open terminal/cmd promt

Goto that Path

Example

```
cd ~/Destop/Movie-Recommender-System
```
Create a new virtual environment on that directory

```
virtualenv .
```

Activate Your Virtual Environment

for Linux
```
source bin/activate
```
for Windows
```
cd Scripts
then
activate
```
To install Dependencies

```
pip install -r requirements.txt
```

### Creating Local Server

Goto src directory, example

```
cd ../Movie-Recommender-System/src
```
To run
```
python manage.py runserver
```
Now open your browser and go to this address
```
http://127.0.0.1:8000
```


# Visualization

To deploy the visualization environment, firstly you need to install the node.js (version 14.18.3) and yarn to your computer. Note that node.js with version 17.x will cause error 
when compiling the code of manifold. I test the deployment on Ubuntu 20.04 and macOS 10.15.7.

## Run locally

Please run the following commands to set up your environment.

```
# install all dependencies in the root directory (visualization folder)
yarn
# go to examples/manifold directory
cd examples/manifold
# install dependencies for the app
yarn
# run the app
yarn start
```

![Compile](https://raw.githubusercontent.com/XiangchunChen/MovieRecommend/master/images/compile.jpg)

![interface](https://raw.githubusercontent.com/XiangchunChen/MovieRecommend/master/images/interface.jpg)


## Demo data

The demo data is saved in 'visualization/Demo_data/'. When uploading CSV to manifold, please upload features.csv to the first rectangle, mlp_yPred.csv and cla_yPred.csv to the second rectangle, yTrue.csv to the third rectangle.

