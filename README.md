# SVM Visualizer

https://j-zohdi.herokuapp.com/svm_visualizer

This project creates visualizations and provides a web API for a list of support vector machine models. This gives easy access as a tool for analyzing data with visual output. That can be used by anyone with internet access, anywhere, with any skillset. Just input your data and sit back.

The models included are implemented by the python library sci-kit learn.:

- Linear
- Polynomial
- Radial Basis Function
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest

Web API is written in [Flask][1]
Charts by Plotly[2]
Hosted on [Heroku][3]

## [API](https://j-zohdi.herokuapp.com/svm_visualizer#api-endpoints)

![Post Train](/static/api_post_train.PNG)

![Get Retrieve](/static/api_get_retrieve.PNG)

## Preview

#### 2D data

![Preview 2D](/static/preview_2d.PNG)

#### Result

![Preview 2D Result](/static/preview_result.PNG)

#### 3D data

![Preview 3D](/static/preview_3d.PNG)

#### Result

![Preview 3D Result](/static/preview_3d_result.PNG)

## Usage

The intended use is as a web app. But you can install and run locally as long as you set up a mongo database.
The reason for doing this is that the models may take quite some time to compute the result, depending out the input. My solution to providing an API is to send back an ID to the initial request. This ID will then be used to poll the result from the database which will be uploaded once the model is finished computing.

## License

Project is under MIT License

[1]: https://flask.palletsprojects.com/en/1.1.x/
[2]: https://plotly.com/javascript/
[3]: heroku.com
