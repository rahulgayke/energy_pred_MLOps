

# Introduction:

This project predicts the energy consumption a building would consume in terms of the given building characteristics.

## Startup:
**The best way to start the application is through creating a conda envireonment, for example:** (This process has been tested on windows but should be similar in linux versions.)

```conda create -n residential-building-energy-labeling Python=3.9.6 ```

The dependencies required to run the app are listed in the requirements.txt. simply run the following:

``` pip install -r requirements.txt ```

After installing the required dependencies, run the application by:

```python app.py```

You can alternatively dockerize the application. The dockerfile and docker-compose files are given. Simply run:

``` sudo docker-compose up ```
