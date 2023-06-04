### Earth Surface Temperature

This is a spin-off from Geospatial project. 

I intended to add interactive dashboards of business data to my portfolio, but it turned out climate data is more readily available than business data, so this was done first.

To use the dashboard app:

1. With or withour virtualenv, install requirements with ```pip install -r requirements.txt```
   
2. Setup your own kaggle API (For Windows, you'll have to save ```kaggle.json``` into ```.kaggle``` folder inside your user folder), then use the ```Load from Kaggle``` checkbox inside the app.
       
3. Run ```streamlit run earth_surface_temperature.py``` after installing the requirements

Alternatively, try the online app on [cloud](https://berkeley-data-earth-temperature.streamlit.app/)

#### Preview images

![Min and max temperature, averaged over cities](preview_images/minmax_yearly_temperature.jpg)

![Cities temperature record on map](preview_images/cities_temperature_map.jpg)
