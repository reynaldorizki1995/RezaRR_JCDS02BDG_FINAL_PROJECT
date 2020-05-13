import numpy as np
import pandas as pd
import seaborn as sns
import folium
from folium.plugins import FastMarkerCluster
import geopandas as gpd
from branca.colormap import LinearColormap

df_list = pd.read_csv('clean_listing.csv')



lats2018 = df_list['latitude'].tolist()
lons2018 = df_list['longitude'].tolist()
locations = list(zip(lats2018, lons2018))

map1 = folium.Map(location=[52.520008, 13.404954], zoom_start=11.5)
FastMarkerCluster(data=locations).add_to(map1)

map1.save('map1.html')