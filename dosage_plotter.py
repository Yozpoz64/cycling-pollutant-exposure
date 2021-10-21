import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import fiona
import geopandas as gpd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import random
import statistics as stats
import cartopy.crs as ccrs
from matplotlib_scalebar.scalebar import ScaleBar
import pandas as pd
from matplotlib.lines import Line2D
import haversine as hs
from haversine import Unit
from pyproj import Geod
import seaborn as sns

plt.rc('axes', labelsize=12)
plt.rc('figure', titlesize=25)

def calc_distance(points):
    all_distances = []
    for i in range(len(points)):
        if i != len(points) - 1:
            dist = hs.haversine(points[i], points[i + 1], unit=Unit.METERS)
            all_distances.append(dist)
            
    return round((sum(all_distances)), 2)
        
def get_dosage(gdf, line_index, datetime_index, temporal_interval):   
    temporal_interval = int(temporal_interval.split('s')[0])
   
    gdf_array = gdf.to_numpy()
        
    lengths = []
    times = []
    
    geod = Geod(ellps="WGS84")
    
    for item in gdf_array:
        #lat, lon = item[-2].y, item[-2].x
        #points.append((lat, lon))
        line_length = geod.geometry_length(item[line_index])
        lengths.append(line_length)
        
    end = pd.to_datetime(gdf_array[-1][datetime_index]).timestamp()
    start = pd.to_datetime(gdf_array[0][datetime_index]).timestamp()
    
    time = end - start
    
    if time <= 0:
        # default if only one point - set to 
        time = temporal_interval - 0.5
    
    # m
    #ride_distance = calc_distance(points)
    ride_distance = sum(lengths)

    # m/s
    ride_velocity = ride_distance / time
    
    #print(len(gdf))
    
    # no2
    # np.nanmean gives error here for some reason. I think it is when the length of a dataframe is 1
    #   it gets the individual value rather than the series, so it cannot access dtype
    ride_no2 = stats.mean(gdf['RASTERVALU'])
    
    # ventilation multiplier
    vr = 2
    
    #print(ride_distance, ride_velocity, ride_no2)
    # model
    dosage = (ride_distance / ride_velocity) * ride_no2 * vr
    
    return dosage


def plot_dosage_roads():
    gdb = '/home/sophie/GitHub/uninotes_2021/sem2/gisci399/arc/segmentation/final_segments.gdb/'
    
    layers = fiona.listlayers(gdb)
    
    #layers_of_interest = ['roads_5s_20m', 'roads_60s_20m', 'roads_300s_20m']
    layers_of_interest = ['roads_5s_20m', 'roads_5s_50m', 'roads_5s_300m']
    roads_of_interest = ['Domain Drive', 'Remuera Road',
                         'Kohimarama Road', 'Tamaki Drive']
    
    data = defaultdict(dict)
    
    # bar chart
    fig, ax = plt.subplots(1, figsize=(10, 10))
    
    layer_col = []
    road_col = []
    dose_col = []
    time_col = []
    spatial_col = []
    
    for layer in layers:
        if layer in layers_of_interest:
            gdf = gpd.read_file(gdb, layer=layer)
            gdf.dropna(inplace=True)
            gdf['DateTimeS'] = pd.to_datetime(gdf['DateTimeS'])
            gdf['epoch'] = gdf['DateTimeS'].apply(lambda x: x.timestamp())
            gdf = gdf.sort_values('epoch', axis=0)
            gdf = gdf.reset_index()
            gdf = gdf.loc[gdf['full_road_'].isin(roads_of_interest)]
            df = gdf.groupby('full_road_', as_index=False)['RASTERVALU'].mean()
            
            temporal = layer.split('_')[1]
            spatial = layer.split('_')[2]
            
            label = '{}, {}'.format(temporal, spatial)
            
            df['dosage'] = [i for i in range(len(df))]
            df = df.set_index('full_road_')

            geom_index = list(gdf.columns).index('geometry')
            time_index = list(gdf.columns).index('DateTimeS')
            
            #print(gdf.drop_duplicates(subset='full_road_')['full_road_'])
             
            for road in roads_of_interest:
                this_road = gdf.loc[gdf['full_road_'] == road]
          
                if len(this_road) > 0:
                    dosage = get_dosage(this_road, geom_index, time_index, temporal)
                    df.loc[road, 'dosage'] = round(dosage, 0)
                    
                    label = '{}, {}'.format(temporal, spatial)
                    layer_col.append(label)
                    road_col.append(road)
                    dose_col.append(dosage)
                    time_col.append(int(temporal.split('s')[0]))
                    spatial_col.append(int(spatial.split('m')[0]))
                       
    df = pd.DataFrame([layer_col, time_col, spatial_col, road_col, dose_col]).T
    
    # sort by (1=time, 2=space)
    df = df.sort_values(by=2, axis=0)
    
    sns.barplot(x=3, y=4, hue=0, data=df, ax=ax)
    
    #ax.bar(df.index, df['dosage'], alpha=0.5, label=label)
    data[layer] = df

    #sns.factorplot(x='variable', y='value')
    
    #ax.legend()
    ax.set_xlabel('Street Name')
    ax.set_ylabel('Predicted NO$_2$ Exposure (\u00B5g)')


def plot_dosage_track():
    gdb = '/home/sophie/GitHub/uninotes_2021/sem2/gisci399/arc/segmentation/final_segments.gdb/'
    
    layers = fiona.listlayers(gdb)
    
    temporal = False
    '''
    layers_of_interest = ['roads_1s_20m', 'roads_5s_20m', 'roads_10s_20m', 
                          'roads_60s_20m', 'roads_120s_20m', 'roads_300s_20m']
    '''
    
    layers_of_interest = ['roads_5s_20m', 'roads_5s_50m', 'roads_5s_100m', 
                          'roads_5s_300m']
    
    layer_col = []
    #dose_col = []
    time_col = []
    spatial_col = []
    
    all_gdfs = defaultdict(dict)
    
    for layer in layers:
        if layer in layers_of_interest:
            gdf = gpd.read_file(gdb, layer=layer)
            gdf.dropna(inplace=True)
            gdf['DateTimeS'] = pd.to_datetime(gdf['DateTimeS'])
            gdf['epoch'] = gdf['DateTimeS'].apply(lambda x: x.timestamp())
            gdf = gdf.sort_values('epoch', axis=0)
            gdf = gdf.reset_index()
            
            temporal = layer.split('_')[1]
            spatial = layer.split('_')[2]
            
            label = '{}, {}'.format(temporal, spatial)
            
            geom_index = list(gdf.columns).index('geometry')
            time_index = list(gdf.columns).index('DateTimeS')
            
            dosage = get_dosage(gdf, geom_index, time_index, temporal)
            
            gdf_array = gdf.to_numpy()
            
            dose_col = []

            for item in gdf_array:
                df = (pd.DataFrame(item).T)
                df.columns = gdf.columns
                
                this_dose = get_dosage(df, geom_index, time_index, temporal)
                dose_col.append(this_dose)
                
            gdf['dosage'] = dose_col
            
            all_gdfs[layer] = gdf
            
    fig, ax = plt.subplots(1, figsize=(13, 8))
    
    for layer in layers_of_interest:
        gdf = all_gdfs[layer]
        ax.plot(gdf['epoch'], gdf['dosage'], label=layer)
        
    ax.legend()
    
    min_dose = []
    max_dose = []
    for gdf in all_gdfs.values():
        min_dose.append(min(gdf['dosage']))
        max_dose.append(max(gdf['dosage']))
    
    if not temporal:
        fig, ax = plt.subplots(3, 2, figsize=(20, 15), sharex=True, sharey=True)
    
    
        titles = ['1s Interval', '5s Interval', '10s Interval', '60s Interval', '120s Interval', '300s Interval']
        
        all_gdfs[layers_of_interest[0]].plot(ax=ax[0, 0], column='RASTERVALU', cmap='viridis_r')
        ax[0, 0].set_title(titles[0])
        
        all_gdfs[layers_of_interest[1]].plot(ax=ax[0, 1], column='RASTERVALU', cmap='viridis_r')
        ax[0, 1].set_title(titles[1])
        
        all_gdfs[layers_of_interest[2]].plot(ax=ax[1, 0], column='RASTERVALU', cmap='viridis_r')
        ax[1, 0].set_title(titles[2])
        
        all_gdfs[layers_of_interest[3]].plot(ax=ax[1, 1], column='RASTERVALU', cmap='viridis_r')
        ax[1, 1].set_title(titles[3])
        
        all_gdfs[layers_of_interest[4]].plot(ax=ax[2, 0], column='RASTERVALU', cmap='viridis_r')
        ax[2, 0].set_title(titles[4])
        
        all_gdfs[layers_of_interest[5]].plot(ax=ax[2, 1], column='RASTERVALU', cmap='viridis_r')
        ax[2, 1].set_title(titles[5])
        
    else:
        # MAP
        vmin, vmax = 10, 45
            
        fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharex=True, sharey=True)
        
        layers_list = list(all_gdfs.values())
        
        layers_list[0].plot(ax=ax[0, 0], column='RASTERVALU', cmap='viridis_r', 
                            vmin=vmin, vmax=vmax)
        ax[0, 0].set_title('20m Segments')
        
        layers_list[1].plot(ax=ax[0, 1], column='RASTERVALU', cmap='viridis_r',
                            vmin=vmin, vmax=vmax)
        ax[0, 1].set_title('50m Segments')
        
        layers_list[2].plot(ax=ax[1, 0], column='RASTERVALU', cmap='viridis_r',
                            vmin=vmin, vmax=vmax)
        ax[1, 0].set_title('100m Segments')
        
        layers_list[3].plot(ax=ax[1, 1], column='RASTERVALU', cmap='viridis_r',
                            vmin=vmin, vmax=vmax)
        ax[1, 1].set_title('300m Segments')
        
    
    # set labels
    fig.text(0.49, 0.08, 'Longitude', ha='center', fontsize=22)
    fig.text(0.075, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=22)

    # add legend (colour map)
    norm = mpl.colors.Normalize(vmin=min(min_dose), vmax=max(max_dose))
    cbar = plt.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r'),
                ax=ax, pad=0.05, fraction=0.02,
                label='NO$_2$ Dosage (\u00B5g)')
    
    # set labels
    fig.text(0.49, 0.08, 'Longitude', ha='center', fontsize=22)
    fig.text(0.075, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=22)

    '''    
    # plot whole track time series
    fig, ax = plt.subplots(1)
    
    for gdf in all_gdfs.values():
        ax.plot(gdf['epoch'], gdf['dosage'])
    '''
    
    

plot_dosage_track()
#plot_dosage_roads()
#plot_temporal()
#plot_by_road()
#plot_two_segments()
#plot_all_segments()
#plot_spatial_segments()

