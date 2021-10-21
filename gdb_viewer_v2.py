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


def get_random_colours(num_colours):
    return ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
         for i in range(num_colours)]


def get_binary_colours(num_segments, c1, c2):
    colours = []
    for i in range(num_segments):
        if i % 2 == 0:
            colours.append(c1)
        else:
            colours.append(c2)
    return colours


def plot_no2():
    
    gpx_gdb = '/home/sophie/GitHub/uninotes_2021/sem2/gisci399/arc/no2_gpx_fusion.gdb/'
    
    layers = fiona.listlayers(gpx_gdb)
    
    data = defaultdict(dict)
                
            
    # get data from geodatabase
    for layer in layers: 
        if '_no2' in layer:
            gdf = gpd.read_file(gpx_gdb, layer=layer)
            gdf = gdf.dropna()
            data[layer] = gdf
    
    fig, ax = plt.subplots(figsize=(20, 20))
  
    vmin = 10
    vmax = 45
    
    for item in data.values():
        item.plot(ax=ax, column='RASTERVALU', cmap='viridis_r',
                 vmin=vmin, vmax=vmax)
    
    # set labels
    ax.set_title('Predicted NO$_2$ Exposure while Cycling along Selected Routes in Auckland', fontsize=25)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # add legend (colour map)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r'),
                ax=ax, pad=0.05, fraction=0.03,
                label='NO$_2$ (\u00B5g / m$^3$)')
    
    # add scale bar
    scalebar = ScaleBar(70000)
    plt.gca().add_artist(scalebar)
    
    # add north arrow
    x, y, arrow_length = 0.985, 0.07, 0.05
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=12),
            ha='center', va='center', fontsize=22,
            xycoords=ax.transAxes)



def plot_spatial_segments():
    
    seg_gdb = '/home/sophie/GitHub/uninotes_2021/sem2/gisci399/arc/segmentation/no2_road_segmentation.gdb/'
    
    layers = fiona.listlayers(seg_gdb)
    
    data = defaultdict(dict)
    
    for layer in layers:
        gdf = gpd.read_file(seg_gdb, layer=layer)
        gdf.dropna(inplace=True)
        
        gdf = gdf.sort_values('DateTimeS', axis=0)
        gdf.reset_index(inplace=True)
        #gdf = gdf.dropna()
        data[layer] = gdf
        
    
    # MAP
    vmin, vmax = 10, 45
        
    fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharex=True, sharey=True)
    
    layers_list = list(data.values())
    
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
    
    # add legend (colour map)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r'),
                ax=ax, pad=0.05, fraction=0.02,
                label='NO$_2$ (\u00B5g / m$^3$)')
    
    # set labels
    fig.text(0.49, 0.08, 'Longitude', ha='center', fontsize=22)
    fig.text(0.075, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=22)


    # PLOT OF RASTERVALUE OVER TIME
    fig, ax = plt.subplots(1, figsize=(13, 8))

    
    segs = ['20m', '50m', '100m', '300m']
    i = 0
    
    for layer in layers_list:
        layer['DateTimeS'] = pd.to_datetime(layer['DateTimeS'])
        layer['epoch'] = layer['DateTimeS'].apply(lambda x: x.timestamp())
        label = 'Segment: {} (mean NO$_2$: {})'.format(segs[i], round(np.nanmean(layer['RASTERVALU']), 1))
        ax.plot(layer['epoch'], layer['RASTERVALU'], label=label)
        i += 1
    
    ax.legend()
    ax.set_title('Track A')
    ax.set_ylabel('NO$_2$ (\u00B5g / m$^3$)')
    ax.set_xlabel('Time (Unix Epoch)')

    
    
def plot_all_segments():
    gdb = '/home/sophie/GitHub/uninotes_2021/sem2/gisci399/arc/segmentation/final_segments.gdb/'
    layers = fiona.listlayers(gdb)
    
    data = defaultdict(dict)
    
    colours = {
        '300m': 'red',
        '100m': 'orange',
        '50m': 'blue',
        '20m': 'green'}
    
    for layer in layers:
        gdf = gpd.read_file(gdb, layer=layer)
        gdf = gdf.dropna()
        gdf['DateTimeS'] = pd.to_datetime(gdf['DateTimeS'])
        gdf['epoch'] = gdf['DateTimeS'].apply(lambda x: x.timestamp())
        gdf = gdf.sort_values('epoch', axis=0)
        gdf = gdf.reset_index()
        
        temporal = layer.split('_')[1]
        spatial = layer.split('_')[2]
        
        data[temporal][spatial] = gdf
    
    fig, ax = plt.subplots(3, 2, figsize=(20, 20), sharex=True, sharey=True)
    
    
    for seg in data['1s']:
        ax[0, 0].plot(data['1s'][seg]['epoch'], data['1s'][seg]['RASTERVALU'], label=seg, color=colours[seg])
        
    for seg in data['5s']:
        ax[0, 1].plot(data['5s'][seg]['epoch'], data['5s'][seg]['RASTERVALU'], label=seg, color=colours[seg])
        
    for seg in data['10s']:
        ax[1, 0].plot(data['10s'][seg]['epoch'], data['10s'][seg]['RASTERVALU'], label=seg, color=colours[seg])
        
    for seg in data['60s']:
        ax[1, 1].plot(data['60s'][seg]['epoch'], data['60s'][seg]['RASTERVALU'], label=seg, color=colours[seg])
        
    for seg in data['120s']:
        ax[2, 0].plot(data['120s'][seg]['epoch'], data['120s'][seg]['RASTERVALU'], label=seg, color=colours[seg])
        
    for seg in data['300s']:
        ax[2, 1].plot(data['300s'][seg]['epoch'], data['300s'][seg]['RASTERVALU'], label=seg, color=colours[seg])
        
    font = 18
    ax[0, 0].set_title('1 Second', fontsize=font)
    ax[0, 1].set_title('5 Seconds', fontsize=font)
    ax[1, 0].set_title('10 Seconds', fontsize=font)
    ax[1, 1].set_title('60 Seconds', fontsize=font)
    ax[2, 0].set_title('120 Seconds', fontsize=font)
    ax[2, 1].set_title('300 Seconds', fontsize=font)
    
    lines = []
    labels = []
    for colour in colours:
        lines.append(Line2D([0], [0], color=colours[colour], lw=3))
        labels.append(colour)
        
    fig.legend(lines, labels, loc='center right', prop={'size': 19})
    fig.text(0.51, 0.08, 'Time (Unix Epoch)', ha='center', fontsize=font)
    fig.text(0.075, 0.5, 'NO$_2$ (\u00B5g / m$^3$)', va='center', rotation='vertical', fontsize=font)
    
    
def plot_two_segments():
    gdb = '/home/sophie/GitHub/uninotes_2021/sem2/gisci399/arc/segmentation/final_segments.gdb/'
    layers = fiona.listlayers(gdb)
    
    data = defaultdict(dict)
    
    colours = {
        '300m': 'red',
        '100m': 'orange',
        '50m': 'blue',
        '20m': 'green'}
    
    for layer in layers:
        gdf = gpd.read_file(gdb, layer=layer)
        gdf = gdf.dropna()
        gdf['DateTimeS'] = pd.to_datetime(gdf['DateTimeS'])
        gdf['epoch'] = gdf['DateTimeS'].apply(lambda x: x.timestamp())
        gdf = gdf.sort_values('epoch', axis=0)
        gdf = gdf.reset_index()
        
        temporal = layer.split('_')[1]
        spatial = layer.split('_')[2]
        
        data[temporal][spatial] = gdf
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [1.5, 0.7]})
    vmin, vmax = 10, 45
    
    data['10s']['50m'].plot(ax=ax[0, 0], column='RASTERVALU', cmap='viridis_r',
                        vmin=vmin, vmax=vmax)
    ax[0, 1].plot(data['10s']['50m']['epoch'], data['10s']['50m']['RASTERVALU'])
    ax[0, 1].set_ylabel('NO$_2$ (\u00B5g / m$^3$)')
    ax[0, 1].set_xlabel('Time (Unix Epoch)', labelpad=15)
    ax[0, 0].set(xlabel='Longitude', ylabel='Latitude')
    
    data['60s']['50m'].plot(ax=ax[1, 0], column='RASTERVALU', cmap='viridis_r',
                        vmin=vmin, vmax=vmax)
    ax[1, 1].plot(data['60s']['50m']['epoch'], data['60s']['50m']['RASTERVALU'])
    ax[1, 1].set_ylabel('NO$_2$ (\u00B5g / m$^3$)')
    ax[1, 1].set_xlabel('Time (Unix Epoch)', labelpad=15)
    ax[1, 0].set(xlabel='Longitude', ylabel='Latitude')


    fig.text(0.5, 0.5, s='60 Seconds, 50 Metres Segmented', ha='center', fontsize=18)
    fig.text(0.5, 0.97, s='10 Seconds, 50 Metres Segmented', ha='center', fontsize=18)
    
    
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r'),
                ax=ax[1, 0], pad=0.02, fraction=0.1,
                label='NO$_2$ (\u00B5g / m$^3$)')
    cbar = plt.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r'),
                ax=ax[0, 0], pad=0.02, fraction=0.1,
                label='NO$_2$ (\u00B5g / m$^3$)')
    
    fig.tight_layout(pad=4)
    
    
    # add scale bar
    scalebar1 = ScaleBar(100000)
    ax[0, 0].add_artist(scalebar1)
    scalebar2 = ScaleBar(100000)
    ax[1, 0].add_artist(scalebar2)
    
    # add north arrow
    x, y, arrow_length = 0.98, 0.14, 0.1
    ax[0, 0].annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=3, headwidth=7),
            ha='center', va='center', fontsize=12,
            xycoords=ax[0, 0].transAxes)
    ax[1, 0].annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=3, headwidth=7),
            ha='center', va='center', fontsize=12,
            xycoords=ax[1, 0].transAxes)
    
    
def plot_by_road():
    gdb = '/home/sophie/GitHub/uninotes_2021/sem2/gisci399/arc/segmentation/final_segments.gdb/'
    
    layers = fiona.listlayers(gdb)
    
    layers_of_interest = ['roads_10s_20m', 'roads_60s_20m']
    roads_of_interest = ['Symonds Street', 'Domain Drive', 'Remuera Road',
                         'Kohimarama Road', 'Tamaki Drive', 'Queen Street']
    
    data = defaultdict(dict)
    
    # bar chart
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    
    road_highlight = 'Remuera Road'
    
    layer_col = []
    road_col = []
    no2_col = []
    
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
            
            data[temporal][spatial] = gdf
            
            tmp = defaultdict(dict)
            
            label = '{}, {}'.format(temporal, spatial)
            
            for road in roads_of_interest:
                this_road = gdf.loc[gdf['full_road_'] == road]
                
                if len(this_road) > 0:
                    layer_col.append(label)
                    road_col.append(road)
                    no2_col.append(stats.mean(this_road['RASTERVALU']))
                    
            #ax[0].bar(df['full_road_'], df['RASTERVALU'], alpha=0.5, label=label)
            
            series = gdf.loc[gdf['full_road_'] == road_highlight]
            ax[1].plot(series['epoch'], series['RASTERVALU'], label=label)
            
    df = pd.DataFrame([layer_col, road_col, no2_col]).T
    
    sns.barplot(x=1, y=2, hue=0, data=df, ax=ax[0])
            
    ax[0].legend()
    ax[0].set_xlabel('Street Name', fontsize=15)
    ax[0].set_ylabel('Average NO$_2$ (\u00B5g / m$^3$)', fontsize=15)
    ax[0].tick_params(axis='x', labelrotation=45, labelsize=13)
    
    ax[1].legend()
    ax[1].set_title('{}'.format(road_highlight))
    ax[1].set_xlabel('Time (Unix Epoch)')
    ax[1].set_ylabel('NO$_2$ (\u00B5g / m$^3$)')
    
    
def plot_temporal():
    gdb = '/home/sophie/GitHub/uninotes_2021/sem2/gisci399/arc/segmentation/temporal.gdb/'
    
    layers = fiona.listlayers(gdb)
    
    layers = ['roads_1s', 'roads_5s', 'roads_10s', 'roads_60s', 'roads_120s', 'roads_300s']
    
    data = defaultdict(dict)
    
    for layer in layers:
        gdf = gpd.read_file(gdb, layer=layer)
        gdf.dropna(inplace=True)
        gdf['DateTimeS'] = pd.to_datetime(gdf['DateTimeS'])
        gdf['epoch'] = gdf['DateTimeS'].apply(lambda x: x.timestamp())
        gdf = gdf.sort_values('epoch', axis=0)
        gdf = gdf.reset_index()
        
        time = layer.split('_')[1]
        
        data[time] = gdf
        
        
    fig, ax = plt.subplots(1, figsize=(13, 8))
     
    for gdf in data:
        label = 'Interval: {} (mean NO$_2$: {})'.format(gdf, round(stats.mean(data[gdf]['RASTERVALU']), 2))
        ax.plot(data[gdf]['epoch'], data[gdf]['RASTERVALU'], label=label)
        
    ax.legend()
    ax.set_title('Track A')
    ax.set_xlabel('Time (Unix Epoch)')
    ax.set_ylabel('NO$_2$ (\u00B5g / m$^3$)')
    
    
    # MAP
    vmin, vmax = 10, 45
        
    fig, ax = plt.subplots(3, 2, figsize=(20, 15), sharex=True, sharey=True)
    
    data['1s'].plot(ax=ax[0, 0], column='RASTERVALU', cmap='viridis_r', 
                        vmin=vmin, vmax=vmax)
    ax[0, 0].set_title('1s Interval')
    
    data['5s'].plot(ax=ax[0, 1], column='RASTERVALU', cmap='viridis_r',
                        vmin=vmin, vmax=vmax)
    ax[0, 1].set_title('5s Interval')
    
    data['10s'].plot(ax=ax[1, 0], column='RASTERVALU', cmap='viridis_r',
                        vmin=vmin, vmax=vmax)
    ax[1, 0].set_title('10s Interval')
    
    data['60s'].plot(ax=ax[1, 1], column='RASTERVALU', cmap='viridis_r',
                        vmin=vmin, vmax=vmax)
    ax[1, 1].set_title('60s Interval')
    
    data['120s'].plot(ax=ax[2, 0], column='RASTERVALU', cmap='viridis_r',
                        vmin=vmin, vmax=vmax)
    ax[2, 0].set_title('60s Interval')
    
    data['300s'].plot(ax=ax[2, 1], column='RASTERVALU', cmap='viridis_r',
                        vmin=vmin, vmax=vmax)
    ax[2, 1].set_title('300s Interval')
    
    # add legend (colour map)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r'),
                ax=ax, pad=0.05, fraction=0.02,
                label='NO$_2$ (\u00B5g / m$^3$)')
    
    # set labels
    fig.text(0.49, 0.08, 'Longitude', ha='center', fontsize=22)
    fig.text(0.075, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=22)


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

