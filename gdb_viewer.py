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
from matplotlib.lines import Line2D

file = '/home/sophie/GitHub/uninotes_2021/sem2/gisci399/arc/no2_gpx_fusion.gdb/'

plt.rc('axes', labelsize=22)
plt.rc('figure', titlesize=25)

layers = fiona.listlayers(file)

data = defaultdict(dict)


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
    
    fig, ax = plt.subplots(figsize=(20, 20))
 
    vmin = 10
    vmax = 45
    
    for item in data:
        if item != 'T20210905190046_no2':
            data[item].plot(ax=ax, column='RASTERVALU', cmap='viridis_r',
                     vmin=vmin, vmax=vmax)
        else:
            data[item].plot(ax=ax, color='red', lw=10)
            data[item].plot(ax=ax, column='RASTERVALU', cmap='viridis_r',
                     vmin=vmin, vmax=vmax)
            
    lines = [Line2D([0], [0], color='red', lw=3)]
    labels = ['Track A']

        
    ax.legend(lines, labels, loc='lower left', prop={'size': 20})
    
    
    # set labels
    ax.set_title('Predicted NO$_2$ Exposure while Cycling along Selected Routes in Auckland', fontsize=25)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # add legend (colour map)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r'),
                ax=ax, pad=0.05, fraction=0.03,
                label='NO$_2$ (\u00B5g / m$^3$')
    
    # add scale bar
    scalebar = ScaleBar(70000)
    plt.gca().add_artist(scalebar)
    
    # add north arrow
    x, y, arrow_length = 0.985, 0.07, 0.05
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=12),
            ha='center', va='center', fontsize=22,
            xycoords=ax.transAxes)

   
def plot_segments():
    fig, ax = plt.subplots(2, figsize=(20, 20), sharex=True, sharey=True)
    
    top_interval = 60
    segment(top_interval)
    for track in data.values():
        
        vmin = 0
        vmax = list(track['Segment'])[-1]
    
        #colours = get_random_colours(vmax)
        colours = get_binary_colours(vmax, 'orange', 'black')
        
        #cmap = mpl.colors.LinearSegmentedColormap.from_list("", colours)
        cmap = mpl.colors.ListedColormap(colours)
        track.plot(ax=ax[0], column='Segment', cmap=cmap,
                  vmin=vmin, vmax=vmax)
        
    # add scale bar
    scalebar = ScaleBar(70000)
    ax[0].add_artist(scalebar)
    
    # add north arrow
    x, y, arrow_length = 0.985, 0.07, 0.05
    ax[0].annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=12),
            ha='center', va='center', fontsize=22,
            xycoords=ax[0].transAxes)
    
    bottom_interval = 300
    segment(bottom_interval)
    for track in data.values():
        
        vmin = 0
        vmax = list(track['Segment'])[-1]
    
        #colours = get_random_colours(vmax)
        colours = get_binary_colours(vmax, 'orange', 'black')
        
        #cmap = mpl.colors.LinearSegmentedColormap.from_list("", colours)
        cmap = mpl.colors.ListedColormap(colours)
        track.plot(ax=ax[1], column='Segment', cmap=cmap,
                  vmin=vmin, vmax=vmax)
        
    # add scale bar
    scalebar = ScaleBar(70000)
    ax[1].add_artist(scalebar)
    
    # add north arrow
    x, y, arrow_length = 0.985, 0.07, 0.05
    ax[1].annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=12),
            ha='center', va='center', fontsize=22,
            xycoords=ax[1].transAxes)
    
    
    # set labels
    fig.text(0.52, 0.08, 'Longitude', ha='center', fontsize=22)
    fig.text(0.18, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=22)
    ax[0].set_title('{} Second Segmenting'.format(top_interval), fontsize=22)
    ax[1].set_title('{} Second Segmenting'.format(bottom_interval), fontsize=22)
    
    # add legends
    custom_lines = [mpl.lines.Line2D([0], [0], color='orange'),
                    mpl.lines.Line2D([0], [0], color='black')]
    ax[1].legend(custom_lines, ['Segment A', 'Segment B'], loc='lower left')
    ax[1].legend(custom_lines, ['Segment A', 'Segment B'], loc='lower left')
        
    
def segment(interval):
    for track in data.values():
        #track = track.dropna()
        
        seg_id = 0
        seg_list = []
        
        count = 0
        for i in range(len(track)):
            
            # not sure if this makes sense
            i = i + 1
            
            seg_list.append(seg_id)
       
            if i != 0 and i % interval == 0:
                seg_id += 1
    
     
        track['Segment'] = seg_list
        
        

def rasterval_byseg():
    # for each file in the geodatabase
    for track in data.values():
        track = track.to_numpy()
        
        # drop unnecessary columns
        track = np.delete(track, [0, 1, 2, 3, 4, 6], axis=1)
        
        prev_seg = 0
        values = []
        this_seg_values = []

        # for each point in the track
        for x in track:
            if x[4] != prev_seg:
                for i in range(interval):
                    values.append(stats.mean(this_seg_values))
                
                this_seg_values = []
                prev_seg = x[4]
            else:
                this_seg_values.append(x[2])

        # this returns an uneven number so I cannot append as column
        print(len(track), len(values), len(track) - len(values))
        
            
        
# get data from geodatabase
for layer in layers: 
    if '_no2' in layer:
        gdf = gpd.read_file(file, layer=layer)
        gdf = gdf.dropna()
        data[layer] = gdf
        

# interval between segments, in seconds
interval = 60

# segment track
#segment(interval)

#rasterval_byseg()

plot_no2()

#plot_segments()

        

           



