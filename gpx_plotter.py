'''
plots gpx files from input folder
can select either a static (matplotlib) or dynamic (folium) map

NOTES: 
    -get video from autographer photos using command: ffmpeg -framerate 10 -pattern_type glob -i '*.JPG' video.mp4
    -convert this video to .gif using: ffmpeg -i video.mp4 -vf "fps=15,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif

'''
# need quite a few for static cartography, as well as local secondary methods
import matplotlib.pyplot as plt
import fiona
import glob
from shapely.geometry import shape
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os

# folium
import webbrowser # for showing folium maps easily
import folium
import gpxpy

# set cwd. having weird issues with conda this should fix
os.chdir('/home/sophie/GitHub/uninotes_2021/sem2/gisci399/code/')
from custom_scalebar import scale_bar

# constants
CENTRE = (-36.85, 174.75)
EXTENT = [-185.363388, -185.103836, -36.996520, -36.819180] # http://bboxfinder.com/
CRS = ccrs.PlateCarree()
FOLDER = 'data/gpxs/'
COLOURS = ['red', 'blue', 'green']
MAP_TYPE = 'dynamic'
HOVER = True
LINE_WEIGHT = 5
LINE_OPACITY = 0.5
ZOOM_START = 12


# get files with extension in a folder
def get_files(folder, extension):
    glob_arg = folder + '*.' + extension
    return glob.glob(glob_arg)


# extracts trackpoints from a gpx file (the easy way)
def get_track(file_location):
    file = fiona.open(file_location, layer='tracks')
    points = {'type': 'MultiLineString', 
              'coordinates': file[0]['geometry']['coordinates']}
    gpx_shp = shape(points)
    return gpx_shp


# extracts trackpoints from a gpx file (the hard way, for folium)
def get_polyline(file_location):
    file = open(file_location, 'r')
    gpx = gpxpy.parse(file)
    points = []
    
    raw_start_time = gpx.tracks[0].segments[0].points[0].time
    raw_end_time = gpx.tracks[0].segments[0].points[-1].time
    
    length_hours = str(round((raw_end_time - raw_start_time)
                             .seconds / 60 / 60, 2)).split('.')
    length_mins = str((float(length_hours[1]) * 0.01) * 60).split('.')[0]
    length = '{} hour{} {} minutes'.format(length_hours[0], 's' if int(length_hours[0]) > 1 else '', length_mins)
    
    start_time = raw_start_time.strftime('%H:%M:%S')
    end_time = raw_end_time.strftime('%H:%M:%S')
    
    point_count = len(gpx.tracks[0].segments[0].points)
    
    date = raw_start_time.strftime('%A %-d %B (%d-%m-%Y)')
    
    
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append(tuple([point.latitude, point.longitude]))

    track_data = {
        'points': points,
        'date': date,
        'length': length,
        'start': start_time,
        'end': end_time,
        'point n': point_count}
    
    
    return track_data


# create and format map
def get_static_map():
    # get basemap
    request = cimgt.Stamen(style='terrain')
    
    # make map
    fig, ax = plt.subplots(figsize=(20, 20),
                           subplot_kw=dict(projection=CRS))
    gl = ax.gridlines(draw_labels=True)
    '''gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER'''
    ax.set_extent(EXTENT)
    ax.add_image(request, 14)
    ax.set_title('Capstone Cycling Routes', fontsize=30, pad=20)

    # get scale
    scale_args = dict(linestyle='dashed')
    scale_bar(ax, (0.8, 0.95), 2, plot_kwargs=scale_args)
    
    return fig, ax


# run if folder exists
if os.path.exists(FOLDER):
    files = get_files(FOLDER, 'gpx')
    
    # makes matplotlib map 
    if MAP_TYPE == 'static':
        figure, axes = get_static_map()
        for file in files:
            shp = get_track(file)
            axes.add_geometries(shp, ccrs.PlateCarree(),
                              facecolor='none',
                              edgecolor=COLOURS[files.index(file)],
                              linewidth=2)
            
    # makes folium map
    elif MAP_TYPE == 'dynamic':
        folium_map = folium.Map(location=CENTRE, zoom_start=ZOOM_START, 
                                tiles='Stamen Terrain', control_scale=True)

        for file in files:
            data = get_polyline(file)
            
            points = data['points']
       
            popup_string = ('<b>{}</b><br><br><b>Date:</b> {}<br><b>Start '
                            'time:</b> {}<br><b>End time:</b> {}<br><b>'
                            'Ride length:</b> {}<br><b>Total points:</b> {}'
                .format(os.path.basename(file), data['date'], data['start'], 
                        data['end'], data['length'], data['point n']))
            
            popup_iframe = folium.IFrame(popup_string, width=400, height=150)
            popup = folium.Popup(popup_iframe)
            
            # choose between hover or click for item. click is better for html
            if not HOVER:
                folium.PolyLine(points, color=COLOURS[files.index(file)], 
                                weight=LINE_WEIGHT, opacity=LINE_OPACITY, 
                                popup=popup).add_to(folium_map)
            if HOVER:
                  folium.PolyLine(points, color=COLOURS[files.index(file)], 
                                weight=LINE_WEIGHT, opacity=LINE_OPACITY, 
                                tooltip=folium.Html(popup_string, 
                                    script=True).render()).add_to(folium_map)
            
        folium_map.save('index.html')
    
        webbrowser.open_new_tab('index.html')
        
else:
    print('folder does not exist. please check constants and make sure you have appropriate permissions')

    