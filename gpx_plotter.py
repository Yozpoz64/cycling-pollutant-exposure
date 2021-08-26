'''
plots gpx files from input folder
can select either a static (matplotlib) or dynamic (folium) map

NOTES: 
    -get video from autographer photos using command: ffmpeg -framerate 10 -pattern_type glob -i '*.JPG' video.mp4
    -convert this video to .gif using: ffmpeg -i video.mp4 -vf "fps=15,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif

TO DO:
    -create plots and include on tooltips
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
from folium import plugins
from folium.features import DivIcon
import subprocess
from random import randint

# set cwd. having weird issues with conda this should fix
os.chdir('/home/sophie/GitHub/cycling-pollutant-exposure/')
from custom_scalebar import scale_bar

# constants
CENTRE = (-36.88, 174.75)
EXTENT = [-185.363388, -185.103836, -36.996520, -36.819180] # http://bboxfinder.com/
CRS = ccrs.PlateCarree()
FOLDER = 'data/gpxs/'
COLOURS = ['red', 'blue', 'green', 'pink', 'orange']
MAP_TYPE = 'dynamic'
HOVER = True
LINE_WEIGHT = 5
LINE_OPACITY = 0.5
ZOOM_START = 12
TITLE = 'Cycling Routes for Pollutant Exposure Study'
SUBTITLE = 'Created by Sophie Kolston for GISCI 399. Data and code can be found on'
REPO_LINK = 'https://github.com/Yozpoz64/cycling-pollutant-exposure'


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
    times = []
    
    raw_start_time = gpx.tracks[0].segments[0].points[0].time
    raw_end_time = gpx.tracks[0].segments[0].points[-1].time
    
    length_hours = str(round((raw_end_time - raw_start_time)
                             .seconds / 60 / 60, 2)).split('.')
    length_mins = str((float(length_hours[1]) * 0.01) * 60).split('.')[0]
    length = '{} hour{} {} minutes'.format(length_hours[0], 's' if int(length_hours[0]) > 1 else '', length_mins)
    
    point_count = len(gpx.tracks[0].segments[0].points)
    
    date = raw_start_time.strftime('%A %-d %B (%d-%m-%Y)')
    
    
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append(tuple([point.latitude, point.longitude]))
                times.append(point.time)

    track_data = {
        'points': points,
        'times': times,
        'date': date,
        'length': length,
        'point n': point_count}
    
    
    return track_data


# gets pollution data (currently empty as I dont have a sensor)
def get_pollution_data(timestamps, file_name):
    no_pollution_data = True
    pollution_folder = 'data/pollution/'
    gpx_name = os.path.splitext(file_name)[0].split('/')[-1] + '.png'
    plot_name = pollution_folder + gpx_name
    
    if not (os.path.exists(plot_name)):

        # creates fake empty plot with message reading no input data
        if no_pollution_data:
            times = []
            fake_data = []
            
            # create fake data and format time
            for i in range(len(timestamps)):
                fake_data.append(0)
                times.append(timestamps[i].replace(tzinfo=None))
                
    
            # create plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(x=times, y=fake_data, marker='')
            
            fig.canvas.draw()
            
            formatted_labels = []
            for label in ax.get_xticklabels():
                new_label = (str(label).split(',')[2].replace(')', '')
                             .replace("'", '').split(' ')[2])
                formatted_labels.append(new_label)
                
            ax.set_xticklabels(formatted_labels)
            
            fig.text(0.1, 0.5, 'No pollution data found', fontsize=50)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # save plot
            plt.savefig(plot_name)
           
    plot_html = '<img src="{}" width="300"/>'.format(plot_name)
    return plot_html
   

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


# checks for autographer gif
def get_autograph(file_location):
    name = os.path.basename(file_location).split('.')[0]
    
    image_path = 'data/autographer/{}.gif'.format(name)
    
    if os.path.isfile(image_path):
        return '<img src="{}"/>'.format(image_path)

    else:
        # check for folder of photos
        autograph_folder = 'data/autographer/autograph_{}'.format(name)
        if os.path.isdir(autograph_folder):
            # get video, gif
            run_ffmpeg(autograph_folder, name)
            return '<img src="{}"/>'.format(image_path)
        else:
            return 'No Autograph images found.'
    

# calls ffmpeg with arguments
def run_ffmpeg(pic_folder, file_name, video_fps=10, gif_fps=15, gif_scale=320):
    # it does not really have to have both a video and gif, I just want to see both
    
    # create video
    vid_file = '{}/{}.mp4'.format(pic_folder, file_name)
    vid_command = ['ffmpeg', '-framerate', str(video_fps), '-pattern_type', 'glob',
                   '-i', '{}/*.JPG'.format(pic_folder), vid_file]

    subprocess.Popen(vid_command).wait()

    # create gif
    gif_file = 'data/autographer/{}.gif'.format(file_name)
    gif_command = ('ffmpeg -i {} -vf "fps={},scale={}:-1:flags=lanczos,'
                   'split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" '
                   '-loop 0 {}'.format(vid_file, gif_fps, gif_scale, gif_file))
    subprocess.Popen(gif_command, shell=True).wait()


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
                                tiles='Stamen Terrain', control_scale=True,
                                height='85%')
        
        # create title and add to map
        title = ('<h3 align="center" style="font-size:20px"><b>{}</b></h3>'
                 '<h2 align="center" style="font-size:12px">{} <a href="{}">'
                 'GitHub</a></h2>'
                    ).format(TITLE, SUBTITLE, REPO_LINK)
        folium_map.get_root().html.add_child(folium.Element(title))
    
        # add full screen button
        folium.plugins.Fullscreen(position='topleft').add_to(folium_map)
        
        # add mouse position
        folium.plugins.MousePosition(position='topright').add_to(folium_map)
        
        # add measure tool
        folium.plugins.MeasureControl(primary_length_unit='meters',
            secondary_length_unit='kilometers', primary_area_unit='sqmeters',
            secondary_area_unit ='sqkilometers').add_to(folium_map)
        
    
        for file in files:
            # get polyline and information 
            data = get_polyline(file)
            
            # plot pollution data
            plot = get_pollution_data(data['times'], file)
            print(plot)
            
            # get start and end times
            start_time = data['times'][0].strftime('%H:%M:%S')
            end_time = data['times'][-1].strftime('%H:%M:%S')
            
            # get raw coords
            points = data['points']
            
            # get autograph gif if applicable
            image = get_autograph(file)
            
            # html string for popup
            popup_string = ('<b>{}</b><br><br><b>Date:</b> {}<br><b>Start '
                            'time:</b> {}<br><b>End time:</b> {}<br><b>'
                            'Ride length:</b> {}<br><b>Total points:</b> {}'
                            '<br>{}<br><br>{}'
                .format(os.path.basename(file), data['date'], start_time, 
                        end_time, data['length'], data['point n'], image,
                        plot))
            
            popup_iframe = folium.IFrame(popup_string, width=400, height=150)
            popup = folium.Popup(popup_iframe)
            
            # choose between hover or click for item. click is better for html
            if not HOVER:

                folium.PolyLine(points, color=COLOURS[files.index(file)], 
                                weight=LINE_WEIGHT, opacity=LINE_OPACITY, 
                                popup=popup).add_to(folium_map)
        

            if HOVER:
                '''
                  folium.PolyLine(points, color=COLOURS[files.index(file)], 
                                weight=LINE_WEIGHT, opacity=LINE_OPACITY, 
                                tooltip=folium.Html(popup_string, 
                                    script=True).render()).add_to(folium_map)
                  '''
                path = folium.plugins.AntPath(points, delay=3000, weight=LINE_WEIGHT,
                    dashArray=(10, 200),
                    color=COLOURS[files.index(file)], opacity=LINE_OPACITY,
                    tooltip=folium.Html(popup_string, script=True).render()
                    ).add_to(folium_map)
                
                path.options.update(dashArray=[1, 12],
                                    hardwareAcceleration=True,
                                    pulseColor='#3f4145')
            
        folium_map.save('index.html')
    
        webbrowser.open_new_tab('index.html')
        
else:
    print('folder does not exist. please check constants and make sure you have appropriate permissions')

    