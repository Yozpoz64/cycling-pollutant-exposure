'''
GPX PLOTTER

Description
plots gpx files, autographer pictures, pollution data from input folder
can select either a static (matplotlib) or dynamic (folium) map

Info
Created by Sophie Kolston
MIT License
'''
# need quite a few for static cartography, as well as local secondary methods
import matplotlib.pyplot as plt
import fiona
import glob
from shapely.geometry import shape
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import os

# folium
import webbrowser # for showing folium maps easily
import folium
import gpxpy
from folium import plugins
from folium.features import DivIcon
import subprocess
import math
import haversine as hs
from haversine import Unit
import random

# set cwd. having weird issues with conda this should fix
os.chdir('/home/sophie/GitHub/cycling-pollutant-exposure/')
from custom_scalebar import scale_bar

# constants
CENTRE = (-36.88, 174.75)
EXTENT = [-185.36, -185.10, -36.99, -36.82] # http://bboxfinder.com/
CRS = ccrs.PlateCarree()
FOLDER = 'data/gpxs/'
LINE_WEIGHT = 5
LINE_OPACITY = 0.5
ZOOM_START = 12
TITLE = 'Cycling Routes for Pollutant Exposure Study'
SUBTITLE = 'Created by Sophie Kolston for GISCI 399. Data and code can be found on'
REPO_LINK = 'https://github.com/Yozpoz64/cycling-pollutant-exposure'

IMAGE_PATH = 'data/autographer/'
POLLUTION_PATH = 'data/pollution/'

INDEX_NAME = 'index.html'


class StaticMap():
    
    def __init__(self, data, automate, fig_size=(20, 20), projection=CRS, 
                 title='Lockdown Cycling Routes'):
        # get and set local variables
        self.data = data
        self.size = fig_size
        self.crs = projection
        
        if automate:
            # create plot (map)
            self.fig, self.ax = self.draw_map()
            
            # get tracks
            self.iterrate_data()
            
            
    # gets random colours for the tracks on the map
    def get_random_colours(self, num_colours):
        return ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(num_colours)]

    
    # iterates over data, adding geometries to map
    def iterrate_data(self):
        self.colours = self.get_random_colours(len(self.data))
        for file in self.data:
            shape = self.get_track(file)
            self.ax.add_geometries(shape, ccrs.PlateCarree(),
                                   facecolor='none', 
                                   edgecolor=self.colours[self.data.index(file)],
                                   linewidth=2)

        
    # extracts trackpoints from a gpx file (the easy way)
    def get_track(self, file_location):
        file = fiona.open(file_location, layer='tracks')
        points = {'type': 'MultiLineString', 
                  'coordinates': file[0]['geometry']['coordinates']}
        gpx_shp = shape(points)
        return gpx_shp
        
    
    # create and format map
    def draw_map(self):
        # get basemap
        request = cimgt.Stamen(style='terrain')
        
        # make map
        fig, ax = plt.subplots(figsize=self.size,
                               subplot_kw=dict(projection=self.crs))
        ax.gridlines(draw_labels=True)

        ax.set_extent(EXTENT)
        ax.add_image(request, 14)
        ax.set_title('Lockdown Cycling Routes', fontsize=30, pad=20)
    
        # get scale
        scale_args = dict(linestyle='dashed')
        scale_bar(ax, (0.8, 0.95), 2, plot_kwargs=scale_args)
        
        return fig, ax
    
    

class WebMap():
    
    def __init__(self, data, automate, zoom_start=ZOOM_START, centre=CENTRE, 
                 map_tiles='Stamen Terrain', title=TITLE, subtitle=SUBTITLE,
                 repo_link=REPO_LINK, ffmpeg_loc='ffmpeg',
                 image_path=IMAGE_PATH, video_fps=10, gif_fps=15,
                 gif_scale=320, pollution_path=POLLUTION_PATH):
        
        # define map variables
        self.zoom = zoom_start
        self.centre = centre
        self.tiles = map_tiles
        self.title = title
        self.subtitle = subtitle
        self.repo = repo_link
        
        # define data variables
        self.data = data
        
        # define external tool variables
        self.ffmpeg = ffmpeg_loc # location of runtime. if symlink just 'ffmpeg' else eg. /bin/ffmpeg.sh
        self.video_fps = video_fps
        self.gif_fps = gif_fps
        self.gif_scale = gif_scale
        
        # define file locations
        self.image_path = image_path
        self.pollution_path = pollution_path
        
        
        if automate:
            # build map
            self.simple_map = self.draw_map()
            self.advanced_map = self.draw_map()
            
            # itterate through data and map. calls other functions
            self.map_data()
            
            
    # get distance ridden 
    def calc_distance(self, points):
        all_distances = []
        for i in range(len(points)):
            if i != len(points) - 1:
                dist = hs.haversine(points[i], points[i + 1], unit=Unit.METERS)
                all_distances.append(dist)
                
        return round((sum(all_distances) / 1000), 2)
    
    
    # gets random colours for the tracks on the map
    def get_random_colours(self, num_colours):
        return ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(num_colours)]
        
    
    # organizes all of the metadata required for mapping
    def map_data(self):
        
        self.total_distance = 0
        
        self.colours = self.get_random_colours(len(self.data))
        
        for file in self.data:
            # get polyline and information 
            track_data = self.get_polyline(file)
            
            # get raw coords
            points = track_data['points']
            
            # add distance to total
            self.total_distance += track_data['distance']
            
            # get autograph gif if applicable
            image = self.get_autograph(file)

            # plot pollution data
            plot = self.get_pollution_data(track_data['times'], file)
            
            # html string for popup
            popup_string = ('<b>{}</b><br><br><b>Date:</b> {}<br><b>'
                            'Ride length:</b> {}<br><b>Ride distance:</b> {}km'
                            '<br><b>Average speed:</b> {}km/h<br><b>Total GPS '
                            'points:</b> {}<br>{}<br><br>{}'
                .format(os.path.basename(file), track_data['date'], 
                        track_data['length'], track_data['distance'], 
                        track_data['speed'], track_data['point n'], image, plot))
            
            '''
            popup_iframe = folium.IFrame(popup_string, width=400, height=200)
            popup = folium.Popup(popup_iframe)
            '''
            
            # add polyline to simple map
            folium.PolyLine(points, color=self.colours[files.index(file)], 
                            weight=LINE_WEIGHT, opacity=LINE_OPACITY, 
                            popup=folium.Html(popup_string, script=True)
                            .render()).add_to(self.simple_map)
        
            # set antpath delay to be proportional to the rider speed
            antpath_speed = 4000 - (track_data['speed'] * 100)
            # add antpath to advanced map
            path = folium.plugins.AntPath(points, delay=antpath_speed, weight=LINE_WEIGHT,
                dashArray=(10, 200),
                color=self.colours[files.index(file)], opacity=LINE_OPACITY,
                tooltip=folium.Html(popup_string, script=True).render()
                ).add_to(self.advanced_map)
                
            path.options.update(dashArray=[1, 12],
                                hardwareAcceleration=True,
                                pulseColor='#3f4145')
            
        self.total_distance = round(self.total_distance, 2)
            
            
     # extracts trackpoints from a gpx file (the hard way, for folium)
    def get_polyline(self, file):
        this_file = open(file, 'r')
        gpx = gpxpy.parse(this_file)
        points = []
        times = []
        
        raw_start_time = gpx.tracks[0].segments[0].points[0].time
        raw_end_time = gpx.tracks[0].segments[0].points[-1].time
        
        length_hours = str(round((raw_end_time - raw_start_time)
                                 .seconds / 60 / 60, 2))
        length_mins = str((float(length_hours.split('.')[1]) * 0.01) 
                          * 60).split('.')[0]
        length = '{} hour{} {} minutes'.format(length_hours[0], 's' if int(length_hours[0]) > 1 else '', length_mins)
        
        point_count = len(gpx.tracks[0].segments[0].points)
        
        date = raw_start_time.strftime('%A %-d %B (%d-%m-%Y)')

        
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    points.append(tuple([point.latitude, point.longitude]))
                    times.append(point.time)
    
        distance = self.calc_distance(points)
      
        speed = round(distance / float(length_hours), 2)
        
        track_data = {
            'points': points,
            'times': times,
            'date': date,
            'length': length,
            'point n': point_count,
            'distance': distance,
            'speed': speed}
        
        return track_data


    def draw_map(self):
        folium_map = folium.Map(location=self.centre, zoom_start=self.zoom, 
                                tiles=self.tiles, control_scale=True,
                                height='85%')
       
        # create title and add to map
        title = ('<h3 align="center" style="font-size:20px"><b>{}</b></h3>'
                 '<h2 align="center" style="font-size:12px">{} <a href="{}">'
                 'GitHub</a></h2>'
                    ).format(self.title, self.subtitle, self.repo)
        folium_map.get_root().html.add_child(folium.Element(title))
 
        # add full screen button
        folium.plugins.Fullscreen(position='topleft').add_to(folium_map)
        
        # add mouse position
        folium.plugins.MousePosition(position='topright').add_to(folium_map)
        
        # add measure tool
        folium.plugins.MeasureControl(primary_length_unit='meters',
            secondary_length_unit='kilometers', primary_area_unit='sqmeters',
            secondary_area_unit ='sqkilometers').add_to(folium_map)
        
        return folium_map
        
    
    # gets pollution data (currently empty as I dont have a sensor)
    def get_pollution_data(self, timestamps, file_name):
        no_pollution_data = True
        
        gpx_name = os.path.splitext(file_name)[0].split('/')[-1] + '.png'
        plot_name = self.pollution_path + gpx_name
        
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
       
    
    # checks for autographer gif
    def get_autograph(self, file_location):
        name = os.path.basename(file_location).split('.')[0]
        
        image_path = '{}{}.gif'.format(self.image_path, name)
        
        if os.path.isfile(image_path):
            return '<img src="{}"/>'.format(image_path)
    
        else:
            # check for folder of photos
            autograph_folder = '{}autograph_{}'.format(self.image_path, name)
            if os.path.isdir(autograph_folder):
                # get video, gif
                self.run_ffmpeg(autograph_folder, name)
                return '<img src="{}"/>'.format(image_path)
            else:
                return 'No Autograph images found.'
            
            
    # calls ffmpeg with arguments
    def run_ffmpeg(self, pic_folder, file_name):
        # it does not really have to have both a video and gif, I just want to see both
        
        # create video
        vid_file = '{}/{}.mp4'.format(pic_folder, file_name)
        vid_command = ['ffmpeg', '-framerate', str(self.video_fps), '-pattern_type', 'glob',
                       '-i', '{}/*.JPG'.format(pic_folder), vid_file]
    
        subprocess.Popen(vid_command).wait()
    
        # create gif
        gif_file = 'data/autographer/{}.gif'.format(file_name)
        gif_command = ('ffmpeg -i {} -vf "fps={},scale={}:-1:flags=lanczos,'
                       'split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" '
                       '-loop 0 {}'.format(vid_file, self.gif_fps, 
                                           self.gif_scale, gif_file))
        subprocess.Popen(gif_command, shell=True).wait()
        
        
    # saves map
    def save_map(self, index='advanced'):
        # choose name based on index. has to be index.html for github pages
        self.index = index
        if self.index == 'advanced':
            self.advanced_filename = 'index.html'
            self.simple_filename = 'webmap_simple.html'
        else:
            self.advanced_filename = 'webmap_advanced.html'
            self.simple_filename = 'index.html'
        
        # save simple map
        self.simple_map.save(self.simple_filename)
        file = open(self.simple_filename, 'a')
        button_html = """<center style="margin-top: 0.5cm;">
            Total cycling distance: {}km<br><br>
            <button onclick="window.location.href='{}'
            ">Go to full site (more CPU intensive)</button>
            </center>""".format(self.total_distance, self.advanced_filename)
        file.write(button_html)
        
        # save advanced map
        self.advanced_map.save(self.advanced_filename)
        file = open(self.advanced_filename, 'a')
        button_html = """<center style="margin-top: 0.5cm;">
                Total cycling distance: {}km<br><br>
                <button onclick="window.location.href='{}'
                ">Go to basic site (less CPU intensive)</button>
                </center>""".format(self.total_distance, self.simple_filename)
        file.write(button_html)
            
        
    # opens map in browser
    def open_map(self):
        webbrowser.open('index.html')
        


# get files with extension in a folder
def get_files(folder, extension):
    glob_arg = folder + '*.' + extension
    return glob.glob(glob_arg)


# run if folder exists
if os.path.exists(FOLDER):
    files = get_files(FOLDER, 'gpx')
    
    #static_map = StaticMap(files, automate=True)
   
    dynamic_map = WebMap(files, automate=True)
    dynamic_map.save_map(index='simple')
    
    dynamic_map.open_map()
            
      
else:
    print('folder does not exist. please check constants and make sure you have appropriate permissions')

    