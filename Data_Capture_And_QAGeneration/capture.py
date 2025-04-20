import carla
import pygame
import numpy as np
import math
import queue
import os
import argparse
from datetime import datetime
import cv2
import carla_weather
import py360convert   # py360convert, an important library for converting cubemap to panorama

def parse_args():
    parser = argparse.ArgumentParser(description='CARLA Walker Control with Equirectangular Data Collection')
    parser.add_argument('-w', '--weather', type=str, default='sunny',
                        choices=['sunny', 'rainy', 'foggy', 'cloudy', 'night'],
                        help='Set Weather (sunny, rainy, foggy, cloudy, night)')
    parser.add_argument('-rainy', action='store_const', const='rainy', dest='weather',
                        help='Set Rainy')
    parser.add_argument('-foggy', action='store_const', const='foggy', dest='weather',
                        help='Set Foggy')
    parser.add_argument('-cloudy', action='store_const', const='cloudy', dest='weather',
                        help='Set Cloudy')
    parser.add_argument('-night', action='store_const', const='night', dest='weather',
                        help='Set Night')
    return parser.parse_args()

class WalkerControlWithData:
    def __init__(self, weather_type='sunny'):
        print("Initializing pygame...")
        pygame.init()
        self.display = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Manual Walker Control with Equirectangular Data Collection")
        print("Pygame initialized successfully")
        
        # Preset size for panorama
        self.eq_width = 1600
        self.eq_height = 900
        
        # Connecting Carla
        print("Connecting to CARLA...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(100.0)
        self.world = self.client.get_world()
        
        # Obtain the current position of SPECTATOR(Manual Control in Carla Window)
        spectator = self.world.get_spectator()
        spawn_transform = spectator.get_transform()
        print(f"Spawning walker at: Location(x={spawn_transform.location.x:.2f}, y={spawn_transform.location.y:.2f}, z={spawn_transform.location.z:.2f})")
        print(f"With rotation: Rotation(pitch={spawn_transform.rotation.pitch:.2f}, yaw={spawn_transform.rotation.yaw:.2f}, roll={spawn_transform.rotation.roll:.2f})")
        
        # Synchronized
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        # set weather
        carla_weather.set_weather(self.world, weather_type)
        
        # Output Directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join('YOUR_PATH', f'walker_data_{weather_type}_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.equirect_folders = {
            # Walker
            'rgb_walker': os.path.join(self.output_dir, 'walker/rgb'),
            'depth_walker': os.path.join(self.output_dir, 'walker/depth'),
            'semantic_walker': os.path.join(self.output_dir, 'walker/semantic'),
            
            # Dog
            'rgb_dog': os.path.join(self.output_dir, 'dog/rgb'),
            'depth_dog': os.path.join(self.output_dir, 'dog/depth'),
            'semantic_dog': os.path.join(self.output_dir, 'dog/semantic'),
            
            # Drone
            'rgb_drone': os.path.join(self.output_dir, 'drone/rgb'),
            'depth_drone': os.path.join(self.output_dir, 'drone/depth'),
            'semantic_drone': os.path.join(self.output_dir, 'drone/semantic'),
            
            # IMU
            'imu': os.path.join(self.output_dir, 'imu'),
            
            # Action
            'action': os.path.join(self.output_dir, 'action')
        }
        
        for folder in self.equirect_folders.values():
            os.makedirs(folder, exist_ok=True)

        # Create NPC Vehicles around
        self.npc_vehicles = []

        try:
            spawn_points = self.world.get_map().get_spawn_points()
            
            spectator = self.world.get_spectator()
            center_location = spectator.get_transform().location
            
            valid_spawn_points = [p for p in spawn_points 
                                if p.location.distance(center_location) < 100.0]
            np.random.shuffle(valid_spawn_points)
            
            vehicle_bps = list(self.world.get_blueprint_library().filter('vehicle.*'))
            for i in range(min(10, len(valid_spawn_points))):
                try:
                    # Random Choose Vehicle
                    bp = np.random.choice(vehicle_bps)
                    if bp.has_attribute('number_of_wheels') and int(bp.get_attribute('number_of_wheels')) <= 2:
                        continue
                    npc = self.world.try_spawn_actor(bp, valid_spawn_points[i])
                    if npc is not None:
                        npc.set_autopilot(True)
                        self.npc_vehicles.append(npc)
                        print(f"NPC Vehicle Spawning: {bp.id} at location ({valid_spawn_points[i].location.x:.1f}, {valid_spawn_points[i].location.y:.1f})")
                except:
                    continue
                    
            print(f"Successfully Spawning {len(self.npc_vehicles)} Vehicles")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            # CLEANUP
            for npc in self.npc_vehicles:
                if npc is not None and npc.is_alive:
                    npc.destroy()

        walker_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')[0]
        spectator = self.world.get_spectator()
        spawn_transform = spectator.get_transform()
        self.current_yaw = spawn_transform.rotation.yaw
        self.walker = self.world.spawn_actor(walker_bp, spawn_transform)
        self.world.tick()
        
        self.view_types = ['walker', 'dog', 'drone']
        self.sensor_types = ['rgb', 'depth', 'semantic']
        
        # set directions
        self.cubemap_directions = {
            'walker': ['front', 'right', 'back', 'left'],
            'dog': ['front', 'right', 'back', 'left'],
            'drone': ['front', 'right', 'back', 'left', 'up', 'down']
        }
        
        # set image cache
        self.image_cache = {}
        for view_type in self.view_types:
            for sensor_type in self.sensor_types:
                self.image_cache[(sensor_type, view_type)] = {}
                for direction in self.cubemap_directions[view_type]:
                    self.image_cache[(sensor_type, view_type)][direction] = None
        
        # set sensors
        self.setup_sensors()
        
        # PYGAME INITIATE(NEW WINDOW TO CAPTURE TRAJECTORY MANUALLY)
        self.clock = pygame.time.Clock()
        self.surface = None
        
        self.control = carla.WalkerControl()
        #self.control.speed = 0.0
        #self.control.jump = False
        #self.frame_number = 0
        
        # Capture Current Keyboard Input for Action Detection
        self.current_action = {'W': False, 'A': False, 'S': False, 'D': False}
        
    def setup_sensors(self):
        self.sensors = []
        self.sensor_queues = {}
        
        # Def Cameras
        rgb_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', '400')
        rgb_bp.set_attribute('image_size_y', '400')
        rgb_bp.set_attribute('fov', '90')
        
        depth_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '400')
        depth_bp.set_attribute('image_size_y', '400')
        depth_bp.set_attribute('fov', '90')
        
        semantic_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        semantic_bp.set_attribute('image_size_x', '400')
        semantic_bp.set_attribute('image_size_y', '400')
        semantic_bp.set_attribute('fov', '90')

        # Set Camera
        display_camera_transform = carla.Transform(carla.Location(x=0.5, z=1.5))
        self.display_camera = self.world.spawn_actor(rgb_bp, display_camera_transform, attach_to=self.walker)
        self.display_camera.listen(lambda image: self._process_display_image(image))
        self.sensors.append(self.display_camera)

        # Def Height
        view_heights = {
            'walker': 1.5, 
            'dog': 0.1,    
            'drone': 3.0   
        }

        # Dog Position
        yaw_rad = math.radians(self.current_yaw)
        dog_offset_x = 0.8 * math.cos(yaw_rad)
        dog_offset_y = 0.8 * math.sin(yaw_rad)
        dog_location = carla.Location(x=dog_offset_x, y=dog_offset_y, z=view_heights['dog'])
        
        # Set Cubemap
        camera_rotations = {
            'front': carla.Rotation(pitch=0, yaw=0, roll=0),
            'right': carla.Rotation(pitch=0, yaw=90, roll=0),
            'back': carla.Rotation(pitch=0, yaw=180, roll=0),
            'left': carla.Rotation(pitch=0, yaw=270, roll=0),
            'up': carla.Rotation(pitch=90, yaw=0, roll=0),
            'down': carla.Rotation(pitch=-90, yaw=0, roll=0)
        }
        
        # Walker
        for direction, rotation in camera_rotations.items():
            if direction in ['up', 'down'] and direction not in self.cubemap_directions['walker']:
                continue
                
            # RGB camera
            sensor_name = f'rgb_walker_{direction}'
            transform = carla.Transform(carla.Location(z=view_heights['walker']), rotation)
            rgb_sensor = self.world.spawn_actor(rgb_bp, transform, attach_to=self.walker)
            self.sensor_queues[sensor_name] = queue.Queue()
            rgb_sensor.listen(lambda image, name=sensor_name: self.sensor_queues[name].put(image))
            self.sensors.append(rgb_sensor)
            
            # Depth camera
            sensor_name = f'depth_walker_{direction}'
            depth_sensor = self.world.spawn_actor(depth_bp, transform, attach_to=self.walker)
            self.sensor_queues[sensor_name] = queue.Queue()
            depth_sensor.listen(lambda image, name=sensor_name: self.sensor_queues[name].put(image))
            self.sensors.append(depth_sensor)
            
            # Semantic camera
            sensor_name = f'semantic_walker_{direction}'
            semantic_sensor = self.world.spawn_actor(semantic_bp, transform, attach_to=self.walker)
            self.sensor_queues[sensor_name] = queue.Queue()
            semantic_sensor.listen(lambda image, name=sensor_name: self.sensor_queues[name].put(image))
            self.sensors.append(semantic_sensor)
        
        #Dog
        for direction, rotation in camera_rotations.items():
            if direction in ['up', 'down'] and direction not in self.cubemap_directions['dog']:
                continue
                
            # RGB camera
            sensor_name = f'rgb_dog_{direction}'
            transform = carla.Transform(dog_location, rotation)
            rgb_sensor = self.world.spawn_actor(rgb_bp, transform, attach_to=self.walker)
            self.sensor_queues[sensor_name] = queue.Queue()
            rgb_sensor.listen(lambda image, name=sensor_name: self.sensor_queues[name].put(image))
            self.sensors.append(rgb_sensor)
            
            # Depth camera
            sensor_name = f'depth_dog_{direction}'
            depth_sensor = self.world.spawn_actor(depth_bp, transform, attach_to=self.walker)
            self.sensor_queues[sensor_name] = queue.Queue()
            depth_sensor.listen(lambda image, name=sensor_name: self.sensor_queues[name].put(image))
            self.sensors.append(depth_sensor)
            
            # Semantic camera
            sensor_name = f'semantic_dog_{direction}'
            semantic_sensor = self.world.spawn_actor(semantic_bp, transform, attach_to=self.walker)
            self.sensor_queues[sensor_name] = queue.Queue()
            semantic_sensor.listen(lambda image, name=sensor_name: self.sensor_queues[name].put(image))
            self.sensors.append(semantic_sensor)
        
        #Drone
        for direction, rotation in camera_rotations.items():
            if direction not in self.cubemap_directions['drone']:
                continue
                
            # RGB camera
            sensor_name = f'rgb_drone_{direction}'
            transform = carla.Transform(carla.Location(z=view_heights['drone']), rotation)
            rgb_sensor = self.world.spawn_actor(rgb_bp, transform, attach_to=self.walker)
            self.sensor_queues[sensor_name] = queue.Queue()
            rgb_sensor.listen(lambda image, name=sensor_name: self.sensor_queues[name].put(image))
            self.sensors.append(rgb_sensor)
            
            # Depth camera
            sensor_name = f'depth_drone_{direction}'
            depth_sensor = self.world.spawn_actor(depth_bp, transform, attach_to=self.walker)
            self.sensor_queues[sensor_name] = queue.Queue()
            depth_sensor.listen(lambda image, name=sensor_name: self.sensor_queues[name].put(image))
            self.sensors.append(depth_sensor)
            
            # Semantic camera
            sensor_name = f'semantic_drone_{direction}'
            semantic_sensor = self.world.spawn_actor(semantic_bp, transform, attach_to=self.walker)
            self.sensor_queues[sensor_name] = queue.Queue()
            semantic_sensor.listen(lambda image, name=sensor_name: self.sensor_queues[name].put(image))
            self.sensors.append(semantic_sensor)

        # IMU
        imu_bp = self.world.get_blueprint_library().find('sensor.other.imu')
        self.imu_sensor = self.world.spawn_actor(
            imu_bp, 
            carla.Transform(carla.Location(x=0.5, z=1.5)), 
            attach_to=self.walker
        )
        self.sensor_queues['imu'] = queue.Queue()
        self.imu_sensor.listen(lambda data: self.sensor_queues['imu'].put(data))
        self.sensors.append(self.imu_sensor)

    def create_dice_cubemap(self, images, sensor_type, view_type):
        """
        Use Dice Cubemap prepared for py360convert
        +------+------+------+------+
        |      |  U   |      |      |
        +------+------+------+------+
        |  L   |  F   |  R   |  B   |
        +------+------+------+------+
        |      |  D   |      |      |
        +------+------+------+------+
        """
        required_directions = ['front', 'right', 'back', 'left']
        for direction in required_directions:
            if direction not in images or images[direction] is None:
                return None
        
        height, width = images['front'].shape[:2]
        
        if sensor_type == 'rgb':
            dice_cubemap = np.zeros((height * 3, width * 4, 3), dtype=np.uint8)
        elif sensor_type == 'depth':
            dice_cubemap = np.zeros((height * 3, width * 4), dtype=np.uint16)
        else:
            dice_cubemap = np.zeros((height * 3, width * 4), dtype=np.uint8)
        
        if view_type == 'drone' and 'up' in images and 'down' in images:
            up_img = images['up']
            down_img = images['down']
        else:
            if sensor_type == 'rgb':
                up_img = np.zeros((height, width, 3), dtype=np.uint8)
                down_img = np.zeros((height, width, 3), dtype=np.uint8)
            elif sensor_type == 'depth':
                up_img = np.zeros((height, width), dtype=np.uint16)
                down_img = np.zeros((height, width), dtype=np.uint16)
            else: 
                up_img = np.zeros((height, width), dtype=np.uint8)
                down_img = np.zeros((height, width), dtype=np.uint8)

        if sensor_type == 'rgb':
            dice_cubemap[0:height, width:width*2] = up_img
        else: 
            dice_cubemap[0:height, width:width*2] = up_img

        if sensor_type == 'rgb':
            dice_cubemap[height:height*2, 0:width] = images['left']
            dice_cubemap[height:height*2, width:width*2] = images['front']
            dice_cubemap[height:height*2, width*2:width*3] = images['right']
            dice_cubemap[height:height*2, width*3:width*4] = images['back']
        else:
            dice_cubemap[height:height*2, 0:width] = images['left']
            dice_cubemap[height:height*2, width:width*2] = images['front']
            dice_cubemap[height:height*2, width*2:width*3] = images['right']
            dice_cubemap[height:height*2, width*3:width*4] = images['back']

        if sensor_type == 'rgb':
            dice_cubemap[height*2:height*3, width:width*2] = down_img
        else: 
            dice_cubemap[height*2:height*3, width:width*2] = down_img
        
        return dice_cubemap

    def convert_cubemap_to_equirect(self, dice_cubemap, sensor_type):
        """
        Convert image into PANORAMA
        """

        if sensor_type == 'semantic':
            mode = 'nearest'
        else:
            mode = 'bilinear'
        
        h, w = self.eq_height, self.eq_width
        

        if sensor_type == 'rgb':
            equirect = py360convert.c2e(dice_cubemap, h, w, mode=mode, cube_format='dice')
            return equirect
        elif sensor_type == 'depth':
            depth_float = dice_cubemap.astype(np.float32)
            equirect_float = py360convert.c2e(depth_float, h, w, mode=mode, cube_format='dice')
            return equirect_float.astype(np.uint16)
        else: 
            equirect = py360convert.c2e(dice_cubemap, h, w, mode=mode, cube_format='dice')
            return equirect
    # Saving sensor data, see CALRA Documentation for more information
    def save_sensor_data(self):
        action_file = os.path.join(self.equirect_folders['action'], f'{self.frame_number:06d}.txt')
        with open(action_file, 'w') as f:
            f.write(f"W: {int(self.current_action['W'])}\n")
            f.write(f"A: {int(self.current_action['A'])}\n")
            f.write(f"S: {int(self.current_action['S'])}\n")
            f.write(f"D: {int(self.current_action['D'])}\n")
        
        for sensor_name, sensor_queue in self.sensor_queues.items():
            try:
                data = sensor_queue.get(timeout=0.1)
                
                if sensor_name.startswith(('rgb_', 'depth_', 'semantic_')):
                    parts = sensor_name.split('_')
                    sensor_type = parts[0]  # rgb/depth/semantic
                    view_type = parts[1]    # walker/dog/drone
                    direction = parts[2]    # front/right/back/left/up/down
                    
                    if sensor_type == 'rgb':
                        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                        array = np.reshape(array, (data.height, data.width, 4))
                        bgr_array = array[:, :, :3]
                        self.image_cache[(sensor_type, view_type)][direction] = bgr_array
                        
                    elif sensor_type == 'depth':
                        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                        array = np.reshape(array, (data.height, data.width, 4))
                        array = array.astype(np.float32)
                        normalized = (array[:, :, 2] + array[:, :, 1] * 256 + array[:, :, 0] * 256 * 256) / (256 * 256 * 256 - 1)
                        depth_meters = 1000 * normalized
                        depth_array = (depth_meters * 256).astype(np.uint16)
                        self.image_cache[(sensor_type, view_type)][direction] = depth_array
                        
                    elif sensor_type == 'semantic':
                        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                        array = np.reshape(array, (data.height, data.width, 4))
                        semantic_array = array[:, :, 2]
                        self.image_cache[(sensor_type, view_type)][direction] = semantic_array
                        
                elif sensor_name == 'imu':
                    filename = os.path.join(self.equirect_folders['imu'], f'{self.frame_number:06d}.txt')
                    with open(filename, 'w') as f:
                        f.write(f"Accelerometer: ({data.accelerometer.x}, {data.accelerometer.y}, {data.accelerometer.z})\n")
                        f.write(f"Gyroscope: ({data.gyroscope.x}, {data.gyroscope.y}, {data.gyroscope.z})\n")
                        f.write(f"Compass: {data.compass}\n")
                #CLEANUP
                while not sensor_queue.empty():
                    sensor_queue.get_nowait()
                    
            except queue.Empty:
                continue
        

        for view_type in self.view_types:
            for sensor_type in self.sensor_types:
                images = self.image_cache[(sensor_type, view_type)]
                
                dice_cubemap = self.create_dice_cubemap(images, sensor_type, view_type)
                
                if dice_cubemap is not None:
                    equirect_img = self.convert_cubemap_to_equirect(dice_cubemap, sensor_type)
                    
                    equirect_filename = os.path.join(
                        self.equirect_folders[f'{sensor_type}_{view_type}'], 
                        f'{self.frame_number:06d}.png'
                    )
                    
                    if sensor_type == 'depth':
                        cv2.imwrite(equirect_filename, equirect_img)
                    else:
                        cv2.imwrite(equirect_filename, equirect_img)

    def _process_display_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def run(self):
        try:
            while True:
                self.world.tick()
                
                if self.surface is not None:
                    self.display.blit(self.surface, (0, 0))
                    font = pygame.font.Font(None, 36)
                    text = font.render(f'Frame: {self.frame_number} Yaw: {self.current_yaw:.1f}', True, (255, 255, 255))
                    self.display.blit(text, (10, 10))
                    eq_text = font.render(f'Equirect: {self.eq_width}x{self.eq_height}', True, (255, 255, 0))
                    self.display.blit(eq_text, (10, 50))
                    
                    pygame.display.flip()

                # Keyboard input
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_ESCAPE:
                            return
                
                # Keyboard listener
                keys = pygame.key.get_pressed()
                
                direction = carla.Vector3D(0.0, 0.0, 0.0)
                
                self.current_action['W'] = bool(keys[pygame.K_w])
                self.current_action['A'] = bool(keys[pygame.K_a])
                self.current_action['S'] = bool(keys[pygame.K_s])
                self.current_action['D'] = bool(keys[pygame.K_d])
                
                if self.current_action['W']:  
                    direction.x = math.cos(math.radians(self.current_yaw))
                    direction.y = math.sin(math.radians(self.current_yaw))
                elif self.current_action['S']:  
                    direction.x = -math.cos(math.radians(self.current_yaw))
                    direction.y = -math.sin(math.radians(self.current_yaw))
                
                if self.current_action['A']:  
                    self.current_yaw = (self.current_yaw - 2) % 360  
                elif self.current_action['D']:  
                    self.current_yaw = (self.current_yaw + 2) % 360  
                
                if keys[pygame.K_SPACE]:
                    self.control.jump = True
                else:
                    self.control.jump = False
                
                if abs(direction.x) > 0.0 or abs(direction.y) > 0.0:
                    self.control.speed = 3.0
                    self.control.direction = direction
                else:
                    self.control.speed = 0.0
                
                self.walker.apply_control(self.control)
                
                walker_transform = self.walker.get_transform()
                walker_transform.rotation.yaw = self.current_yaw
                self.walker.set_transform(walker_transform)
                
                self.save_sensor_data()
                self.frame_number += 1
                
                self.clock.tick(60)
        
        finally:
            pygame.quit()
            #CLEANUP
            print("\nCleaning up actors...")
            
            for npc in self.npc_vehicles:
                if npc is not None and npc.is_alive:
                    npc.destroy()
            
            for sensor in self.sensors:
                if sensor is not None and sensor.is_alive:
                    sensor.destroy()
            if self.walker is not None and self.walker.is_alive:
                self.walker.destroy()
            
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            print("Cleanup completed.")
if __name__ == '__main__':
    try:
        args = parse_args()
        print(f"Starting CARLA Walker Control with Equirectangular Data Collection (Weather: {args.weather})...")
        print(f"Equirectangular resolution: 1600x900 (固定)")
        
        control = WalkerControlWithData(weather_type=args.weather)
        
        print("Initialization completed. Starting main loop...")
        control.run()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()