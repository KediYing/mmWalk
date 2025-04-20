import carla
import time
# Change the current Map for data capturing, choose a map you want!
# IMPORTANT:THIS IS A SCRIPT IN CARLA SIMULATOR, ONLY USE IN CARLA ENVIRONMENT!
def switchMap():
    # CONNECTING...
    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)
    
    # Load Map
    world = client.load_world('A_MAP')
    #[Town01,Town02,........], see Carla Simulator Documentation for more information
    
    # Set spectator
    spectator = world.get_spectator()
    spawn_points = world.get_map().get_spawn_points()
    
    initial_location = spawn_points[0].location
    spectator.set_transform(carla.Transform(
        initial_location + carla.Location(z=50),
        carla.Rotation(pitch=-90)
    ))

if __name__ == '__main__':
    switchMap()