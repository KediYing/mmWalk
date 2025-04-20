import carla
# use cmd command to determine the weather while capturing trajectories.
# IMPORTANT:THIS IS A SCRIPT IN CARLA SIMULATOR, ONLY USE IN CARLA ENVIRONMENT!
def set_weather(world, weather_type='sunny'):
    
    # Weather Setting
    # world: carla.World
    # weather_type: string []'sunny', 'rainy', 'foggy', 'cloudy', 'night']
    weather_functions = {
        'sunny': set_sunny,
        'rainy': set_rainy,
        'foggy': set_foggy,
        'cloudy': set_cloudy,
        'night': set_night
    }
    
    if weather_type in weather_functions:
        weather_functions[weather_type](world)
    else:
        print(f"Unknown Type, use default set 'sunny'")
        set_sunny(world)


def set_sunny(world):
    weather = carla.WeatherParameters(
        cloudiness=0,
        precipitation=0,
        precipitation_deposits=0,
        wind_intensity=30,
        wetness=0,
        sun_altitude_angle=75.0
    )
    world.set_weather(weather)
    print("Weather is set to sunny!")


def set_rainy(world):
    weather = carla.WeatherParameters(
        cloudiness=80.0,
        precipitation=60.0,
        precipitation_deposits=40.0,
        wind_intensity=40.0,
        wetness=60.0,
        sun_altitude_angle=45.0,
        fog_density=0.0,
        fog_distance=0.0
    )
    world.set_weather(weather)
    print("Weather is set to rainy!")


def set_foggy(world):
    weather = carla.WeatherParameters(
        cloudiness=50.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=30.0,
        sun_altitude_angle=60.0,
        fog_density=65.0,  
        fog_distance=10.0, 
        fog_falloff=1.0  
    )
    world.set_weather(weather)
    print("Weather is set to foggy!")


def set_cloudy(world):
    weather = carla.WeatherParameters(
        cloudiness=80.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=50.0,
        sun_altitude_angle=65.0,
        fog_density=0.0
    )
    world.set_weather(weather)
    print("Weather is set to cloudy!")


def set_night(world):
    weather = carla.WeatherParameters(
        cloudiness=20.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=15.0,
        sun_altitude_angle=-30.0,
        fog_density=0.0,
        sun_azimuth_angle=270.0
    )
    world.set_weather(weather)
    print("Weather is set to night, all lights are automatically triggered!")