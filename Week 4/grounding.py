# import context
# import rlang
# from rlang.grounding import MDPObject


# class TaxiClass(MDPObject):
#     attr_list = ['name', 'touch_n', 'touch_s', 'touch_e', 'touch_w', 'on_destination', 'on_passenger', 'location']

#     def __init__(self, name, touch_n=False, touch_s=False, touch_e=False, touch_w=False, on_destination=False, on_passenger=False, location=None):
#         self.name = name
#         self.touch_n = touch_n
#         self.touch_s = touch_s
#         self.touch_e = touch_e
#         self.touch_w = touch_w
#         self.on_destination = on_destination
#         self.on_passenger = on_passenger
#         self.location = location


# class PassengerClass(MDPObject):
#     attr_list = ['name', 'location', 'in_taxi']

#     def __init__(self, name, location=None, in_taxi=False):
#         self.name = name
#         self.location = location
#         self.in_taxi = in_taxi


from rlang.grounding import Feature

# Define feature functions
def _passenger_x(state):
    passenger_locs = [[0, 0], [0, 4], [4, 0], [4, 3]]
    passenger_location = int(state[2])  # Index for passenger location

    return passenger_locs[passenger_location][0]  # Get X coordinate

def _passenger_y(state):
    passenger_locs = [[0, 0], [0, 4], [4, 0], [4, 3]]
    passenger_location = int(state[2])
    return passenger_locs[passenger_location][1]  # Get Y coordinate

def _destination_x(state):
    destination_locs = [[0, 0], [0, 4], [4, 0], [4, 3]]
    destination = int(state[3])  # Index for destination
    return destination_locs[destination][0]  # Get X coordinate

def _destination_y(state):
    destination_locs = [[0, 0], [0, 4], [4, 0], [4, 3]]
    destination = int(state[3])
    return destination_locs[destination][1]  # Get Y coordinate

# Register these functions as Features
passenger_x = Feature(_passenger_x)
passenger_y = Feature(_passenger_y)
destination_x = Feature(_destination_x)
destination_y = Feature(_destination_y)


