"""World class - for storing and updating the state of the world."""

class World():

    def __init__(self, realtime_tick, simtime_tick, delivery_path, events_path):
        self.realtime_tick = realtime_tick
        self.simtime_tick = simtime_tick
        self.delivery_path = delivery_path
        self.events_path = events_path
    
