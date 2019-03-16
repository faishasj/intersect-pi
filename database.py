import pyrebase
from config import config

class Database:

    def __init__(self):
        self.data = pyrebase.initialize_app(config).database()

    def add_car(self,road,lane):
        val = self.data.child("traffic").child("street_" + road).child(lane).get().val() + 1
        self.set_cars(road,lane,val)

    def set_cars(self, road, lane, val):
        self.data.child("traffic").child("street_" + road).update({lane:val})

    def reset_values(self):
        for road in ["n","e","s","w"]:
            for lane in range(3):
                self.data.child("traffic").child("street_" + road).update({lane: 0})
