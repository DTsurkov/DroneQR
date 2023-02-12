import prettyPrint as pp


class Drone:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.log = pp.Log(f"Drone{self.camera_id}")
        self.points = [int]
        self.log.print("Drone object has been created")

    def __del__(self):
        self.log.print("Drone object has been deleted")
