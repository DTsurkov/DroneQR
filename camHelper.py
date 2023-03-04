import cv2
import subprocess
import re


def get_VendorIDs():
    cam_names = subprocess.run(["system_profiler", "SPCameraDataType"], stdout=subprocess.PIPE, text=True)
    pattern = r'(?:Model|Unique) ID: (.*)'
    VendorIDs = re.findall(pattern, cam_names.stdout)
    return VendorIDs


def get_indexes(stop_after=2):
    empty_indexes = []
    camera_indexes = []
    index = 0
    while len(empty_indexes) < stop_after:
        camera = cv2.VideoCapture(index)
        if camera.isOpened():
            is_reading, img = camera.read()
            # w = camera.get(3)
            # h = camera.get(4)
            if is_reading:
                camera_indexes.append(index)
        else:
            empty_indexes.append(index)
        index += 1

    return camera_indexes


def get_cameras(vendor_filter=""):
    cameras = []
    indexes = get_indexes()
    vendorIDs = get_VendorIDs()
    for index in indexes:
        if vendor_filter in vendorIDs[index * 2] or vendor_filter in vendorIDs[index * 2 + 1]:
            cameras.append([index, vendorIDs[index*2], vendorIDs[index*2 + 1]])
    return cameras


if __name__ == '__main__':
    print(get_cameras("6380"))
