from typing import Union, Dict
from pathlib import Path

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from datetime import datetime


def decode_gps_data(gps: Dict):
    return {GPSTAGS.get(tag, tag): value for tag, value in gps.items()}


def get_exif_data(image: Image):
    """Extract exif data from a PIL Image to a dictionary, converting GPS tags."""
    exif = {TAGS.get(tagId, tagId): value for tagId, value in image._getexif().items()}
    if 'GPSInfo' in exif:
        exif['GPSInfo'] = decode_gps_data(exif['GPSInfo'])
    return exif


def get_date_taken(exif: Dict) -> datetime:
    time_key = 'DateTimeOriginal'
    try:
        time_str = exif[time_key]
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
        return time
    except KeyError as e:
        print(f'Error: invalid key {e}')


def get_place_taken(exif: Dict):
    gps_key = 'GPSInfo'
    try:
        return exif[gps_key]
    except KeyError as e:
        print(f'Error: invalid key {e}')


if __name__ == '__main__':
    exif_data = get_exif_data(Image.open('../data_samples/IMG_5459.JPG'))
    print(get_date_taken(exif_data))
