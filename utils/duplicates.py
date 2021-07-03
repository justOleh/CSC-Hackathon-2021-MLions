import imagehash
from PIL import Image


def are_duplicates(image1: Image, image2: Image, threshold: int = 10) -> bool:
    hash1 = imagehash.phash(image1)
    hash2 = imagehash.phash(image2)
    return hash1 - hash2 <= threshold


if __name__ == '__main__':
    street1 = Image.open('../data_samples/IMG_0711.JPG')
    street2 = Image.open('../data_samples/IMG_0712.JPG')
    assert are_duplicates(street1, street2)

    page1 = Image.open('../data_samples/IMG_0943.JPG')
    page2 = Image.open('../data_samples/IMG_0944.JPG')
    page3 = Image.open('../data_samples/IMG_0945.JPG')
    assert are_duplicates(page1, page2)
    assert not are_duplicates(page1, page3)
    assert not are_duplicates(page2, page3)

    assert not are_duplicates(street1, page1)
