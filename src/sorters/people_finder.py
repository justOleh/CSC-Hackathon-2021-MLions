from torchvision import models, transforms
import torch
from PIL import Image
import cv2 as cv
import numpy as np
import os
import face_detection
from shutil import copyfile


from src.sorters.abstract import AbstractSorter


class PeopleFinder(AbstractSorter):
    def __init__(self, input_path: str, output_path: str, config_path: str = 'configs/people_finder.yaml'):
        super().__init__(input_path, output_path, config_path)
        self.transforms_f = transforms.Compose([
            lambda x: cv.cvtColor(x, cv.COLOR_BGR2RGB),
            Image.fromarray,
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.dist_threshold = 250
        self.face_detector = face_detection.build_detector(
            "DSFDDetector", confidence_threshold = .5, nms_iou_threshold = .3)

        # self.res_folder_name = self.input_path / 'blurred'
        # os.makedirs(self.res_folder_name, exist_ok=True)

    def _find_face(self, img: np.ndarray) -> tuple:
        pass

    def _calc_description(self, img: np.ndarray):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        model_ft = models.resnet18(pretrained=True)
        model_ft.layer4.register_forward_hook(get_activation('layer4'))
        output = model_ft(torch.unsqueeze(self.transforms_f(img), dim=0))
        return torch.flatten(activation['layer4'])

    def _calc_distance(self, descriptor1: torch.Tensor, descriptor2: torch.Tensor) -> float:
        return np.linalg.norm(descriptor1-descriptor2)

    def process(self):
        images_path = [file for file in self.input_path.glob("**/*")
                       if 'png' in str(file) or 'jpg' in str(file) or 'jpeg' in str(file)]

        #calc descriptions for all images in dir
        descriptors = []
        images_path_cluster = []
        for idx, image_path in enumerate(images_path):
            img = cv.imread(str(image_path))
            faces_bb = self.face_detector.detect(cv.cvtColor(img, cv.COLOR_BGR2RGB)).astype(int)
            if not len(faces_bb):
                continue
            xmin, ymin, xmax, ymax, _ = faces_bb[0]
            # xmin, ymin, xmax, ymax, score
            roi = img[ymin:ymax, xmin:xmax, :]
            # self.imshow(roi)
            # print(roi)
            # self.imshow(roi)
            descriptor = self._calc_description(roi)
            print(descriptor)
            images_path_cluster.append(image_path)
            descriptors.append(descriptor)
        print(descriptors)

        #calc distances, find nearest and group
        clusters = []
        i = 0
        while i < len(images_path_cluster):
            cluster = [images_path_cluster[i]]
            j = 0
            while j < len(images_path_cluster):
                if i == j:
                    j += 1
                    continue
                print(self._calc_distance(descriptors[i], descriptors[j]))
                if self._calc_distance(descriptors[i], descriptors[j]) <= self.dist_threshold:
                    cluster.append(images_path_cluster[j])
                    del descriptors[j]
                    del images_path_cluster[j]
                else:
                    j += 1
            clusters.append(cluster)
            i += 1
        print(clusters)

        #move clusters to folders
        for idx, cluster in enumerate(clusters):
            for image_path in cluster:
                cluster_folder_path = self.output_path / f'{idx}'
                os.makedirs(cluster_folder_path, exist_ok=True)
                copyfile(image_path, cluster_folder_path/image_path.name)

    def imshow(self, img, window_name='main', wait=0):
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.imshow(window_name, img)
        cv.waitKey(wait)
        cv.destroyWindow(window_name)

