import enum
import glob
import json
import os
import time
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import roboflow
import supervision as sv
from PIL import Image
from supervision.utils.file import save_text_file
from tqdm import tqdm

from autodistill.core import BaseModel
from autodistill.helpers import load_image, split_data

from .detection_ontology import DetectionOntology


class NmsSetting(str, enum.Enum):
    NONE = "no_nms"
    CLASS_SPECIFIC = "class_specific"
    CLASS_AGNOSTIC = "class_agnostic"


@dataclass
class DetectionBaseModel(BaseModel):
    ontology: DetectionOntology

    @abstractmethod
    def predict(self, input: str | np.ndarray | Image.Image) -> sv.Detections:
        pass

    def sahi_predict(self, input: str | np.ndarray | Image.Image) -> sv.Detections:
        slicer = sv.InferenceSlicer(callback=self.predict)

        return slicer(load_image(input, return_format="cv2"))

    def _record_confidence_in_files(
        self,
        annotations_directory_path: str,
        image_names: List[str],
        annotations: Dict[str, sv.Detections],
    ) -> None:
        Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)
        for image_name in image_names:
            detections = annotations[image_name]
            yolo_annotations_name, _ = os.path.splitext(image_name)
            confidence_path = os.path.join(
                annotations_directory_path,
                "confidence-" + yolo_annotations_name + ".txt",
            )
            if detections.confidence is None:
                raise ValueError("Expected detections to have confidence values.")
            confidence_list = [str(x) for x in detections.confidence.tolist()]
            save_text_file(lines=confidence_list, file_path=confidence_path)
            print("Saved confidence file: " + confidence_path)

    def label(
        self,
        input_folder: str,
        extension: str = ".jpg",
        output_folder: str | None = None,
        human_in_the_loop: bool = False,
        roboflow_project: str | None = None,
        roboflow_tags: list[str] = ["autodistill"],
        sahi: bool = False,
        record_confidence: bool = False,
        nms_settings: NmsSetting = NmsSetting.NONE,
    ) -> sv.DetectionDataset:
        """
        Label a dataset with the model.
        """
        if output_folder is None:
            output_folder = input_folder + "_labeled"

        os.makedirs(output_folder, exist_ok=True)

        image_paths = glob.glob(input_folder + "/*" + extension)
        detections_map = {}

        # if output_folder/autodistill.json exists
        if os.path.exists(output_folder + "/data.yaml"):
            dataset = sv.DetectionDataset.from_yolo(
                output_folder + "/images",
                output_folder + "/annotations",
                output_folder + "/data.yaml",
            )

            # DetectionsDataset iterator returns
            # image_name, image, self.annotations.get(image_name, None)
            # ref: https://supervision.roboflow.com/datasets/#supervision.dataset.core.DetectionDataset
            for item in dataset:
                image_name = item[0]
                image = item[1]
                detections = item[2]

                image_base_name = os.path.basename(image_name)
                images_map[image_base_name] = image.copy()

                annotation_path = os.path.join(output_folder, "annotations", image_base_name + ".txt")
                detections_map[image_base_name] = detections

        files = glob.glob(input_folder + "/*" + extension)
        progress_bar = tqdm(files, desc="Labeling images")

        for f_path in progress_bar:
            progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)

            image = cv2.imread(f_path)
            if sahi:
                detections = slicer(image)
            else:
                detections = self.predict(image)

            if nms_settings == NmsSetting.CLASS_SPECIFIC:
                detections = detections.with_nms()
            if nms_settings == NmsSetting.CLASS_AGNOSTIC:
                detections = detections.with_nms(class_agnostic=True)

            f_path_short = os.path.basename(f_path)
            images_map[f_path_short] = image.copy()

            annotation_path = os.path.join(
                output_folder,
                "annotations/",
                ".".join(f_path_short.split(".")[:-1]) + ".txt",
            )

            if not os.path.exists(annotation_path):
                detections = self.predict(f_path)
                detections_map[f_path_short] = detections


        dataset = sv.DetectionDataset(
            self.ontology.classes(), image_paths, detections_map
        )

        dataset.as_yolo(
            output_folder + "/images",
            output_folder + "/annotations",
            min_image_area_percentage=0.01,
            data_yaml_path=output_folder + "/data.yaml",
        )

        if record_confidence:
            image_names = [os.path.basename(f_path) for f_path in image_paths]
            self._record_confidence_in_files(
                output_folder + "/annotations", image_names, detections_map
            )
        split_data(output_folder, record_confidence=record_confidence)

        if human_in_the_loop:
            roboflow.login()

            rf = roboflow.Roboflow()

            workspace = rf.workspace()

            workspace.upload_dataset(output_folder, project_name=roboflow_project)

        config["end_time"] = time.time()
        config["labeled_image_count"] = len(dataset)
        config["human_in_the_loop"] = human_in_the_loop
        config["roboflow_project"] = roboflow_project
        config["roboflow_tags"] = roboflow_tags
        config["task"] = "detection"

        with open(os.path.join(output_folder, "config.json"), "w+") as f:
            json.dump(config, f)

        print("Labeled dataset created - ready for distillation.")

        return dataset, output_folder
