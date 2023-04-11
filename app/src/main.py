import asyncio
import math
import logging
import platform
from jsonargparse import CLI
from pathlib import Path

import cv2
import torch

from utils.general import scale_boxes
from utils.plots import Annotator, colors

import upload

from algorithm import DetectionTask, TrackingTask
from configuration import InputConfig, OutputVideoConfig, OutputResultConfig, AlgorithmConfig


EVENT_LOOP = asyncio.get_event_loop()


def get_center(bbox):
    return (bbox[2] - bbox[0]) / 2, (bbox[3] - bbox[1]) / 2


def get_velocity(x1: float, x2: float, y1: float, y2: float) -> float:
    return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


class TrackedObject:

    def __init__(self, v_id: int = -1, vehicle_type: str = '', confidence: int = -1, bbox: list = None):
        self.v_id = v_id
        self.vehicle_type = vehicle_type
        self.confidence = confidence
        self.first_bbox = bbox
        self.bbox = bbox
        # For estimate whether if a vehicle will exist
        # self.area_history = [abs(self.bbox[2] - self.bbox[0]) * abs(self.bbox[3] - self.bbox[1])]
        self.age = 1

    # def get_rate_of_area(self, delta=10):
    #     length = len(self.area_history)
    #     if delta > length:
    #         return (self.area_history[length - 1] - self.area_history[0]) / length
    #     else:
    #         return (self.area_history[length - 1] - self.area_history[length - delta]) / delta

    def get_label(self, config: OutputVideoConfig) -> str:
        label = 'Empty'
        if not config.hide_labels:
            class_label = None if config.hide_class else f'{self.vehicle_type}'
            confidence_label = None if config.hide_conf else f'{self.confidence:.2f}'

            label = f'{self.v_id} {class_label} {confidence_label}'

        return label

    def get_velocity(self) -> float:
        x1, y1 = get_center(self.first_bbox)
        x2, y2 = get_center(self.bbox)
        return get_velocity(x1, x2, y1, y2)

    """
    Check whether if two objects overlap each other
    bbox is x1', y1', x2', and y2'
    self is x1 , y1 , x2 , and y2
    """

    # def is_overlap(self, bbox):
    #     return (self.bbox[0] <= bbox[2] and bbox[0] <= self.bbox[2]) and (  # X overlaps
    #             self.bbox[1] <= bbox[3] and bbox[1] <= self.bbox[3])  # Y overlaps

    def update(self, vehicle_type: str = '', confidence: int = -1, bbox: list = None):
        self.vehicle_type = vehicle_type
        self.confidence = confidence
        self.bbox = bbox
        # self.area_history.append(abs(self.bbox[2] - self.bbox[0]) * abs(self.bbox[3] - self.bbox[1]))
        self.age += 1

    def __str__(self):
        return f'Vehicle Id: {self.v_id} | Vehicle Type: {self.vehicle_type} ' \
               f'| Confidence: {self.confidence} | (X1, Y1, X2, Y2): ' \
               f'({self.bbox[0]}, {self.bbox[1]}, {self.bbox[2]}, {self.bbox[3]})'

    def __repr__(self):
        return self.__str__()


def legacy_selection(v_id, class_name, confidence, bbox, box_annotator, output_video_config,
                     candidates, selective_candidates):
    if v_id in selective_candidates.keys():
        selective_candidates[v_id].update(class_name, confidence, bbox)
    else:
        selective_candidates[v_id] = TrackedObject(
                v_id,
                class_name,
                confidence,
                bbox
        )

    box_annotator.box_label(
            selective_candidates[v_id].bbox,
            selective_candidates[v_id].get_label(output_video_config),
            color=colors(1, True)
    )


def selection(v_id, class_name, confidence, bbox, box_annotator, output_video_config,
              candidates, selective_candidates):
    if v_id in selective_candidates.keys():
        return
    if v_id not in candidates.keys():
        candidates[v_id] = TrackedObject(
                v_id,
                class_name,
                confidence,
                bbox
        )
    else:
        candidates[v_id].update(class_name, confidence, bbox)

    box_annotator.box_label(
            candidates[v_id].bbox,
            candidates[v_id].get_label(output_video_config),
            color=colors(1, True)
    )


def stream_result(box_annotator: Annotator, im0, source_path, stream_windows: list):
    im0 = box_annotator.result()

    if platform.system() == 'Linux' and source_path not in stream_windows:
        stream_windows.append(source_path)

        cv2.namedWindow(str(source_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(str(source_path), im0.shape[1], im0.shape[0])

    cv2.imshow(str(source_path), im0)

    if cv2.waitKey(1) == ord('q'):
        EVENT_LOOP.close()
        exit()


def is_at_boundary(x: float, y: float, WIDTH: float, HEIGHT: float):
    return x <= 5.0 or y <= 5.0 or WIDTH - x <= 5.0 or HEIGHT - y <= 5.0


def main(
        input_config: InputConfig,
        output_video_config: OutputVideoConfig,
        output_result_config: OutputResultConfig,
        algorithm_config: AlgorithmConfig,
        legacy: bool = False
):
    # load video frames (media dataset) and AI model
    algorithm_config.inference_img_size, media_dataset, media_dataset_size, model = \
        input_config.load_dataset_model(
            algorithm_config.inference_img_size,
            output_video_config.vid_frame_stride,
            dnn=algorithm_config.dnn,
            fp16=algorithm_config.fp16,
        )

    # names of the classification defined in the AI model
    class_names = model.names

    # tracked vehicles that are identify as moving vehicle
    selective_candidates: {str: TrackedObject} = {}

    # Determine the dimension of media
    if input_config.webcam_enable:
        WIDTH, HEIGHT = input_config.web_cam_ratio
    else:
        WIDTH = media_dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        HEIGHT = media_dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    blur_frames_count: int = 0

    candidates: {str: TrackedObject} = {}  # currently tracked vehicles

    # Velocity thersold for determine whether if a vehicle is moving or noting moving
    VEL_THERSOLD = math.sqrt(WIDTH * WIDTH + HEIGHT * HEIGHT) \
        / algorithm_config.update_period
    VEL_THERSOLD = VEL_THERSOLD * (1 + algorithm_config.velocity_thersold_delta)

    tracker_outputs = []  # tracker_outputs = [[]] * media_dataset_size
    for i in range(media_dataset_size):
        tracker_outputs.append([])

    # current_frames, prev_frames = [[None] * media_dataset_size for i in range(2)]
    current_frames = []
    for i in range(media_dataset_size):
        current_frames.append(None)
    prev_frames = []
    for i in range(media_dataset_size):
        prev_frames.append(None)

    streaming_windows: list = []

    detection_task = DetectionTask(input_config, algorithm_config, output_result_config)
    tracking_task = TrackingTask(
        media_dataset_size,
        algorithm_config.tracking_method,
        algorithm_config.tracking_config,
        input_config.reid_models,
        input_config.device,
        algorithm_config.fp16
    )

    assert 0 < media_dataset_size == len(tracker_outputs) \
           and len(current_frames) > 0 and len(prev_frames) > 0 \
           and len(tracking_task.tracker_hosts) == media_dataset_size

    # OpenCV convention : im means image after modify, im0 means copy
    # of the image before modification (i.e. original image)
    logging.info('Entering main loop')
    for frame_index, batch in enumerate(media_dataset):
        source_paths, im, im0s, video_capture = batch
        logging.info(f'Start processing {frame_index} frame')

        # Blur detection
        # TODO: Blur detection does not work with camera
        if not input_config.webcam_enable and \
                cv2.Laplacian(
                    cv2.cvtColor(im0s, cv2.COLOR_BGR2GRAY),
                    cv2.CV_64F).var() < algorithm_config.blur_thersold:
            blur_frames_count += 1

        detection_objs, im,  model = detection_task.get_detection_objs(
            im, model, algorithm_config.max_det)
        logging.info(f'{len(detection_objs)} objects were detected')

        for i, detection in enumerate(detection_objs):
            if input_config.webcam_enable:
                im0 = im0s[i].copy()
                source_path = Path(source_paths[i])
            else:
                im0 = im0s.copy()
                source_path = source_paths

            current_frames[i] = im0

            box_annotator = Annotator(
                im0,
                line_width=output_video_config.line_thickness,
                example=str(class_names)
            )

            tracking_task.motion_compensation(i, current_frames[i], prev_frames[i])
            logging.info('Applying motion_compensation')

            if detection is None or not len(detection):
                continue

            detection[:, :4] = scale_boxes(
                im.shape[2:],
                detection[:, :4],
                im0.shape).round()

            tracker_outputs[i] = tracking_task.tracker_hosts[i].update(detection.cpu(), im0)
            logging.info(f'{len(tracker_outputs[i])} objects were tracked')

            if len(tracker_outputs[i]) < 0:
                continue

            for tracker_output in tracker_outputs[i]:
                v_id = tracker_output[4]
                class_name = class_names[int(tracker_output[5])]
                confidence = tracker_output[6]
                bbox = tracker_output[0:4]

                # New vehicle being tracked
                if legacy:
                    legacy_selection(v_id, class_name, confidence, bbox, box_annotator, output_video_config,
                                     None, selective_candidates)
                else:
                    selection(v_id, class_name, confidence, bbox, box_annotator, output_video_config,
                              candidates, selective_candidates)

            # Select candidate (moving not static)
            if not legacy and frame_index != 0 and frame_index % algorithm_config.update_period == 0:
                num_new_candidate = 0
                for v_id, candidate in candidates.items():
                    if candidate.get_velocity() >= VEL_THERSOLD \
                            and v_id not in selective_candidates.keys():

                        selective_candidates[v_id] = candidate
                        candidates[v_id] = None
                        num_new_candidate += 1

                candidates = {v_id: candidate
                              for v_id, candidate in candidates.items()
                              if candidate is not None}

                logging.info(f'{num_new_candidate} candidate were selected')

            if not input_config.webcam_enable and \
                    blur_frames_count != 0 and \
                    blur_frames_count >= algorithm_config.blur_frames_limit:

                asyncio.run(upload.notify_blur())
                blur_frames_count = 0

            if frame_index != 0 and frame_index % output_result_config.upload_period == 0:
                if legacy:
                    EVENT_LOOP.run_until_complete(
                            upload.static_time(
                                selective_candidates.copy(), 
                                class_names, 
                                algorithm_config.class_filter, 
                                output_result_config.save_csv
                            )
                    )
                else:
                    asyncio.run(upload.static_time(
                        selective_candidates.copy(),
                        class_names,
                        algorithm_config.class_filter,
                        output_result_config.save_csv
                    ))

            stream_result(box_annotator, im0, source_path, streaming_windows)
            logging.info(f'Finish processing {frame_index} frame')

            prev_frames[i] = current_frames[i]


if __name__ == '__main__':
    with torch.no_grad():
        CLI(main, as_positional=False)
