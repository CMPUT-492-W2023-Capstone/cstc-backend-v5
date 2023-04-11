import torch

from utils.general import non_max_suppression

from trackers.multi_tracker_zoo import create_tracker

from configuration import InputConfig, AlgorithmConfig, OutputResultConfig


class DetectionTask:

    def __init__(self,
                 input_config: InputConfig,
                 algorithm_config: AlgorithmConfig,
                 output_result_config: OutputResultConfig,
                 ):
        self.input_config = input_config
        self.algorithm_config = algorithm_config
        self.output_result_config = output_result_config

    def _uint_to_float(self, im):
        new_im = torch.from_numpy(im).to(self.input_config.device)
        new_im = new_im.half() if self.algorithm_config.fp16 else new_im.float()
        new_im /= 255.0
        if len(new_im) == 3:
            new_im = new_im[None]

        return new_im

    def get_detection_objs(self, im, model, max_det):
        im = self._uint_to_float(im)

        predictions = model(im, augment=self.algorithm_config.augment)
        detections = non_max_suppression(
            predictions,
            self.algorithm_config.conf_thres,
            self.algorithm_config.iou_thres,
            classes=self.algorithm_config.class_filter,
            agnostic=self.algorithm_config.agnostic_nms,
            max_det=max_det
        )

        return detections, im, model


class TrackingTask:

    def __init__(self, dataset_size: int, tracking_method, tracking_config, reid_models, device, fp16: bool):
        self.tracker_hosts: list = []

        for i in range(dataset_size):
            tracker = create_tracker(
                tracking_method,
                tracking_config,
                reid_models,
                device,
                fp16
            )

            self.tracker_hosts.append(tracker)
            if hasattr(self.tracker_hosts[i], 'model') \
                    and hasattr(self.tracker_hosts[i].model, 'warmup'):
                self.tracker_hosts[i].model.warmup()

    def motion_compensation(self, tracker_index, current_frame, prev_frame):
        if hasattr(self.tracker_hosts[tracker_index], 'tracker') \
                and hasattr(self.tracker_hosts[tracker_index].tracker, 'camera_update'):
            if current_frame is not None and prev_frame is not None:
                self.tracker_hosts[tracker_index].tracker.camera_update(prev_frame, current_frame)
