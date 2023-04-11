import logging

from enum import Enum
from pathlib import Path

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import check_file, check_img_size
from utils.torch_utils import select_device


APP_PATH = Path(__file__).resolve().parents[0].parents[0]
MODELS_PATH = APP_PATH / 'models'
TRACKING_CONFIG = APP_PATH / 'tracking_configs'
VALID_URLs = ('rtsp://', 'rtmp://', 'https://')
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes


def _is_valid_file(source) -> bool:
    return Path(source).suffix[1:] in VID_FORMATS


def _is_valid_url(source) -> bool:
    return source.lower().startswith(VALID_URLs)


def _is_valid_webcam(source) -> bool:
    return source.isnumeric() or source.endswith('.txt')


class ComputingDevice(Enum):
    CUDA_0 = 0
    CUDA_1 = 1
    CUDA_2 = 2
    CUDA_3 = 3
    CPU = 'CPU'


class TrackingMethod(Enum):
    BOTSORT = 'botsort'
    BYTETRACK = 'bytetrack'
    DEEPOCSORT = 'deepocsort'
    OCSORT = 'ocsort'
    STRONGSORT = 'strongsort'


class InputConfig:

    def __init__(
            self,
            device=ComputingDevice.CPU.value,
            reid_models=MODELS_PATH / 'osnet_x0_25_msmt17.pt',
            media_source: str = '0',
            yolo_models=MODELS_PATH / 'model.pt',
            web_cam_ratio=None
    ):
        self.device = select_device(device)
        logging.info(f'{device} selected')

        self.media_source = check_file(media_source) \
            if _is_valid_file(media_source) and _is_valid_url(media_source) else media_source

        self.reid_models = reid_models
        self.yolo_models = yolo_models

        self.webcam_enable = _is_valid_webcam(media_source) or \
            (_is_valid_url(media_source) and not _is_valid_file(media_source))
        self.segmentation = self.yolo_models.name.endswith('-seg')

        self.web_cam_ratio = web_cam_ratio
        if self.webcam_enable and self.web_cam_ratio is None:
            logging.warning('Please specify aspect of a camera (example: 1920 x 1080) '
                            'in the config.yaml')
            exit(1)

    def load_dataset_model(self, inference_img_size, vid_frame_stride, dnn=False, fp16=False):
        model = DetectMultiBackend(self.yolo_models, device=self.device,
                                   dnn=dnn, data=None, fp16=fp16)
        logging.info(f'{self.yolo_models} loaded')

        stride, pt = model.stride, model.pt
        inference_img_size = check_img_size(inference_img_size, s=stride)

        if self.webcam_enable:
            media_dataset = LoadStreams(self.media_source, auto=pt,
                                        img_size=inference_img_size,
                                        stride=stride, vid_stride=vid_frame_stride)
            logging.info('Input frames from stream / camera')
        else:
            media_dataset = LoadImages(self.media_source, auto=pt,
                                       img_size=inference_img_size,
                                       stride=stride, vid_stride=vid_frame_stride)
            logging.info('Input frames from media files')

        media_dataset_size = 1 if self.webcam_enable else len(media_dataset)

        logging.info(f'{media_dataset_size} media sources are detected and loaded')

        return inference_img_size, media_dataset, media_dataset_size, model


class OutputVideoConfig:

    def __init__(
            self,
            line_thickness: int = 2,
            hide_conf: bool = False,
            hide_class: bool = False,
            hide_labels: bool = False,
            show_video: bool = False,
            enable_trace: bool = False,
            vid_frame_stride: int = 1,  # number of frame will skip per second
            retina_masks: bool = False,
    ):
        self.line_thickness = line_thickness
        self.hide_conf = hide_conf
        self.hide_class = hide_class
        self.hide_labels = hide_labels
        self.show_video = show_video
        self.enable_trace = enable_trace
        self.vid_frame_stride = vid_frame_stride
        self.retina_masks = retina_masks


class OutputResultConfig:

    def __init__(
            self,
            no_save: bool = False,
            save_csv: bool = False,
            upload_period: int = 300,
    ):
        self.no_save = no_save
        self.save_csv = save_csv
        self.upload_period = upload_period


class AlgorithmConfig:

    def __init__(
            self,
            agnostic_nms: bool = False,
            augment: bool = False,
            blur_frames_limit: int = 50,
            blur_thersold: float = 100.0,
            classify: bool = False,
            class_filter: list = None,
            conf_thres: float = 0.25,
            device: ComputingDevice = ComputingDevice.CPU,
            dnn: bool = False,
            fp16: bool = False,
            inference_img_size: list = None,
            iou_thres: float = 0.5,
            max_det: int = 1000,
            tracking_method: TrackingMethod = TrackingMethod.BYTETRACK,
            tracking_config=TRACKING_CONFIG / 'bytetrack.yaml',
            velocity_thersold_delta: float = 0,
            update_period: int = 50
    ):
        if class_filter is None:
            class_filter = []
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.blur_frames_limit = blur_frames_limit
        self.blur_thersold = blur_thersold
        self.classify = classify
        self.class_filter = class_filter
        self.conf_thres = conf_thres
        self.device = device
        self.dnn = dnn
        self.fp16 = fp16
        self.iou_thres = iou_thres
        self.max_det = max_det

        if inference_img_size is None or len(inference_img_size) > 2 or len(inference_img_size) < 1:
            self.inference_img_size = [640, 640]
        else:
            self.inference_img_size = inference_img_size if len(inference_img_size) == 2 else 2 * inference_img_size

        self.tracking_method = tracking_method.value
        self.tracking_config = tracking_config
        self.velocity_thersold_delta = velocity_thersold_delta
        self.update_period = update_period
