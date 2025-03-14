import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLOv10
 
if __name__ == '__main__':
    model = YOLOv10('ultralytics/cfg/models/v10/yolov10s-C2f_deformable_LKA+dysample.yaml')
    # model.load('yolov10S.pt')  # Uncomment if you want to load pretrained weights
    model.train(data='yolo-bvn.yaml',
                cache=False,
                imgsz=416,
                epochs=10,
                batch=2,
                close_mosaic=10,
                device='cpu',
                optimizer='SGD',  # using SGD
                project='runs/train',
                name='fish_detection_v10',
                )