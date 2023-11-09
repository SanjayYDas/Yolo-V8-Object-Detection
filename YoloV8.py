import torch
import time
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv

class ObjectDetection:
    def __init__(self, source, model_name):
        self.source = source
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model(model_name)
        self.CLASS_NAMES_DICT = self.load_class_names('/home/dell/yolov8/coco.names')
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4)

    def process_frame(self, frame):
        results = self.model(frame)
        return results

    def resize_frame(self, frame, target_size=(640,480)):
        # Get the original size of the frame
        height, width = frame.shape[:2]

        # Calculate the aspect ratio
        aspect_ratio = width / height

        # Calculate the new width based on the target height while maintaining the aspect ratio
        new_width = int(target_size[1] * aspect_ratio)

        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, target_size[1]))

        return resized_frame

    def load_model(self, model_name):
        model = YOLO(model_name)  # load a pretrained YOLOv8n model
        model.fuse()
        return model

    def load_class_names(self, class_file):
        with open(class_file, 'rt') as f:
            class_names = f.read().rstrip('\n').split('\n')
        return class_names

    def annotate_frame(self, frame, results):
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        # Create labels with class labels and confidence values
        labels = [f"{self.CLASS_NAMES_DICT[class_id]}: {confidence:.2f}" for class_id, confidence in zip(class_ids, confidences)]

        # Setup detections for visualization
        detections = sv.Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=confidences,
            class_id=class_ids,
        )

        # Annotate and display frame
        annotated_frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        return annotated_frame

    def process_video(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("Error: Unable to open video source or image file.")
            return

        # Get the video's original width and height
        original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Define the output video path
        output_path = "output.mp4"

        # Get the video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(original_width)
        frame_height = int(original_height)

        # Create a VideoWriter object to save the output video
        output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        # Create a named window and resize it
        cv2.namedWindow('YOLOv8 Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLOv8 Detection', frame_width, frame_height)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()

            # Resize frame
            resized_frame = self.resize_frame(frame)

            # Process frame and obtain results
            results = self.process_frame(resized_frame)

            # Annotate the resized frame
            annotated_frame = self.annotate_frame(resized_frame, results)

            end_time = time.time()
            fps = 1 / np.round(end_time - start_time, 2)

            cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write the frame to the output video file
            output_video.write(annotated_frame)

            cv2.imshow('YOLOv8 Detection', annotated_frame)

            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q'):
                break

        # Release the resources
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

    def process_image(self):
        frame = cv2.imread(self.source)

        # Resize frame
        resized_frame = self.resize_frame(frame)

        results = self.process_frame(resized_frame)
        annotated_frame = self.annotate_frame(resized_frame, results)

        # Display the annotated image
        cv2.imshow('YOLOv8 Detection', annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __call__(self):
        if self.source.endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv', '.webm', '.wmv')):
            self.process_video()
        elif self.source.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            self.process_image()
        else:
            print("Error: Unsupported file format.")

if __name__ == "__main__":
    source = "/home/dell/yolov8/pexels-rachel-claire-4846221.jpg"  # Update with your input source
    model_path = "/home/dell/yolov8/yolov8l.pt"
    detector = ObjectDetection(source, model_path)
    detector()
