# Import necessary libraries
import cv2
import argparse
import supervision as sv
from ultralytics import YOLO
import numpy as np

## CHANGE TO YOUR RETRAINED WEIGHTS
model = YOLO("/Users/noah_/Documents/Development/2024_Projects/real_time_video_analytics/runs/detect/train3/weights/best.pt")

tracker = sv.ByteTrack(minimum_consecutive_frames=3)
tracker.reset()

color_mapping = {
    'Excavator': sv.Color.YELLOW,
    'Gloves': sv.Color.GREEN,
    'Hardhat': sv.Color.BLUE,
    'Ladder': sv.Color.YELLOW,
    'Mask': sv.Color.GREEN,
    'NO-Hardhat': sv.Color.GREEN,  # New mapping
    'NO-Mask': sv.Color.RED,      # New mapping
    'NO-Safety Vest': sv.Color.RED,  # New mapping
    'Person': sv.Color.BLUE,       # New mapping
    'SUV': sv.Color.YELLOW,     # New mapping
    'Safety Cone': sv.Color.YELLOW, # New mapping
    'Safety Vest': sv.Color.GREEN,  # New mapping
    'bus': sv.Color.BLUE,            # New mapping
    'dump truck': sv.Color.BLACK,    # New mapping
    'fire hydrant': sv.Color.YELLOW,    # New mapping
    'machinery': sv.Color.BLACK,       # New mapping
    'mini-van': sv.Color.BLACK,  # New mapping
    'sedan': sv.Color.BLACK,    # New mapping
    'semi': sv.Color.BLACK,       # New mapping
    'trailer': sv.Color.BLACK,    # New mapping
    'truck and trailer': sv.Color.BLACK,  # New mapping
    'truck': sv.Color.BLACK,      # New mapping
    'van': sv.Color.BLACK,      # New mapping
    'vehicle': sv.Color.BLACK,  # New mapping
    'wheel loader': sv.Color.BLACK,  #
}

# Create a ColorPalette from the color_mapping
colors_list = list(color_mapping.values())  # Use the Color objects directly
COLORS = sv.ColorPalette(colors_list)

# Initialize the BoxAnnotator with the color palette
box_annotator = sv.BoxAnnotator(color=COLORS)
label_annotator = sv.LabelAnnotator(color=COLORS, text_color=sv.Color.BLACK)
trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2)

## Filtering specific classes
# classes = [0, 2, 3, ]


classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
              'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
              'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']


def main(video_file_path):
    frame_generator = sv.get_video_frames_generator(source_path=video_file_path)
    paused = False  # Initialize the pause state

    while True:
        if not paused:  # Only get the next frame if not paused
            try:
                frame = next(frame_generator)  # Get the next frame from the generator
            except StopIteration:
                break  # Exit the loop if there are no more frames

            result = model(frame, device="mps")[0]
            detections = sv.Detections.from_ultralytics(result)

            # Get the indices for the classes of interest
            no_safety_vest_index = classNames.index('NO-Safety Vest')
            no_hardhat_index = classNames.index('NO-Hardhat')

            # Filter detections based on confidence levels
            filtered_detections = []
            filtered_class_ids = []
            filtered_confidences = []

            for idx in range(len(detections.class_id)):
                class_id = detections.class_id[idx]
                confidence = detections.confidence[idx]

                # Determine if the detection should be shown
                if (class_id == no_safety_vest_index or class_id == no_hardhat_index):
                    if confidence >= 0.6:  # More stringent requirement for "NO-Safety Vest" and "NO-Hardhat"
                        filtered_detections.append(detections.xyxy[idx])  # Append bounding box coordinates
                        filtered_class_ids.append(class_id)  # Append class ID
                        filtered_confidences.append(confidence)  # Append confidence score
                else:
                    if confidence >= 0.4:  # Show other classes with confidence above 0.5
                        filtered_detections.append(detections.xyxy[idx])  # Append bounding box coordinates
                        filtered_class_ids.append(class_id)  # Append class ID
                        filtered_confidences.append(confidence)  # Append confidence score

            # Create a new Detections object for the tracker
            if filtered_detections:
                filtered_detections = sv.Detections(
                    xyxy=np.array(filtered_detections),
                    confidence=np.array(filtered_confidences),
                    class_id=np.array(filtered_class_ids)
                )
            else:
                filtered_detections = sv.Detections(xyxy=np.empty((0, 4)), confidence=np.empty((0,)), class_id=np.empty((0,)))

            # Update detections with the filtered results
            updated_detections = tracker.update_with_detections(filtered_detections)

            # Create labels using the tracker IDs and filtered confidences
            labels = [f"{classNames[class_id]} ({confidence:.2f})" for class_id, confidence in zip(updated_detections.class_id, updated_detections.confidence)]

            annotated_frame = frame.copy()

            annotated_frame = box_annotator.annotate(
                scene=annotated_frame,
                detections=updated_detections
            )

            # Annotate labels with confidence scores
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=updated_detections,
                labels=labels  # Pass the labels with confidence scores
            )

            cv2.imshow("Processed Video", annotated_frame)

        # Key event handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Quit the video
            break
        elif key == ord("p"):  # Pause the video
            paused = not paused  # Toggle pause state

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file_path", type=str, required=True)
    args = parser.parse_args()
    main(args.video_file_path)