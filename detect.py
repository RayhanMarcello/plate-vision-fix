import cv2
from ultralytics import YOLO

def main():
    # Load the ONNX model
    # Ultralytics automatically handles ONNX loading if the file extension is .onnx
    print("Loading model...")
    try:
        model = YOLO("best.pt")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Open the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam detection. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Run inference on the frame
        # stream=True is recommended for video sources
        results = model(frame, stream=True)

        # Iterate over the results and plot them on the frame
        for result in results:
            annotated_frame = result.plot()
            
            # Display the frame with annotations
            cv2.imshow("YOLO Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
