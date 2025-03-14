import gradio as gr
from ultralytics import YOLOv10 

def yolov10_inference(image, model_path, image_size, conf_threshold):
    import torch
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running inference on device: {device}")
    
    model = YOLOv10(model_path)
    model.model.to(device)  # Ensure the model is moved to device

    # Perform prediction
    results = model.predict(source=image, imgsz=image_size, conf=conf_threshold, save=True)

    # Extract detection details
    def extract_detection_details(results):
        all_detections = []
        for r in results:  # loop over batch
            boxes = r.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
            class_ids = r.boxes.cls.cpu().numpy().astype(int)  # class IDs
            confidences = r.boxes.conf.cpu().numpy()  # confidence scores
            class_names = [r.names[cid] for cid in class_ids]  # map IDs to names

            detections = []
            for box, cls_id, conf, cls_name in zip(boxes, class_ids, confidences, class_names):
                detections.append({
                    "bounding_box": [float(b) for b in box],
                    "class_id": int(cls_id),
                    "confidence": float(conf),
                    "class_name": cls_name
                })
            all_detections.append(detections)
        return all_detections

    detections = extract_detection_details(results)

    # Format detections as text
    def format_detections_as_text(detections):
        formatted = []
        for det in detections:
            line = (
                f"Class: {det['class_name']} (ID: {det['class_id']}) | "
                f"Confidence: {det['confidence']:.2f} | "
                f"Bounding Box: {det['bounding_box']}"
            )
            formatted.append(line)
        return "\n".join(formatted)

    detection_text = format_detections_as_text(detections[0]) if detections else "No detections"

    # Return annotated image and detection text
    return model.predictor.plotted_img[:, :, ::-1], detection_text

def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image")
                
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov10n.pt",
                        "yolov10s.pt",
                        "yolov10m.pt",
                        "yolov10b.pt",
                        "yolov10l.pt",
                        "yolov10x.pt",
                    ],
                    value="yolov10s.pt",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.25,
                )
                yolov10_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image")
                output_text = gr.Textbox(label="Detection Details")

        yolov10_infer.click(
            fn=yolov10_inference,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image, output_text],
        )

        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    "yolov10s.pt",
                    640,
                    0.25,
                ],
                [
                    "ultralytics/assets/zidane.jpg",
                    "yolov10s.pt",
                    640,
                    0.25,
                ],
            ],
            fn=yolov10_inference,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image, output_text],
            cache_examples=True,
        )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv10: Real-Time End-to-End Object Detection
    </h1>
    """)
    with gr.Row():
        with gr.Column():
            app()

gradio_app.launch(debug=True)