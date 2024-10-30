from PIL import Image, ImageDraw


def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    return image

def run_inference(model, image, device):
    results = model(image, device=device)
    return results[0]

def post_process(model, image, predictions):
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    for detection in predictions.boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        confidence = detection.conf.item()
        label = model.names[int(detection.cls)]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{label} {confidence:.2f}", fill="white")

    return annotated_image