import base64
import pickle
import threading
import numpy as np
import cv2
from insightface.app import FaceAnalysis

# InsightFace model loading is somewhat expensive (loads several ONNX
# models from disk), so we initialize it once per process and reuse it
# across requests, rather than once per capture call.
_face_app = None
_face_app_lock = threading.Lock()


def get_face_app():
    global _face_app
    if _face_app is None:
        with _face_app_lock:
            if _face_app is None:
                app = FaceAnalysis(name='buffalo_l')
                # ctx_id=0 uses GPU 0 if available via onnxruntime-gpu;
                # falls back to CPU automatically if no GPU provider is
                # installed. Use ctx_id=-1 to force CPU explicitly.
                app.prepare(ctx_id=1, det_size=(320, 320))
                _face_app = app
    return _face_app


def decode_base64_image(data_url):
    """
    Decode a base64 data-URL (e.g. "data:image/jpeg;base64,...") captured by
    the browser's <canvas> into an OpenCV BGR frame. Returns None on failure.
    """
    try:
        if ',' in data_url:
            _, encoded = data_url.split(',', 1)
        else:
            encoded = data_url
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def capture_and_encode_faces(person_name, images_data):
    """
    images_data: a list of base64 data-URL strings, each one a frame captured
    by the user's browser webcam via getUserMedia + canvas.

    Uses InsightFace (buffalo_l: RetinaFace/SCRFD detector + ArcFace
    recognition model) instead of dlib/face_recognition. Detection and
    embedding extraction both happen in a single app.get(frame) call.
    """
    face_app = get_face_app()

    known_face_encodings = []
    known_face_names = []

    for data_url in images_data or []:
        frame = decode_base64_image(data_url)
        if frame is None:
            continue

        faces = face_app.get(frame)
        if not faces:
            continue

        # If multiple faces are detected in one frame, use the largest one
        # (closest to the camera) -- avoids picking up bystanders.
        faces.sort(key=lambda f: (
            f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        best_face = faces[0]

        # normed_embedding is already L2-normalized, which makes matching a
        # plain dot product / cosine similarity later.
        known_face_encodings.append(best_face.normed_embedding)
        known_face_names.append(person_name)

    if known_face_encodings:
        save_data(known_face_encodings, known_face_names)

    return len(known_face_encodings)


def save_data(encodings, names, filename="face_encodings.pkl"):
    existing_encodings, existing_names = load_data(filename)
    updated_encodings = existing_encodings + [np.asarray(e) for e in encodings]
    updated_names = existing_names + names

    data = {"encodings": updated_encodings, "names": updated_names}
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_data(filename="face_encodings.pkl"):
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
    except FileNotFoundError:
        known_face_encodings = []
        known_face_names = []

    return known_face_encodings, known_face_names
