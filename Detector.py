import random
import pickle
import numpy as np
import cv2
from create_dataset import decode_base64_image, get_face_app

# ---------------------------------------------------------------------------
# ArcFace matching
# ---------------------------------------------------------------------------
# Cosine similarity threshold for a match. normed_embedding vectors are
# unit-length, so cosine similarity == dot product. buffalo_l's ArcFace
# embeddings typically separate same-person / different-person pairs well
# above/below ~0.4-0.5; tune this against your own test photos before
# relying on it for anything higher-stakes than a class project.
MATCH_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Liveness / challenge-response config
# ---------------------------------------------------------------------------
CHALLENGES = ("blink", "turn_left", "turn_right")

CHALLENGE_LABELS = {
    "blink": "Blink twice",
    "turn_left": "Turn your head left",
    "turn_right": "Turn your head right",
}

# Eye-openness ratio thresholds (see _eye_openness_ratio). This is an
# EAR-*like* signal (cluster height / cluster width around each eye), not
# the literal dlib-68-point EAR formula, so its absolute scale is its own --
# tune these two numbers against a short recording of yourself blinking
# before trusting them for anything beyond a class project.
EAR_CLOSED_THRESHOLD = 0.16
EAR_OPEN_THRESHOLD = 0.23

# Minimum yaw magnitude (approximate degrees) required to count a head turn
# as deliberate rather than incidental jitter.
YAW_TURN_THRESHOLD = 15.0

# If your webcam/mirroring setup makes "turn left" and "turn right" register
# backwards, flip this rather than touching the pose-estimation math.
YAW_LEFT_IS_NEGATIVE = True

# At least this fraction of submitted frames must contain a detectable face
# or we bail out early -- protects against mostly-blank/garbage submissions
# and against someone only holding a photo up for part of the sequence.
MIN_DETECTED_FRAME_RATIO = 0.5

# Generic, roughly-proportioned 3D face points (arbitrary units) used for
# solvePnP-based yaw estimation. This is the same style of generic model
# used in most lightweight webcam head-pose demos -- it is NOT a calibrated
# biometric measurement, just a consistent, thresholdable signal for "did
# the user turn their head". Adapted to eye *centers* (what InsightFace's
# 5-point kps gives us) rather than eye corners.
_MODEL_POINTS_3D = np.array([
    (-165.0, 170.0, -115.0),   # left eye (subject's left)
    (165.0, 170.0, -115.0),    # right eye (subject's right)
    (0.0, 0.0, 0.0),           # nose tip
    (-150.0, -150.0, -125.0),  # left mouth corner
    (150.0, -150.0, -125.0),   # right mouth corner
], dtype=np.float64)


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


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def choose_challenge():
    """Pick one of the three liveness challenges at random."""
    return random.choice(CHALLENGES)


def _estimate_yaw_degrees(kps, frame_shape):
    """
    Approximate head yaw (left/right turn) in degrees from InsightFace's
    5-point detector keypoints. kps order is [left_eye, right_eye, nose,
    left_mouth_corner, right_mouth_corner] -- this ordering is standard and
    stable across insightface detector releases (unlike the exact index
    layout of the 106-point landmark model, which has drifted between
    versions, so we deliberately don't hardcode indices into that model
    anywhere in this file).

    Uses cv2.solvePnP against the generic 3D model above, then decomposes
    the resulting rotation into Euler angles the same way most OpenCV
    head-pose tutorials do. Returns yaw in degrees, or None if pose can't
    be estimated for this frame.
    """
    if kps is None or len(kps) < 5:
        return None

    h, w = frame_shape[:2]

    image_points = np.array([
        kps[0],  # left eye
        kps[1],  # right eye
        kps[2],  # nose
        kps[3],  # left mouth corner
        kps[4],  # right mouth corner
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not ok:
        return None

    # Refine the EPNP estimate with a Gauss-Newton pass. Passing our own
    # initial guess (useExtrinsicGuess=True) lets ITERATIVE skip the
    # internal DLT initialization step that needs >=6 points -- so this
    # works fine purely as a refinement on top of EPNP's 5-point solution.
    ok_refine, rvec_refined, tvec_refined = cv2.solvePnP(
        _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if ok_refine:
        rvec, tvec = rvec_refined, tvec_refined

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rotation_matrix, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    # euler_angles is a 3x1 array of [pitch, yaw, roll] (degrees) for this
    # decomposition convention.
    yaw = float(euler_angles[1][0])
    return yaw


def _eye_openness_ratio(landmarks_106, kps):
    """
    Approximate an EAR-like eye-openness score using the 106-point landmark
    model, WITHOUT hardcoding a fixed eye-contour index range. Instead, we
    geometrically cluster whichever of the 106 points fall in a small
    ellipse around each of the detector's 5-point eye centers (kps[0] /
    kps[1], a stable and well-documented ordering), then use that cluster's
    bounding box as a stand-in for the eye contour. This is deliberately
    robust to the 106-point index scheme varying slightly between
    insightface package versions, since it never assumes specific indices.

    The ellipse is wider than it is tall (relative to inter-ocular
    distance) so it captures the eye itself without pulling in eyebrow
    points sitting just above it, which would otherwise corrupt the
    "height" measurement.

    Returns the average left/right ratio, or None if it can't be computed.
    """
    if landmarks_106 is None or kps is None:
        return None

    left_eye_center = np.asarray(kps[0], dtype=np.float64)
    right_eye_center = np.asarray(kps[1], dtype=np.float64)
    iod = np.linalg.norm(left_eye_center - right_eye_center)
    if iod <= 0:
        return None

    pts = np.asarray(landmarks_106, dtype=np.float64)
    x_radius = iod * 0.22
    y_radius = iod * 0.14

    def cluster_ratio(center):
        dx = (pts[:, 0] - center[0]) / x_radius
        dy = (pts[:, 1] - center[1]) / y_radius
        mask = (dx ** 2 + dy ** 2) < 1.0
        cluster = pts[mask]
        if cluster.shape[0] < 3:
            return None
        height = cluster[:, 1].max() - cluster[:, 1].min()
        width = cluster[:, 0].max() - cluster[:, 0].min()
        if width <= 0:
            return None
        return height / width

    left_ratio = cluster_ratio(left_eye_center)
    right_ratio = cluster_ratio(right_eye_center)
    ratios = [r for r in (left_ratio, right_ratio) if r is not None]
    if not ratios:
        return None
    return float(np.mean(ratios))


def _count_blinks(ratio_sequence):
    """
    Walk the per-frame eye-openness ratios in order and count Open -> Closed
    -> Open transitions (i.e. completed blinks).
    """
    state = "open"
    blinks = 0
    for r in ratio_sequence:
        if r is None:
            continue
        if state == "open" and r < EAR_CLOSED_THRESHOLD:
            state = "closed"
        elif state == "closed" and r > EAR_OPEN_THRESHOLD:
            state = "open"
            blinks += 1
    return blinks


def _check_challenge(challenge, per_frame_data):
    """per_frame_data: list of dicts with 'yaw' and 'eye_ratio' keys."""
    if challenge == "blink":
        ratios = [f["eye_ratio"] for f in per_frame_data]
        return _count_blinks(ratios) >= 2

    if challenge in ("turn_left", "turn_right"):
        yaws = [f["yaw"] for f in per_frame_data if f["yaw"] is not None]
        if not yaws:
            return False
        wants_negative = (challenge == "turn_left") == YAW_LEFT_IS_NEGATIVE
        if wants_negative:
            return min(yaws) <= -YAW_TURN_THRESHOLD
        else:
            return max(yaws) >= YAW_TURN_THRESHOLD

    return False


def liveness_verification_loop(name_to_match, challenge, frames_data):
    """
    frames_data: a list of ~20-30 base64 data-URL strings captured by the
    browser over ~2-3 seconds while the user performed `challenge`.

    Flow, matching the project's required pipeline:
      1. Decode + run InsightFace on every frame. Detection, 106-point
         landmarks, 5-point kps, and the ArcFace embedding all come from
         the same face_app.get(frame) call already used elsewhere in this
         project.
      2. Verify the requested challenge was actually performed somewhere
         across the sequence (blink count, or head-turn angle).
      3. ONLY if the challenge passes, compare the ArcFace embedding from
         the most front-facing frame in the sequence against the user's
         enrolled embeddings (same cosine-similarity check the old
         single-frame flow used).

    Returns (success: bool, reason: str). reason is one of:
      "no_face"          -- not enough frames had a detectable face
      "challenge_failed" -- face detected fine, but the liveness challenge
                             was not completed
      "no_match"         -- challenge passed but ArcFace didn't match the
                             enrolled user (or the user has no enrollment)
      "ok"                -- passed both checks
    """
    known_face_encodings, known_face_names = load_data()
    matching_indices = [i for i, n in enumerate(known_face_names) if n == name_to_match]
    if not matching_indices:
        return False, "no_match"

    face_app = get_face_app()

    per_frame_data = []
    total_frames = 0

    for data_url in frames_data or []:
        total_frames += 1
        frame = decode_base64_image(data_url)
        if frame is None:
            continue

        faces = face_app.get(frame)
        if not faces:
            continue

        # If multiple faces are in frame, use the largest one (closest to
        # the camera) -- avoids picking up bystanders, same as enrollment.
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        best_face = faces[0]

        kps = getattr(best_face, "kps", None)
        landmarks_106 = getattr(best_face, "landmark_2d_106", None)

        yaw = _estimate_yaw_degrees(kps, frame.shape) if kps is not None else None
        eye_ratio = _eye_openness_ratio(landmarks_106, kps) if landmarks_106 is not None and kps is not None else None

        per_frame_data.append({
            "embedding": best_face.normed_embedding,
            "yaw": yaw,
            "eye_ratio": eye_ratio,
        })

    if total_frames == 0 or (len(per_frame_data) / total_frames) < MIN_DETECTED_FRAME_RATIO:
        return False, "no_face"

    if not _check_challenge(challenge, per_frame_data):
        return False, "challenge_failed"

    # Recognition only runs once the liveness gate has already passed.
    # Use the most front-facing frame (smallest |yaw|) for the ArcFace
    # comparison -- avoids matching against a profile shot, where ArcFace
    # is least reliable. Frames with unknown yaw are treated as frontal.
    def frontal_key(f):
        return abs(f["yaw"]) if f["yaw"] is not None else 0.0

    best_frame = min(per_frame_data, key=frontal_key)
    candidate_embedding = best_frame["embedding"]

    best_similarity = max(
        cosine_similarity(candidate_embedding, known_face_encodings[i])
        for i in matching_indices
    )

    if best_similarity >= MATCH_THRESHOLD:
        return True, "ok"
    return False, "no_match"


def face_recognition_loop(name_to_match, image_data):
    """
    Legacy single-frame verification (no liveness check). Kept for
    reference/backward-compatibility -- the live /verifyface route no
    longer calls this; see liveness_verification_loop above.
    """
    known_face_encodings, known_face_names = load_data()

    matching_indices = [i for i, n in enumerate(known_face_names) if n == name_to_match]
    if not matching_indices:
        return False

    frame = decode_base64_image(image_data)
    if frame is None:
        return False

    face_app = get_face_app()
    faces = face_app.get(frame)
    if not faces:
        return False

    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    candidate_embedding = faces[0].normed_embedding

    best_similarity = max(
        cosine_similarity(candidate_embedding, known_face_encodings[i])
        for i in matching_indices
    )

    return best_similarity >= MATCH_THRESHOLD