

from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import time
import threading

from database import save_detection
from voice.speaker import speak


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# YOLO MODEL
# ============================================================

model = YOLO("yolo11n.pt")

CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# OBJECT PRIORITY
# ============================================================

OBJECT_PRIORITY = {
    "person": "HIGH",
    "car": "HIGH",
    "truck": "HIGH",
    "bus": "HIGH",
    "motorcycle": "HIGH",
    "bicycle": "HIGH",

    "chair": "MEDIUM",
    "table": "MEDIUM",
    "laptop": "MEDIUM",
    "tv": "MEDIUM",
    "backpack": "MEDIUM",

    "bottle": "LOW",
    "book": "LOW",
    "cell phone": "LOW"
}


# ============================================================
# GLOBAL VARIABLES
# ============================================================

current_detection = {
    "object": "None",
    "direction": "CENTER",
    "proximity": "FAR",
    "confidence": 0,
    "priority": "LOW"
}

latest_frame = None
frame_lock = threading.Lock()


# ============================================================
# DETECTION AND VOICE CONTROL
# ============================================================

active_detection_key = None
last_seen_detection_time = 0

# An object must be absent for longer than this
# before its next appearance counts as a new detection.
DETECTION_GAP_RESET = 2.0

voice_lock = threading.Lock()


def announce_twice(message):
    """Announce each new detection exactly twice."""
    speak(message)
    time.sleep(0.5)
    speak(message)


# ============================================================
# PROXIMITY
# ============================================================

def get_proximity(box_width, frame_width):
    ratio = box_width / frame_width

    if ratio > 0.45:
        return "VERY CLOSE"
    elif ratio > 0.30:
        return "CLOSE"
    elif ratio > 0.15:
        return "MEDIUM"
    else:
        return "FAR"


# ============================================================
# DIRECTION
# ============================================================

def get_direction(x_center, frame_width):
    left_limit = frame_width * 0.33
    right_limit = frame_width * 0.66

    if x_center < left_limit:
        return "LEFT"
    elif x_center > right_limit:
        return "RIGHT"
    else:
        return "CENTER"


# ============================================================
# NAVIGATION INSTRUCTION
# ============================================================

def get_navigation_instruction(object_name, direction, proximity):
    object_name = object_name.capitalize()

    if proximity == "VERY CLOSE":
        if direction == "CENTER":
            return f"{object_name} ahead. Very close. Stop."
        elif direction == "LEFT":
            return f"{object_name} on your left. Very close. Move right."
        elif direction == "RIGHT":
            return f"{object_name} on your right. Very close. Move left."

    elif proximity == "CLOSE":
        if direction == "CENTER":
            return f"{object_name} ahead. Close. Move carefully."
        elif direction == "LEFT":
            return f"{object_name} on your left. Move slightly right."
        elif direction == "RIGHT":
            return f"{object_name} on your right. Move slightly left."

    elif proximity == "MEDIUM":
        if direction == "CENTER":
            return f"{object_name} ahead. Keep moving carefully."
        elif direction == "LEFT":
            return f"{object_name} on your left."
        elif direction == "RIGHT":
            return f"{object_name} on your right."

    elif proximity == "FAR":
        if direction == "CENTER":
            return f"{object_name} ahead. You can continue."
        elif direction == "LEFT":
            return f"{object_name} on your left."
        elif direction == "RIGHT":
            return f"{object_name} on your right."

    return f"{object_name} detected {direction.lower()}."


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(frame):
    global current_detection
    global latest_frame
    global active_detection_key
    global last_seen_detection_time

    frame_height, frame_width = frame.shape[:2]

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model(frame, verbose=False)

    best_detection = None
    best_area = 0

    # --------------------------------------------------------
    # FIND LARGEST VALID OBJECT
    # --------------------------------------------------------

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            box_width = x2 - x1
            box_height = y2 - y1
            area = box_width * box_height

            if area > best_area:
                class_id = int(box.cls[0])
                object_name = model.names[class_id]

                best_detection = {
                    "object": object_name,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": box_width,
                    "height": box_height
                }

                best_area = area

    # ========================================================
    # OBJECT DETECTED
    # ========================================================

    if best_detection:
        object_name = best_detection["object"]
        confidence = best_detection["confidence"]

        x1 = best_detection["x1"]
        y1 = best_detection["y1"]
        x2 = best_detection["x2"]
        y2 = best_detection["y2"]

        box_width = best_detection["width"]

        # Object center
        x_center = (x1 + x2) / 2

        # Direction
        direction = get_direction(x_center, frame_width)

        # Proximity
        proximity = get_proximity(box_width, frame_width)

        # Priority
        priority = OBJECT_PRIORITY.get(object_name, "MEDIUM")

        # Update current detection for the webpage
        current_detection = {
            "object": object_name,
            "direction": direction,
            "proximity": proximity,
            "confidence": round(confidence, 2),
            "priority": priority
        }

        # ----------------------------------------------------
        # SAVE ONCE AND ANNOUNCE TWICE
        # ----------------------------------------------------

        current_time = time.time()

        # Identify an ongoing detection by object name.
        # Changes in direction or proximity alone will not
        # create extra history records or repeat the voice.
        detection_key = object_name

        with voice_lock:
            is_new_detection = (
                detection_key != active_detection_key
                or
                current_time - last_seen_detection_time
                > DETECTION_GAP_RESET
            )

            if is_new_detection:
                active_detection_key = detection_key

                # Save exactly one record when this detection
                # starts, instead of saving on every frame.
                save_detection(
                    object_name,
                    direction,
                    proximity,
                    confidence
                )

                # Start the two announcements in a background
                # thread so camera processing can continue.
                message = get_navigation_instruction(
                    object_name,
                    direction,
                    proximity
                )

                threading.Thread(
                    target=announce_twice,
                    args=(message,),
                    daemon=True
                ).start()

            # Refresh while the object remains detected.
            last_seen_detection_time = current_time

        # ----------------------------------------------------
        # DRAW BOUNDING BOX
        # ----------------------------------------------------

        label = (
            f"{object_name} "
            f"{confidence:.2f} "
            f"{direction} "
            f"{proximity}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ========================================================
    # NO OBJECT DETECTED
    # ========================================================

    else:
        current_detection = {
            "object": "None",
            "direction": "CENTER",
            "proximity": "FAR",
            "confidence": 0,
            "priority": "LOW"
        }

        # Do not immediately reset the active detection.
        # This prevents brief camera/YOLO misses from causing
        # repeated announcements. The 2-second gap handles it.

    # --------------------------------------------------------
    # STORE LATEST FRAME
    # --------------------------------------------------------

    with frame_lock:
        latest_frame = frame.copy()

    return frame


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():
    response = app.make_response(render_template("index.html"))

    response.headers["Cache-Control"] = (
        "no-cache, no-store, must-revalidate"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


# ============================================================
# PROCESS PHONE FRAME
# ============================================================

@app.route("/process_frame", methods=["POST"])
def process_phone_frame():
    try:
        # Raw JPEG data
        if request.data:
            image_data = request.data

        # Multipart frame
        elif "frame" in request.files:
            image_data = request.files["frame"].read()

        else:
            return jsonify({
                "error": "No frame received"
            }), 400

        # Decode image
        np_array = np.frombuffer(image_data, np.uint8)

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({
                "error": "Could not decode frame"
            }), 400

        # Process frame
        process_frame(frame)

        # Return detection
        return jsonify({
            "status": "success",
            "detection": current_detection
        })

    except Exception as e:
        print("Frame processing error:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# VIDEO STREAM GENERATOR
# ============================================================

def generate_frames():
    global latest_frame

    while True:
        with frame_lock:
            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# CURRENT DETECTION API
# ============================================================

@app.route("/current_detection")
def current_detection_api():
    return jsonify(current_detection)


# ============================================================
# DETECTION HISTORY API
# ============================================================

@app.route("/history")
def history():
    try:
        from database import get_history

        history_data = get_history()

        return jsonify(history_data)

    except Exception as e:
        print("History error:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":
    print()
    print("========================================")
    print("        SMARTNAV AI STARTED")
    print("========================================")
    print()

    print("Laptop:")
    print("https://10.150.45.183:5000/?device=laptop")
    print()

    print("Phone:")
    print("https://10.150.45.183:5000/?device=phone")
    print()

    print("========================================")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        ssl_context=(
            "cert/server.crt",
            "cert/server.key"
        )
    )
