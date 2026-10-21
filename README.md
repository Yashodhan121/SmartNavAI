
# SmartNav AI
### AI-Based Navigation Assistant for Visually Impaired People

SmartNav AI is a Python-based prototype designed to help visually impaired people identify objects in their surroundings and receive spoken directional guidance.

## Features
- Real-time object detection using YOLO11n
- Camera processing with OpenCV
- Left, centre and right direction estimation
- Voice guidance using pyttsx3
- Flask web interface
- SQLite database support, if enabled in the application

## Technologies Used
- Python
- Flask
- YOLO11n / Ultralytics
- OpenCV
- pyttsx3
- SQLite

## Project Structure
- `app.py` — Main Flask application
- `README.md` — Project documentation
- `requirements.txt` — Python dependencies
- `templates/` — HTML templates, if used
- `static/` — CSS, JavaScript and other static files, if used

## Installation

1. Install Python 3.
2. Clone this repository.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Run the application:

   `python app.py`

5. Open the local URL shown in the terminal.

## Limitations
This is a prototype. Object detection and image-based direction estimates may be inaccurate. It does not guarantee safe navigation or precise distance measurement.

## Future Improvements
- Better obstacle-distance estimation
- Improved voice guidance
- More extensive real-world testing
- Additional accessibility features
