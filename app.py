import os
from flask import Flask, request, jsonify, render_template, send_file
from faster_whisper import WhisperModel
from fpdf import FPDF
import google.generativeai as genai
import cv2
import numpy as np
import base64
import mediapipe as mp
from io import BytesIO
from PIL import Image
from threading import Thread
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles


model_size = "distil-large-v3"
model = WhisperModel(model_size, device="cpu", compute_type="int8")


transcriptions = []
gemini_results = []


@app.route('/')
def index():
    questions = [
        "Tell me about yourself.",
        "What are your strengths?",
        "What is one of your weaknesses?",
        "Why do you want to work here?"]
    return render_template('hom.html', questions=questions)


def detect_eye_direction(landmarks, image_width, image_height):
    left_eye_indices = [33, 133, 159, 145]  # Left eye landmarks
    left_iris_indices = [468, 469, 470, 471]  # Left iris landmarks

    eye_coords = [(int(landmarks[i].x * image_width), int(landmarks[i].y * image_height)) for i in left_eye_indices]
    iris_coords = [(int(landmarks[i].x * image_width), int(landmarks[i].y * image_height)) for i in left_iris_indices]

    iris_center = np.mean(iris_coords, axis=0).astype(int)

    left_corner, right_corner = eye_coords[0], eye_coords[1]
    top_point, bottom_point = eye_coords[2], eye_coords[3]

    horizontal_center = (left_corner[0] + right_corner[0]) // 2
    vertical_center = (top_point[1] + bottom_point[1]) // 2

    horizontal_threshold = (right_corner[0] - left_corner[0]) * 0.09
    vertical_threshold = (bottom_point[1] - top_point[1]) * 0.25  # Adjusted for better sensitivity

    direction = "Straight"
    if iris_center[0] < horizontal_center - horizontal_threshold:
        direction = "Left"
    elif iris_center[0] > horizontal_center + horizontal_threshold:
        direction = "Right"
    elif iris_center[1] < vertical_center - vertical_threshold:
        direction = "Up"
    elif iris_center[1] > vertical_center + vertical_threshold:
        direction = "Down"

    # Refine "Straight" detection
    if (
        abs(iris_center[0] - horizontal_center) <= horizontal_threshold * 0.5 and
        abs(iris_center[1] - vertical_center) <= vertical_threshold * 0.5
    ):
        direction = "Straight"

    return direction


def process_frame(image):
    # Convert the image to OpenCV format
    np_image = np.array(image)
    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = np_image.shape
    
    # Process the frame with MediaPipe
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        # Process the frame
        results = face_mesh.process(np_image)
        np_image.flags.writeable = True
    
    # Draw landmarks if detected and calculate eye direction
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=np_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=np_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks(
                image=np_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            ) 
            # Detect eye direction for the first face
        processed_frame = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    return processed_frame


@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    # Decode base64 image
    image_data = data['image'].split(",")[1]
    decoded_image = base64.b64decode(image_data)
    pil_image = Image.open(BytesIO(decoded_image))
    
    # Process frame and get eye direction
    processed_image = process_frame(pil_image)
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "processed_frame": f"data:image/jpeg;base64,{processed_image_base64}",
    })


@app.route('/questions')
def questions():
    # List of questions to render
    questions = [
        "What is your name?",
        "How old are you?",
        "Where are you from?",
        "What are your hobbies?"
    ]
    return render_template('questions.html', questions=questions)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    global transcriptions
    try:
        # Step 1: Get the uploaded file
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400

        # Step 2: Save the file
        audio_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(audio_path)

        # Step 3: Perform transcription
        try:
            # Run the transcription process
            result = model.transcribe(
                audio_path, beam_size=1, language="en", condition_on_previous_text=False
            )

            # Validate and unpack the result
            if isinstance(result, tuple) and len(result) >= 2:
                segments, info = result
            else:
                raise ValueError("Unexpected output format from model.transcribe.")

            # Step 4: Process the transcription result
            transcription = " ".join([segment.text for segment in segments])
            print("Transcription successful:", transcription)
            transcriptions.append(transcription)

            # Step 5: Send the transcription immediately to the frontend
            response = jsonify({"transcription": transcription})

            # Start a background thread for additional processing
            Thread(
                target=process_transcription_async,
                args=(transcription, audio_path),
                daemon=True
            ).start()

            return response

        except Exception as e:
            transcription = f"Transcription Failed: {e}"
            print(transcription)

        finally:
            # Clean up the uploaded file
            if os.path.exists(audio_path):
                os.remove(audio_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

    return jsonify({"transcription": transcription})


def process_transcription_async(transcription, audio_path):
    """
    Background processing function for transcription.
    """
    try:
        # Get predefined answer
        predefined_response = get_predefined_answer(transcription)

        # Send transcription and response to Gemini
        send_to_gemini(transcription, predefined_response)

    except Exception as e:
        print(f"Error in background processing: {e}")

    finally:
        # Ensure cleanup if needed
        if os.path.exists(audio_path):
            os.remove(audio_path)


def get_predefined_answer(transcription):
    # Simple matching based on keywords
    predefined_answers = [
        "Tell me about yourself.",
        "What are your strengths?",
        "What is one of your weaknesses?",
        "Why do you want to work here?"]

    for phrase in predefined_answers:
        if phrase.lower() in transcription.lower():
            return phrase

    return "Sorry, I didn't understand that."

gemini_api_key = os.getenv('gemini_api_key')
def send_to_gemini(transcription, response):
    global gemini_results
    genai.configure(api_key= gemini_api_key)
    mode = genai.GenerativeModel("gemini-pro")

    # Define the prompt for Gemini
    prompt = f"User's answer: {transcription}\nGiven question: {response}\nPlease rate the user's answer to the given question on a scale from 1 to 10. Then, summarize the user's answer in 5 to 6 lines."
    response = mode.generate_content(prompt)
    print(prompt)
    gemini_results.append(response.text)
    print(response.text)
    return 


@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_right_margin(20)
    pdf.set_left_margin(20)
    pdf.set_auto_page_break(auto=True, margin=10)
    # Assuming transcriptions and gemini_results are lists of transcription text and JSON objects, respectively
    for i, (transcription, gemini_result) in enumerate(zip(transcriptions, gemini_results), start=1):
        # Add transcription to PDF
        pdf.multi_cell(
            0, 10, txt=f"Transcription {i}: {transcription}", border=1)

        pdf.multi_cell(
            0, 10, txt=f"Gemini Result {i}: {gemini_result}", border=1)
        pdf.cell(0, 10, txt="", ln=True)  # Blank line for spacing

    pdf_output_path = "transcriptions.pdf"
    pdf.output(pdf_output_path)

    return send_file(pdf_output_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
