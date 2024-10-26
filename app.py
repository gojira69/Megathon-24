from flask import Flask, render_template, request, redirect, url_for
import os
import subprocess

app = Flask(__name__)

# Base directory to save uploaded files
BASE_UPLOAD_FOLDER = 'Data/'
os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)  # Create the base directory if it doesn't exist

@app.route('/')
def landing_page():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    patient_name = request.args.get('name')  # Get the patient name from URL
    return render_template('dashboard.html', patient_name=patient_name)

@app.route('/upload', methods=['POST'])
def upload():
    # Extract the patient name from the URL
    patient_name = request.args.get('name')  # Get the patient name from the query string
    upload_folder = os.path.join(BASE_UPLOAD_FOLDER, patient_name)
    
    # Create a subdirectory for the patient
    os.makedirs(upload_folder, exist_ok=True)

    therapist_notes = request.form.get('therapist_notes')
    audio_file = request.files.get('audio_file')
    music_file = request.files.get('music_file')

    # Save therapist notes as a text file
    if therapist_notes:
        with open(os.path.join(upload_folder, 'therapist_notes.txt'), 'w') as f:
            f.write(therapist_notes)

    # Save uploaded audio file
    if audio_file:
        audio_file.save(os.path.join(upload_folder, audio_file.filename))

    # Save uploaded music file
    if music_file:
        music_file.save(os.path.join(upload_folder, music_file.filename))

    # Run the subprocess to generate CSV files
    try:
        subprocess.run(['python3', 'script.py', patient_name], check=True)
        print('done bro')
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the subprocess: {e}")


    return redirect(url_for('dashboard', name=patient_name))  # Redirect back to the dashboard

if __name__ == '__main__':
    app.run(debug=True)
