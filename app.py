from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_from_directory,
    flash,
    make_response
    
)
from werkzeug.utils import secure_filename
import os
from Ats import *
from functools import wraps
from rag_functions import (
    write_cold_email,
    create_roadmap,
    write_cover_letter,
    write_resume,
    text_to_pdf
    
)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure secret key

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'Data_source'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Dummy user data for authentication
users = {'user1': 'password1', 'user2': 'password2'}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the login page
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Simple authentication check
        if username in users and users[username] == password:
            session['logged_in'] = True
            session['username'] = username  # Store username in session
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid Credentials. Please try again.'
    return render_template('login.html', error=error)

# Route for the dashboard where form submission happens
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    error = None
    if request.method == 'POST':
        # Debug: Print to check form submission
        print("Form submitted")

        # Validate form fields
        if 'resume' not in request.files or 'cover_letter' not in request.files:
            error = 'Please upload both Resume and Cover Letter.'
            print("Validation error: Missing files")
            return render_template('dashboard.html', error=error, username=session['username'])

        # Get form data
        resume = request.files['resume']
        cover_letter = request.files['cover_letter']
        github_link = request.form['github_link']
        aim_job = request.form['aim_job']

        # Validate Aim Job
        if not aim_job:
            error = 'Please provide the link for Your Aim Job.'
            print("Validation error: Missing aim job link")
            return render_template('dashboard.html', error=error, username=session['username'])

        # Save the resume and cover letter locally
        try:
            if resume and allowed_file(resume.filename):
                resume_filename = secure_filename(resume.filename)
                resume.save(os.path.join(app.config['UPLOAD_FOLDER'], resume_filename))
                session['resume_filename'] = resume_filename
                print("Resume saved:", resume_filename)

            if cover_letter and allowed_file(cover_letter.filename):
                cover_letter_filename = secure_filename(cover_letter.filename)
                cover_letter.save(os.path.join(app.config['UPLOAD_FOLDER'], cover_letter_filename))
                session['cover_letter_filename'] = cover_letter_filename
                print("Cover letter saved:", cover_letter_filename)
        except Exception as e:
            error = f"Error saving files: {str(e)}"
            print(error)
            return render_template('dashboard.html', error=error, username=session['username'])

        # Execute the main process using form data
        directory_path = app.config['UPLOAD_FOLDER']
        username = session['username']
        
        try:
            # Execute the main process
            print("Executing main process")
            from application import executeMainProcess  # Ensure this is correctly imported
            executeMainProcess(github_link, directory_path, aim_job, username)
            session['aim_job'] = aim_job  # Save aim_job in session
            print("Main process executed successfully")
        except Exception as e:
            error = f"Error in main process: {str(e)}"
            print(error)
            return render_template('review.html', error=error, username=session['username'])

        # After processing, redirect to the review page
        print("Redirecting to review page")
        return redirect(url_for('review'))

    # Render the dashboard page on GET request
    return render_template('dashboard.html', error=error, username=session['username'])


# Route for the review page after processing is complete
@app.route('/review', methods=['GET', 'POST'])
@login_required
def review():
    ats_improve_text = None
    road_map_text = None
    cold_email = None

    # Render the review page
    return render_template('review.html', ats_improve_text=ats_improve_text, 
                           road_map_text=road_map_text, cold_email=cold_email)

# Route for logging out
@app.route('/logout')
@login_required
def logout():
    session.clear()  # Clear session on logout
    return redirect(url_for('login'))

# Route to download a PDF file
@app.route('/download_new_resume', methods=['POST'])
def download_resume():
    username = session.get('username')
    url = session.get('aim_job')  # Fetch the username from the session
    if not username:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    text = write_resume(username,url)
    print(text)  # Assume text is generated or passed from a form submission
    pdf_data = text_to_pdf(text)

    response = make_response(pdf_data)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=New_Resume.pdf'
    
    return response


@app.route('/download_new_cover_letter', methods=['POST'])
def download_cover():
    username = session.get('username')
    url = session.get('aim_job')
    text = write_cover_letter(username , url) # Assume text is passed from a form submission
    pdf_data = text_to_pdf(text)
    response = make_response(pdf_data)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=New_Cover.pdf'
    
    return response



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
