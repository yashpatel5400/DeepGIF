import random
import string
import os
from flask import Flask, request, redirect, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter

UPLOAD_FOLDER = './tmp'
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def file_extension(filename):
    if '.' in filename:
        return filename.rsplit('.', 1)[1].lower()
    else:
        return None

def file_name_generator(size=10, chars=string.ascii_uppercase):
    return ''.join(random.choice(chars) for _ in range(size))

def color_generator():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/_image_segmentation', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        segments = []
        original_image = secure_filename(file.filename)
        extension = file_extension(original_image)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], original_image))
        # do segmentation, save images to temp folder
        for i in xrange(5):
            filename = file_name_generator() + '.' + extension
            image = Image.new("RGB", (400, 400), color_generator())
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            segments.append(filename)
        # return urls to segmented images
        return render_template('_image_segmentation.html', original_image=original_image, segments=segments)

@app.route('/_style_transfer', methods=['POST'])
def style_transfer():
    # parse what the user wants to do to each segment of the image
    d = request.get_json()
    # compute the final image
    # for file, style in d.iteritems():
    filename = file_name_generator() + '.jpg'
    image = Image.new("RGB", (400, 400), color_generator())
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('_style_transfer.html', new_image=filename)