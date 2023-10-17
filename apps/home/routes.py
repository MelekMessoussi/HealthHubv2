# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request , jsonify
from flask_login import login_required
from jinja2 import TemplateNotFound
from flask import Flask, render_template, request, send_file
import requests
import io
import PIL
from PIL import Image
from PIL import UnidentifiedImageError  # Import the UnidentifiedImageError class
import requests
import io
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import sys
import pickle
import torch
from typing import Tuple
from io import BytesIO
import base64
import plotly
import plotly.graph_objects as go
import numpy as np
import random

import nibabel as nib
import os
import albumentations as A
import numpy as np
import plotly.graph_objects as go
import numpy as np

from subjective import SubjectiveTest
from objective import ObjectiveTest
import nltk
#nltk.download("all")

# os.system("git clone https://github.com/NVlabs/stylegan3")

sys.path.append("stylegan3")
# Load the StyleGAN model
network_pkl = "brainmrigan.pkl"

with open(network_pkl, 'rb') as f:
    G = pickle.load(f)['G_ema']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G.eval()
G.to(device)





API_URL = "https://api-inference.huggingface.co/models/yahyasmt/brain-tumor-3"
headers = {"Authorization": "Bearer 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"}


app = Flask(__name__)

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@blueprint.route('/')
def route_default():
    return render_template('index.html', segment='index')

@blueprint.route('/index',methods=['GET', 'POST'])

def index():
    return render_template('index.html', segment='index')

@blueprint.route('/dashboard',methods=['GET', 'POST'])
@login_required
def Dashboard():
    return render_template('/dashboard/index.html')
# List of prompts
prompts = [
    "brain tumor mri",
    "tumor mri",
    "brain tumor mri scan",
    "scan of a brain tumor mri",
    "scan of a brain tumor",
    "brain tumor",
    # Add more prompts as needed
]

#########################################################################################
@blueprint.route('/MRI_Generation',methods=['GET', 'POST'])
def indeximage():
    if request.method == "POST":
        # Select a random prompt from the list
        prompt = random.choice(prompts)
        print(prompt)
        image_bytes = query({
            "inputs": prompt,
            "options": {"wait_for_model": True }
        })
        with open("./apps/static/assets_old/mdl/aa.jpeg", "wb") as image_file:
            image_file.write(image_bytes)

    return render_template('/dashboard/MRI_Generation.html')
########################################################################################
@blueprint.route('/predict', methods=['POST'])
def get_prediction():
    try:
        # Generate random Seed
        seed = random.randint(0, 65536)
        # Choose a random noise mode from the list
        noise_mode = "none"
        # Generate a random truncation_psi 
        truncation_psi = random.uniform(0.1, 2.0)
        result_image = predict(seed, noise_mode, truncation_psi)
        
        # Convert the Image to a base64-encoded string
        result_image_base64 = image_to_base64(result_image)
        
        return jsonify({'result_image': result_image_base64})
    except ValueError:
        # Handle the case where 'Seed' is not a valid integer
        return jsonify({'error': 'Invalid Seed value'}), 400

# Function to convert PIL Image to base64-encoded string
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # You can change the format if needed
    return base64.b64encode(buffered.getvalue()).decode()

def predict(Seed, noise_mode, truncation_psi):
    # Generate images.
    z = torch.from_numpy(np.random.RandomState(Seed).randn(1, G.z_dim)).to(device)
    label = torch.zeros([1, G.c_dim], device=device)
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return (PIL.Image.fromarray(img[0].cpu().numpy()[:, :, 0])).resize((512, 512))





#########################################################################################


# Process the uploaded image and display the 3D plot
@blueprint.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Load your image (color or grayscale)
        image = plt.imread(filename)

        # Check if it's a color image
        is_color_image = len(image.shape) == 3 and image.shape[2] in [3, 4]

        # Convert color image to grayscale if needed
        if is_color_image:
            gray_image = np.mean(image, axis=2)  # Convert to grayscale
        else:
            gray_image = image

        # Normalize the image to the range [0, 1]
        gray_image = gray_image / np.max(gray_image)

        # Apply a Gaussian filter to smooth the image (adjust sigma for smoothing effect)
        sigma = 2.0
        smoothed_image = gaussian_filter(gray_image, sigma=sigma)

        # Scale the smoothed image to control the elevation magnitude
        elevation_scale = 50.0  # Adjust this value as needed
        elevated_image = smoothed_image * elevation_scale

        # Create a grid of X and Y coordinates
        x, y = np.meshgrid(np.arange(gray_image.shape[1]), np.arange(gray_image.shape[0]))

        # Create the Z coordinates using the elevated image
        z = elevated_image

        # Create a 3D surface plot of the inflated image using Plotly
        fig = go.Figure(data=[go.Surface(z=z, colorscale='Viridis')])
        fig.update_layout(scene=dict(aspectmode="data"))
        plot_div = fig.to_html(full_html=False)

        return render_template('dashboard/result.html', plot_div=plot_div)

    return redirect(request.url)

# Serve uploaded files
@blueprint.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@blueprint.route('/3D')
@login_required
def ThreeD():
    return render_template('dashboard/3D.html')


@blueprint.route('/chat')
# @login_required
def chat():
    return render_template('home/chat.html')


####################################################################################
UPLOAD_FOLDER2 = 'nii'
ALLOWED_EXTENSIONS2 = {'gz'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER2
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

def allowed_file2(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS2

class ImageReader:
    def __init__(
        self, root: str, file_path2: str = '', img_size: int = 256,
        normalize: bool = False, single_class: bool = False
    ) -> None:
        pad_size = 256 if img_size > 256 else 224
        self.resize = A.Compose(
            [
                A.PadIfNeeded(min_height=pad_size, min_width=pad_size, value=0),
                A.Resize(img_size, img_size)
            ]
        )
        self.normalize = normalize
        self.single_class = single_class
        self.root = root
        self.file_path2 = file_path2

    def read_file(self, path: str) -> dict:
        scan_type = path.split('_')[-1]
        raw_image = nib.load(path).get_fdata()
        raw_mask = nib.load(path.replace(scan_type, 'seg.nii.gz')).get_fdata()
        processed_frames, processed_masks = [], []
        for frame_idx in range(raw_image.shape[2]):
            frame = raw_image[:, :, frame_idx]
            mask = raw_mask[:, :, frame_idx]
            resized = self.resize(image=frame, mask=mask)
            processed_frames.append(resized['image'])
            processed_masks.append(
                1 * (resized['mask'] > 0) if self.single_class else resized['mask']
            )
        scan_data = np.stack(processed_frames, 0)
        if self.normalize:
            if scan_data.max() > 0:
                scan_data = scan_data / scan_data.max()
            scan_data = scan_data.astype(np.float32)
        return {
            'scan': scan_data,
            'segmentation': np.stack(processed_masks, 0),
            'orig_shape': raw_image.shape
        }

    def load_patient_scan(self, idx: int, scan_type: str = 'flair') -> dict:
        patient_id = str(idx).zfill(5)
        scan_filename = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], self.file_path2))
        return self.read_file(scan_filename)
def generate_3d_scatter(
    x:np.array, y:np.array, z:np.array, colors:np.array,
    size:int=3, opacity:float=0.2, scale:str='Teal',
    hover:str='skip', name:str='MRI'
) -> go.Scatter3d:
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers', hoverinfo=hover,
        marker = dict(
            size=size, opacity=opacity,
            color=colors, colorscale=scale
        ),
        name=name
    )


class ImageViewer3d():
    def __init__(
        self, reader:ImageReader,
        mri_downsample:int=10, mri_colorscale:str='Ice'
    ) -> None:
        self.reader = reader
        self.mri_downsample = mri_downsample
        self.mri_colorscale = mri_colorscale

    def load_clean_mri(self, image:np.array, orig_dim:int) -> dict:
        shape_offset = image.shape[1]/orig_dim
        z, x, y = (image > 0).nonzero()
        # only (1/mri_downsample) is sampled for the resulting image
        x, y, z = x[::self.mri_downsample], y[::self.mri_downsample], z[::self.mri_downsample]
        colors = image[z, x, y]
        return dict(x=x/shape_offset, y=y/shape_offset, z=z, colors=colors)

    def load_tumor_segmentation(self, image:np.array, orig_dim:int) -> dict:
        tumors = {}
        shape_offset = image.shape[1]/orig_dim
        # 1/1, 1/3 and 1/5 pixels for tumor tissue classes 1(core), 2(invaded) and 4(enhancing)
        sampling = {
            1: 1, 2: 3, 4: 5
        }
        for class_idx in sampling:
            z, x, y = (image == class_idx).nonzero()
            x, y, z = x[::sampling[class_idx]], y[::sampling[class_idx]], z[::sampling[class_idx]]
            tumors[class_idx] = dict(
                x=x/shape_offset, y=y/shape_offset, z=z,
                colors=class_idx/4
            )
        return tumors

    def collect_patient_data(self, scan:dict) -> tuple:
        clean_mri = self.load_clean_mri(scan['scan'], scan['orig_shape'][0])
        tumors = self.load_tumor_segmentation(scan['segmentation'], scan['orig_shape'][0])
        markers_created = clean_mri['x'].shape[0] + sum(tumors[class_idx]['x'].shape[0] for class_idx in tumors)
        return [
            generate_3d_scatter(
                **clean_mri, scale=self.mri_colorscale, opacity=0.4,
                hover='skip', name='Brain MRI'
            ),
            generate_3d_scatter(
                **tumors[1], opacity=0.8,
                hover='all', name='Necrotic tumor core'
            ),
            generate_3d_scatter(
                **tumors[2], opacity=0.4,
                hover='all', name='Peritumoral invaded tissue'
            ),
            generate_3d_scatter(
                **tumors[4], opacity=0.4,
                hover='all', name='GD-enhancing tumor'
            ),
        ], markers_created

    def get_3d_scan(self, patient_idx:int, scan_type:str='flair') -> go.Figure:
        scan = self.reader.load_patient_scan(patient_idx, scan_type)
        data, num_markers = self.collect_patient_data(scan)
        fig = go.Figure(data=data)
        fig.update_layout(
            title=f"[Patient id:{patient_idx}] brain MRI scan ({num_markers} points)",
            legend_title="Pixel class (click to enable/disable)",
            font=dict(
                family="Courier New, monospace",
                size=14,
            ),
            margin=dict(
                l=0, r=0, b=0, t=30
            ),
            legend=dict(itemsizing='constant')
        )
        return fig

@blueprint.route('/3dtotal', methods=['GET', 'POST'])
def mrithreed():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file2(file.filename):
            # Save the uploaded file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            reader = ImageReader('./data', file_path2=file.filename, img_size=128, normalize=True, single_class=False)
            viewer = ImageViewer3d(reader, mri_downsample=20)

            # Generate the 3D scan based on user input
            fig = viewer.get_3d_scan(0, 't1')

            # Render the HTML template with the generated 3D scan
            return render_template('dashboard/resulttotal3d.html', plot=plotly.offline.plot(fig, output_type='div'))

    # Render the HTML form when the page is first loaded
    return render_template('dashboard/resulttotal3d.html', plot=None)
############################################################################################
@blueprint.route('/quiz')
@login_required
def quizdef():
    return render_template('dashboard/quizindex.html')

@blueprint.route('/test_generate', methods=['POST'])
def test_generate():
    itext = request.form.get('itext', '')
    test_type = request.form.get('test_type', 'objective')
    noq = request.form.get('noq', 1)
    
    if test_type == 'objective':
        objective_generator = ObjectiveTest(itext, noq)
        questions, correct_answers, _, _ = objective_generator.generate_test()
        question_answer_pairs = list(zip(questions, correct_answers))
        return render_template('dashboard/quiz.html', questions=questions, correct_answers=correct_answers, question_answer_pairs=question_answer_pairs)
    elif test_type == 'subjective':
        subjective_generator = SubjectiveTest(itext, noq)
        questions, correct_answers = subjective_generator.generate_test()
        question_answer_pairs = list(zip(questions, correct_answers))
        return render_template('dashboard/quiz_type2.html', questions=questions, correct_answers=correct_answers)

@blueprint.route('/submit_answers', methods=['POST'])
def submit_answers():
    # Retrieve the correct answers based on the test type
    test_type = request.form.get('test_type', 'objective')
    if test_type == 'objective':
        correct_answers = request.form.getlist('correct_answers')
        total_questions = len(correct_answers)
    elif test_type == 'subjective':
        total_questions = int(request.form.get('total_questions'))
        correct_answers = [request.form.get(f'correct_answer_{i}') for i in range(total_questions) if f'correct_answer_{i}' in request.form]

    # Retrieve user answers and questions
    user_answers = [request.form.get(f'user_answer_{i}') for i in range(total_questions)]
    user_questions = [request.form.get(f'question_{i}') for i in range(total_questions)]

    # Calculate the score
    score = sum(user_answer == correct_answer for user_answer, correct_answer in zip(user_answers, correct_answers))

    # Pass the score, total questions, correct answers, and user questions to the template
    return render_template('dashboard/score.html', score=score, total_questions=total_questions, correct_answers=correct_answers, user_questions=user_questions, user_answers=user_answers)

############################################################################################
############################################################################################

API_URL2 = "https://api-inference.huggingface.co/models/amjadfqs/swin-base-patch4-window7-224-in22k-finetuned-brain-tumor-final_13"
headers2 = {"Authorization": "Bearer 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"}

API_URL3 = "https://api-inference.huggingface.co/models/Locutusque/gpt2-large-medical"
headers3 = {"Authorization": "Bearer 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"}

def query3(payload):
	response = requests.post(API_URL3, headers=headers3, json=payload)
	return response.json()
	
def query2(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL2, headers=headers2, data=data)
    return response.json()

# output = query2("cats.jpg")

# Your Flask route to handle file upload and JSON response
import time
@blueprint.route('/decttumor', methods=['GET', 'POST'])
def decttumor():
    tumor = None
    tumor2 = "Here we propose a first piste of reflection about the tumors ..."
    zero = "Name of the tumor"
    chart_image_path = None  # Initialize the chart image path as None

    if request.method == "POST":
        retries = 100
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            while retries > 0:
                response = query2(filename)
                if extract_predicted_tumor(response) == "Error extracting predicted tumor":
                    retries -= 1
                else:
                    zero = extract_predicted_tumor(response)
                    one = "what is brain tumor " + zero
                    tumor = query3({"inputs": extract_predicted_tumor(response), "options": {"wait_for_model": True}}, )
                    tumor2 = tumor[0]['generated_text']
                    chart_image = create_chart(response)

        
                    # Save the chart image as a PNG file
                    with open("./apps/static/assets_old/mdl/chart.png", "wb") as image_file:
                        image_file.write(chart_image)

                    break
        else:
            return "File format not allowed"

    # Provide the chart image path for downloading
    return render_template('dashboard/dector.html',tumor=tumor2, name=zero)

def extract_predicted_tumor(response):
    try:
        # Assuming that the predicted tumor label is the one with the highest score
        tumor_with_highest_score = max(response, key=lambda x: x["score"])
        predicted_tumor = tumor_with_highest_score["label"]
        return predicted_tumor
    except Exception as e:
        # Handle any exceptions that may occur during extraction
        return "Error extracting predicted tumor"
import seaborn as sns

def create_chart(response):
    # Extract data from the JSON response
    labels = [entry['label'] for entry in response]
    scores = [entry['score'] for entry in response]

    # Create a bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(labels, scores)
    plt.xlabel('Label')
    plt.ylabel('Score')

    # Save the chart as an image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Convert the image to base64 for embedding in HTML
    chart_image = img.read()

    return chart_image
############################################################################################
@blueprint.route('/MRISEG',methods=['GET', 'POST'])
# @login_required
def MRISEG():
    return render_template('/MRISEG/index.html')

pixel_dtypes = {"int16": np.int16,"float64": np.float64}
IMG_SIZE=128


import SimpleITK as sitk
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut 
import sys
import time
import os
import numpy as np
import cv2
import nibabel as nib
import scipy.ndimage
from skimage import measure
import glob
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
from keras.models import Model
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut 
import os
from keras.layers import Input
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from scipy.spatial import ConvexHull
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.graph_objs import *
import plotly
import plotly.graph_objects as go

import h5py

import nibabel as nib
import SimpleITK as sitk
import sys
import time
import os
import numpy as np
import cv2

import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

 
writer = sitk.ImageFileWriter()
# Use the study/series/frame of reference information given in the meta-data
# dictionary and not the automatically generated information from the file IO
writer.KeepOriginalImageUIDOn()



def writeSlices(series_tag_values, new_img, out_dir, i):
    image_slice = new_img[:, :, i]

    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0],
                                                       tag_value[1]),
             series_tag_values))

    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))

    # Setting the type to CT so that the slice location is presC:\Users\dell\Downloads\myownproject\BraTS19_2013_10_1_flair.niierved and
    # the thickness is carried over.
    image_slice.SetMetaData("0008|0060", "CT")

    # (0020, 0032) image position patient determines the 3D spacing between
    # slices.
    #   Image Position (Patient)
    image_slice.SetMetaData("0020|0032", '\\'.join(
        map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))))
    #   Instance Number
    image_slice.SetMetaData("0020,0013", str(i))
# Write to the output directory and add the extension dcm, to force
    # writing in DICOM format.
    writer.SetFileName(os.path.join(out_dir, str(i) + '.dcm'))
    writer.Execute(image_slice)

# Create a new series from a numpy array
try:
    pixel_dtype = pixel_dtypes["float64"]
except KeyError:
    pixel_dtype = pixel_dtypes["int16"]





if pixel_dtype == np.float64:
    def tag_func(i,new_img):
        rescale_slope = 0.01  # keep three digits after the decimal point
        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")

        direction = new_img.GetDirection()
        series_tag_values = [
        ("0008|0031", modification_time),  # Series Time
        ("0008|0021", modification_date),  # Series Date
        ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
        ("0020|000e", "1.2.826.0.1.3680043.2.1125."
         + modification_date + ".1" + modification_time),  # Series Instance UID
        ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],
                                          direction[1], direction[4],
                                          direction[7])))),  # Image Orientation
        # (Patient)
        ("0008|103e", "Created-SimpleITK")  # Series Description
    ]
        series_tag_values = series_tag_values + [
            ('0028|1053', str(rescale_slope)),  # rescale slope
            ('0028|1052', '0'),  # rescale intercept
            ('0028|0100', '16'),  # bits allocated
            ('0028|0101', '16'),  # bits stored
            ('0028|0102', '15'),  # high bit
            ('0028|0103', '1'), # pixel representation
            ('0020|0013', str(i))] #Instance Number
        return series_tag_values
    







def resample(image, new_spacing=[1,1,1]):
    spacing = np.array([1,1,1])


    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image




def build_unet(inputs, ker_init, dropout):
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv1)
    
    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool)
    conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv3)
    
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv5)
    drop5 = Dropout(dropout)(conv5)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(drop5))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv7)
    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv9)
    
    up = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv9))
    merge = concatenate([conv1,up], axis = 3)
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge)
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv)
    
    conv10 = Conv2D(4, (1,1), activation = 'softmax')(conv)
    
    return Model(inputs = inputs, outputs = conv10)





def load_scan(paths):
    slices = [pydicom.read_file(path ) for path in paths]
    slices.sort(key = lambda x: int(x.InstanceNumber), reverse = False)
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
brain_paths=glob('./apps/brain_dicom/*.dcm')
brains = load_scan(brain_paths)

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)




def find_volume(Data,):
    Volume=0
    for i in range(Data.shape[0]):
        for j in range (Data.shape[1]):
            Area = 0.1 * brains[i].PixelSpacing[0] * 0.1 * brains[i].PixelSpacing[1] * np.count_nonzero(Data[i,j])
            Volume += 0.1 * brains[i].SliceThickness * Area
    return Volume




def sample_stack(stack, rows=4, cols=4, start_with=20, show_every=4):
    fig,ax = plt.subplots(rows,cols,figsize=[9,10])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title(f'slice {ind}')
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    # plt.show()


def sample_stack(stack, rows=4, cols=4, start_with=20, show_every=5):
    fig,ax = plt.subplots(rows,cols,figsize=[9,10])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title(f'slice {ind}')
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    # plt.show()





def make_mesh(image,istransp='true', threshold=-300, step_size=1):

    print("Transposing surface")
    p = image.transpose(2,1,0)
    
    print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
    
    return verts, faces

import plotly.offline as pyo

def plotly_3d(verts, faces, verts2, faces2):
    x, y, z = zip(*verts) 
    x2, y2, z2 = zip(*verts2)
       
    # Make the colormap single color since the axes are positional not intensity. 
    colormap = ['rgba(255, 0, 0, 0.5)', 'rgba(255, 0, 0, 0.5)']
    
    fig = ff.create_trisurf(x=x, y=y, z=z, plot_edges=False,
                            colormap=colormap,
                            show_colorbar=False,
                            simplices=faces,
                            backgroundcolor='rgb(64, 64, 64)',
                            title="Interactive Visualization")
    
    colormap2 = ['rgba(0, 255, 0, 0.8)', 'rgba(0, 255, 0, 0.8)']
    
    fig2 = ff.create_trisurf(x=x2, y=y2, z=z2, plot_edges=False,
                            colormap=colormap2,
                            show_colorbar=False,
                            simplices=faces2,
                            backgroundcolor='rgb(64, 64, 64)',
                            title="Interactive Visualization")
    fig['data'][0].update(opacity=0.2)
    
    data = [fig.data[0], fig2.data[0]]
    
    fig3 = dict(data=data)
    
    return fig3



@blueprint.route('/process', methods=['POST'])
def process_images():
    if request.method == 'POST':
                if 'flair_image' not in request.files or 't1ce_image' not in request.files:
                    return redirect(request.url)

                # Get the uploaded files
                flair_image = request.files['flair_image']
                t1ce_image = request.files['t1ce_image']

                # Check if files have a valid file name
                if flair_image.filename == '' or t1ce_image.filename == '':
                    return redirect(request.url)

                # Save the uploaded files to the UPLOAD_FOLDER
                flair_filename = os.path.join(app.config['UPLOAD_FOLDER'], flair_image.filename)
                t1ce_filename = os.path.join(app.config['UPLOAD_FOLDER'], t1ce_image.filename)
                flair_image.save(flair_filename)
                t1ce_image.save(t1ce_filename)

                new_img = sitk.ReadImage(flair_filename)
                # Write slices to output directory
                list(map(lambda i: writeSlices(tag_func(i,new_img), new_img, './apps/brain_dicom', i),
                        range(new_img.GetDepth())))
                print('Done')

                flair=nib.load(flair_filename).get_fdata()
                print(flair.shape)
                h = flair.shape[0]/128
                c = flair.shape[2]/100
                flair = resample(flair,[h,h,c])
                print(flair.shape)
                t1ce = nib.load(t1ce_filename).get_fdata()
                ##t1ce.shape
                t1ce = resample(t1ce,[h,h,c])
                ##t1ce.shape
                X = np.zeros((100, 128,128, 2))

                for j in range(100):
                    X[j ,:,:,0] = cv2.resize(flair[:,:,j], (IMG_SIZE, IMG_SIZE))
                    X[j ,:,:,1] = cv2.resize(t1ce[:,:,j], (IMG_SIZE, IMG_SIZE))

                X = X/np.max(X)
                ##print(X.shape)

                input_layer = Input((IMG_SIZE, IMG_SIZE, 2))

                model = build_unet(input_layer, 'he_normal', 0.2)

                #Loading the model we have trained
                model = tf.keras.models.load_model('./model_2019_50.h5',compile=False)
                #getting the segmentation masks
                y_pred = model.predict(X)

                #rounding off the label predictions
                tumor = y_pred[:,:,:,3]
                # tumor = np.round(tumor)
                tumor[tumor>0.5]=1
                tumor[tumor<=0.5]=0

                
                imgs1 = get_pixels_hu(brains)
                np.save(f'fullimages_brain.npy', imgs1)


                brain_volume = find_volume(imgs1)
                tumor_volume = find_volume(tumor)
                file_used= f'./fullimages_brain.npy'
                imgs_to_process = np.load(file_used).astype(np.float64) 
                sample_stack(imgs_to_process)
                sample_stack(tumor)
                file_used= f'./fullimages_brain.npy'
                imgs_to_process = np.load(file_used).astype(np.float64)

                ###print(f'Shape before resampling: {imgs_to_process.shape}')

                imgs_after_resamp = resample(imgs_to_process, [1,1,1])

                ###print(f'Shape after resampling: {imgs_after_resamp.shape}')

                spac = 100/len(imgs_to_process)
                spac2 = 128/imgs_to_process.shape[1]
                tumor= resample(tumor,[spac,spac2,spac2])
                ###print('Tumor after reshape:',tumor.shape)
                ##imgs_after_resamp.shape
                tumor = tumor * 255.0
                np.max(imgs_after_resamp)
                v, f = make_mesh(imgs_after_resamp, istransp=True, threshold=20)  # 350 previously default value
                v2, f2 = make_mesh(tumor, istransp=False, threshold=50)  # 350 previously default value

                fig = plotly_3d(v, f, v2, f2)
                return render_template('/dashboard/3Dseg.html',plot=plotly.offline.plot(fig, output_type='div'))


    return render_template('3Dseg.html',plot=None)
@blueprint.route('/3Dseg', methods=['GET'])
def threeDseg():
        return render_template('/dashboard/3Dseg.html')
############################################################################################

@blueprint.route('/<template>')

def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('404.html'), 404

    except:
        return render_template('home/page-500.html'), 500





# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
