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

import nibabel as nib
import os
import albumentations as A
import numpy as np
import plotly.graph_objects as go
import numpy as np






# os.system("git clone https://github.com/NVlabs/stylegan3")

sys.path.append("stylegan3")
# Load the StyleGAN model
network_pkl = "brainmrigan.pkl"

with open(network_pkl, 'rb') as f:
    G = pickle.load(f)['G_ema']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G.eval()
G.to(device)






API_URL = "https://api-inference.huggingface.co/models/yahyasmt/brain_tumor_2"
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


@blueprint.route('/index',methods=['GET', 'POST'])
@login_required
def index():
    return render_template('home/index.html', segment='index')

#########################################################################################
@blueprint.route('/indeximage',methods=['GET', 'POST'])
@login_required
def indeximage():
    if request.method == "POST":
        prompt = request.form["input-text"]

        image_bytes = query({
            "inputs": prompt,
        })
        with open("./apps/static/assets/mdl/aa.jpeg", "wb") as image_file:
            image_file.write(image_bytes)

    return render_template('home/indexImage.html')

########################################################################################
@blueprint.route('/predict', methods=['POST'])
def get_prediction():
    try:
        Seed = int(request.form['Seed'])
        noise_mode = request.form['noise_mode']
        truncation_psi = float(request.form['truncation_psi'])
        result_image = predict(Seed, noise_mode, truncation_psi)
        
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

@blueprint.route('/brain',methods=['GET'])
@login_required
def ganbrain():
    return render_template('home/Ganbrain.html', segment='index')
    



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

        return render_template('home/result.html', plot_div=plot_div)

    return redirect(request.url)

# Serve uploaded files
@blueprint.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@blueprint.route('/3D')
@login_required
def ThreeD():
    return render_template('home/3D.html')


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
            return render_template('home/resulttotal3d.html', plot=plotly.offline.plot(fig, output_type='div'))

    # Render the HTML form when the page is first loaded
    return render_template('home/resulttotal3d.html', plot=None)
############################################################################################




@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

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
