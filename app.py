import gradio as gr
import os.path
import numpy as np
from collections import OrderedDict
import torch
import cv2
from PIL import Image, ImageOps
import utils_image as util
from network_fbcnn import FBCNN as net
import requests

for model_path in ['fbcnn_gray.pth','fbcnn_color.pth']:
    if os.path.exists(model_path):
        print(f'loading model from {model_path}')
    else:
        #os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = 'https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/{}'.format(os.path.basename(model_path))
        r = requests.get(url, allow_redirects=True)
        open(model_path, 'wb').write(r.content)    

def inference(input_img, is_gray, input_quality, enable_zoom, zoom, x_shift, y_shift, state):

    if is_gray:
        n_channels = 1 # set 1 for grayscale image, set 3 for color image
        model_name = 'fbcnn_gray.pth'
    else:
        n_channels = 3 # set 1 for grayscale image, set 3 for color image
        model_name = 'fbcnn_color.pth'
    nc = [64,128,256,512]
    nb = 4
    

    input_quality = 100 - input_quality

    model_path = model_name   

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    if (not enable_zoom) or (state[1] is None):
        model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnrb'] = []

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        if n_channels == 1:
            open_cv_image = Image.fromarray(input_img)
            open_cv_image = ImageOps.grayscale(open_cv_image)
            open_cv_image = np.array(open_cv_image) # PIL to open cv image
            img = np.expand_dims(open_cv_image, axis=2)  # HxWx1
        elif n_channels == 3:
            open_cv_image = np.array(input_img) # PIL to open cv image
            if open_cv_image.ndim == 2:
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2RGB)  # GGG
            else:
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)  # RGB

        img_L = util.uint2tensor4(open_cv_image)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------
          
        img_E,QF = model(img_L)
        QF = 1- QF
        img_E = util.tensor2single(img_E)
        img_E = util.single2uint(img_E)

        qf_input = torch.tensor([[1-input_quality/100]]).cuda() if device == torch.device('cuda') else torch.tensor([[1-input_quality/100]])
        img_E,QF = model(img_L, qf_input)  
        QF = 1- QF
        img_E = util.tensor2single(img_E)
        img_E = util.single2uint(img_E)

        if img_E.ndim == 3:
            img_E = img_E[:, :, [2, 1, 0]]
    if (state[1] is not None) and enable_zoom:
      img_E = state[1]
    out_img = Image.fromarray(img_E)
    out_img_w, out_img_h = out_img.size # output image size
    zoom = zoom/100
    x_shift = x_shift/100
    y_shift = y_shift/100
    zoom_w, zoom_h = out_img_w*zoom, out_img_h*zoom
    zoom_left, zoom_right = int((out_img_w - zoom_w)*x_shift), int(zoom_w + (out_img_w - zoom_w)*x_shift)
    zoom_top, zoom_bottom = int((out_img_h - zoom_h)*y_shift), int(zoom_h + (out_img_h - zoom_h)*y_shift)
    if (state[0] is None) or not enable_zoom:
        in_img = Image.fromarray(input_img)
        state[0] = input_img
    else:
        in_img = Image.fromarray(state[0])
    in_img = in_img.crop((zoom_left, zoom_top, zoom_right, zoom_bottom))
    in_img = in_img.resize((int(zoom_w/zoom), int(zoom_h/zoom)), Image.NEAREST)
    out_img = out_img.crop((zoom_left, zoom_top, zoom_right, zoom_bottom))
    out_img = out_img.resize((int(zoom_w/zoom), int(zoom_h/zoom)), Image.NEAREST)

    return img_E, in_img, out_img, [state[0],img_E]
    
interface = gr.Interface(
    fn = inference,
    inputs = [gr.inputs.Image(),
              gr.inputs.Checkbox(label="Grayscale (Check this if your image is grayscale)"),
              gr.inputs.Slider(minimum=1, maximum=100, step=1, label="Intensity (Higher = stronger JPEG artifact removal)"),
              gr.inputs.Checkbox(default=False, label="Edit Zoom preview (This is optional. "
                                                      "Check this after the image result is loaded to edit zoom parameters "
                                                      "without processing the input image.)"),
              gr.inputs.Slider(minimum=10, maximum=100, step=1, default=50, label="Zoom Image "
                                                                                  "(Use this to see the image quality up close. "
                                                                                   "100 = original size)"),
              gr.inputs.Slider(minimum=0, maximum=100, step=1, label="Zoom preview horizontal shift "
                                                                     "(Increase to shift to the right)"),
              gr.inputs.Slider(minimum=0, maximum=100, step=1, label="Zoom preview vertical shift "
                                                                     "(Increase to shift downwards)"),
              gr.inputs.State(default=[None,None])
              ],
    outputs = [gr.outputs.Image(label="Result"),
               gr.outputs.Image(label="Before:"),
               gr.outputs.Image(label="After:"),
               "state"],
    examples = [["doraemon.jpg",False,60,False,42,50,50],
               ["tomandjerry.jpg",False,60,False,40,57,44],
               ["somepanda.jpg",True,100,False,30,8,24],
               ["cemetry.jpg",False,70,False,20,44,77],
               ["michelangelo_david.jpg",True,30,False,12,53,27],
               ["elon_musk.jpg",False,45,False,15,33,30]],
    title = "JPEG Artifacts Removal [FBCNN]",
    description = "Gradio Demo for JPEG Artifacts Removal. To use it, simply upload your image, "
                  "or click one of the examples to load them. Check out the paper and the original GitHub at the link below. "
                  "JPEG artifacts are noticeable distortion of images caused by JPEG lossy compression. "
                  "This is not a super resolution AI but a JPEG compression artifact remover.",
    article = "<p style='text-align: center;'><a href='https://github.com/jiaxi-jiang/FBCNN'>FBCNN GitHub Repo</a><br>"
              "<a href='https://arxiv.org/abs/2109.14573'>Towards Flexible Blind JPEG Artifacts Removal (FBCNN, ICCV 2021)</a></p>",
    allow_flagging="never"
).launch(enable_queue=True)