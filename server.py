import base64
import io

from flask import Flask, render_template, request
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import requests
import torch

from train import Net, IMAGE_SIZE, COLOR, CHANNELS

app = Flask(__name__)
net = Net()
net.load_state_dict(torch.load("model.pth", weights_only=False))
net.eval()
transform = transforms.Compose(
    [
        transforms.Lambda(lambda img: img.convert(COLOR)),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * CHANNELS, std=[0.5] * CHANNELS),
    ]
)


def image_to_base64(image):
    buff = io.BytesIO()
    image.save(buff, format="JPEG")
    head = bytes("data:image/jpeg;base64,", encoding="utf-8")
    data = base64.b64encode(buff.getvalue())
    return (head + data).decode("utf-8")


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    file = request.files["file"]
    url = request.form["url"]

    if not url and not file:
        return "No file or url provided"

    if file and url:
        return "Provide only one file or url"

    if url:
        r = requests.get(url)
        file = r.content
    else:
        file = file.read()

    image = Image.open(io.BytesIO(file))
    base_64 = image_to_base64(image)
    output = F.sigmoid(net(transform(image).unsqueeze(0))) * 100
    cat, dog = output.squeeze().tolist()

    winner = "cat" if cat > dog else "dog"
    return render_template(
        "result.html", cat=f"{cat:.2f}%", dog=f"{dog:.2f}%", winner=winner, image=base_64
    )
