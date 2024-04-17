import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import math
import tempfile
import torch
from torchvision import transforms
import tenseal as ts
import torch.nn.functional as F
import time





def show_probabilities(prob_cat, prob_dog):
    st.metric(label="Probabilité d'être un chat", value=f"{prob_cat * 100:.2f}%")
    st.metric(label="Probabilité d'être un chien", value=f"{prob_dog * 100:.2f}%")

def show_jauge(prob_cat, prob_dog):
    fill_width_cat = prob_cat / (prob_cat + prob_dog) * 100  
    fill_width_dog = prob_dog / (prob_cat + prob_dog) * 100

    # Générer les parties de la jauge en fonction des probabilités
    cat_part = f'<div style="flex: {fill_width_cat}; background-color: #76b900; color: white; line-height: 20px; text-align: left; padding-left: 5px;">Chat {fill_width_cat:.2f}%</div>' if prob_cat > 0 else ''
    dog_part = f'<div style="flex: {fill_width_dog}; background-color: #4d79ff; color: white; line-height: 20px; text-align: right; padding-right: 5px;">Chien {fill_width_dog:.2f}%</div>' if prob_dog > 0 else ''

    # HTML/CSS pour la jauge avec conditions pour afficher ou non chaque partie
    gauge_html = f"""
    <div style="display: flex; width: 100%; background-color: #e0e0e0; border-radius: 25px; overflow: hidden;">
        {cat_part}{dog_part}
    </div>
    """

    # Afficher la jauge dans Streamlit
    st.markdown(gauge_html, unsafe_allow_html=True)






# Global variable to store the model
loaded_models = {}

class ConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=2):
        super(ConvNet, self).__init__()        
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        x = x * x  # square activation function
        x = x.view(-1, 256)  # flattening while keeping the batch axis
        x = self.fc1(x)
        x = x * x  # square activation function
        x = self.fc2(x)
        return x

class EncConvNet:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()
        
        
    def forward(self, enc_x, windows_nb):
        # conv layer
        
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        # fc1 layer
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x.square_()
        # fc2 layer
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def crypt_image(input_path, output_path, key=150):
    with open(input_path, 'rb') as file:
        original_data = file.read()
    start_of_image = original_data.find(b'\xff\xda')
    if start_of_image == -1:
        return None
    start_of_image += 2
    metadata = original_data[:start_of_image]
    image_data = bytearray(original_data[start_of_image:])
    for index, value in enumerate(image_data):
        image_data[index] = value ^ key
    with open(output_path, 'wb') as encrypted_file:
        encrypted_file.write(metadata + image_data)
    return output_path

def bytes_to_bmp(file_path):
    with open(file_path, 'rb') as file:
        cyphertext = file.read()
    num_bytes = len(cyphertext)
    num_pixels = (num_bytes + 2) // 3
    W = H = math.ceil(math.sqrt(num_pixels))
    imagedata = cyphertext + b'\0' * (W * H * 3 - len(cyphertext))
    image = Image.frombytes('RGB', (W, H), imagedata)    
    return image

def load_model(model_path, model_type='tf'):
    if model_path not in loaded_models:
        if model_type == 'tf':
            loaded_models[model_path] = tf.keras.models.load_model(model_path)
        elif model_type == 'torch':
            loaded_models[model_path] = torch.load(model_path)
    return loaded_models[model_path]

def predict_tf(image, model_path):
    model = load_model(model_path, 'tf')
    img = image.resize((64, 64))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    return prediction[0][0]

def predict_torch(image, model_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    model = load_model(model_path, 'torch')
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    prediction = torch.max(output, 1)[1].item()
    return prediction



def encrypt_and_display(image, uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    encrypted_file_path = tmp_path + ".encrypted"
    crypt_image(tmp_path, encrypted_file_path)
    image = bytes_to_bmp(encrypted_file_path)
    st.image(image, use_column_width=True)
    return encrypted_file_path

def predict_and_display(image, model_path, model_type):
    # Initialiser les probabilités par défaut
    prob_cat = 0.0
    prob_dog = 0.0

    if model_type == 'tf':
        prediction = predict_tf(image, model_path)
        # Pour un modèle TensorFlow, 'prediction' peut être une probabilité directe
        # pour l'une des classes, selon comment le modèle a été entraîné.
        prob_cat = 1.0 - prediction  # Si la prédiction est pour un chien, la probabilité d'être un chat
        prob_dog = prediction
    else:
        # Pour le modèle PyTorch
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0)
        model = load_model(model_path, 'torch')
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            prob_cat = probabilities[0].item()
            prob_dog = probabilities[1].item()

    # Utilisez prob_cat et prob_dog pour afficher la prédiction
    if prob_cat > prob_dog:
        st.write('C\'est probablement un **chat** 🐱')
    else:
        st.write('C\'est probablement un **chien** 🐶')

    # Afficher les probabilités et la jauge pour toutes les prédictions
    show_probabilities(prob_cat, prob_dog)
    show_jauge(prob_cat, prob_dog)


def encrypt_image(image_path, context, kernel_shape, stride):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Ajoute une dimension de batch
    image_tensor = image_tensor.squeeze().numpy()  # Convertir en numpy array

    enc_image, windows_nb = ts.im2col_encoding(
        context, 
        image_tensor.tolist(), 
        kernel_shape[0], 
        kernel_shape[1], 
        stride
    )
    return enc_image, windows_nb

def predict_enc_cnn(image_path, model, context):
    kernel_shape = (7, 7)
    stride = 3
    enc_image, windows_nb = encrypt_image(image_path, context, kernel_shape, stride)
    enc_output = model.forward(enc_image, windows_nb)
    output = enc_output.decrypt()
    output = np.array(output).reshape(1, -1)
    output_tensor = torch.FloatTensor(output)

    predicted_class = np.argmax(output)
    probabilities = F.softmax(output_tensor, dim=1)
    return probabilities,predicted_class
    


def main():

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, 26, 26, 26, 26, 26, 26, 31]
    )
    context.global_scale = pow(2, 26)
    context.generate_galois_keys()
    model = torch.load("complete_model_64.pth")
    model.eval()
    enc_model = EncConvNet(model)
    
    st.title('Démo de détection de chat ou de chien sur des images chiffrées')
    uploaded_file = st.file_uploader("Choisir une image de chat ou de chien", type=["jpg", "png"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        chiffrement = st.selectbox('Choisir le chiffrement', ['pas de chiffrement', 'EncCNN'])

        if chiffrement == 'EncCNN':
            if st.button('Prédire avec ENcCNN'):
                start = time.time()
                probabilities, predicted_class = predict_enc_cnn(tmp_path, enc_model,context)
                if predicted_class == 0:
                    st.write('C\'est probablement un **chat** 🐱')
                else:
                    st.write('C\'est probablement un **chien** 🐶')
                prob_cat = probabilities[0][0].item()
                prob_dog = probabilities[0][1].item()
                show_probabilities(prob_cat, prob_dog)
                show_jauge(prob_cat, prob_dog)
                end = time.time()
                execution_time = end - start
                execution_time_rounded = round(execution_time, 1)
                st.write("Temps d'exécution de la prédiction en secondes :", execution_time_rounded)
        
        elif chiffrement == 'pas de chiffrement':
            if st.button('Prédire sans chiffrement'):
                start = time.time()
                predict_and_display(image, 'modele_clair.h5', 'tf')
                end = time.time()
                execution_time = end - start
                execution_time_rounded = round(execution_time, 1)
                st.write("Temps d'exécution de la prédiction en secondes :", execution_time_rounded)




if __name__ == '__main__':
    main()
