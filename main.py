import streamlit as st
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import load as L, no_grad, topk
from PIL import Image

classes = ['apple',  'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
 'chilli pepper',  'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon',
  'lettuce', 'mango', 'onion', 'orange',  'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']


st.set_page_config(page_title="Fruit & Vegetable Classifier", layout = 'centered')
st.title("🍏🥕 Fruit & Vegetable Image Classifier")
st.subheader("Upload an image to classify it.")
st.text("""Supported classes include: Apple, Banana, Beetroot', Bell pepper, Cabbage, Capsicum, Carrot, Cauliflower, Chilli pepper, Corn, Cucumber, Eggplant, Garlic, Ginger, Grapes, Jalepeno, Kiwi, Lemon, Lettuce, Mango, Onion, Orange, Paprika, Pear, Peas, Pineapple, Pomegranate, Potato, Raddish, Soy beans, Spinach, Sweetcorn, Sweetpotato, Tomato, Turnip, Watermelon """)

# Load the model once and store it in Streamlit session state
@st.cache_resource
def load_model():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            fc1_input_size = self.get_fc_input_size()
            self.fc1 = nn.Linear(fc1_input_size, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 36)

        def get_fc_input_size(self):
            dummy_input = torch.randn(1, 3, 270, 270)
            dummy_output = self.pool(F.relu(self.conv2(self.pool(F.relu(self.conv1(dummy_input))))))
            return dummy_output.numel()

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    try:
        net.load_state_dict(L("fruit_veg.pth", map_location=torch.device('cpu')))
        net.eval()
        return net
    except FileNotFoundError:
        st.error("Model file not found! Please check the file path.")
        return None

# Define transformations
transform = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(270),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Image uploader
def load_image():
    uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
          # Resize image (adjust size as needed)
        new_size = (224, 224)  # Change width & height as needed
        resized_image = image.resize(new_size)

       # Display the resized image
        st.image(resized_image, use_container_width=False)
        return image
    return None

# Prediction function
def predict(image, model):
    if model is None:
        return

    image = image.convert("RGB")  
    transformed_image = transform(image).unsqueeze(0)

    with no_grad():
        output = model(transformed_image)

    probabilities = F.softmax(output[0], dim=0)
    top5_prob, top5_catid = topk(probabilities, 5)

    results = [{"Class": classes[int(top5_catid[i])], "Confidence": f"{top5_prob[i].item() * 100:.2f}%"}
               for i in range(top5_prob.size(0))]

    st.table(results)

# Main function
def main():
    model = load_model()
    image = load_image()

    if st.button("🔍 Classify Image"):
        if image is None:
            st.warning("⚠️ Please upload an image before proceeding.")
        else:
            with st.spinner("🔄 Running inference..."):
                predict(image, model)

if __name__ == "__main__":
    main()


#commands streamlit run main.py
#pipreqs
#procfile
#setup.sh
#git init
#Heroku login
#
