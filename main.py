import streamlit as st
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


st.set_page_config(page_title="Adenrele's Fruit & Vegetable Classifier", layout='wide')
classes = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
           'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon',
           'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato',
           'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']


transform = transforms.Compose([
    transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5) 
        self.pool = nn.MaxPool2d(4, 4) 
        self.fc1 = nn.Linear(43264, 128)  
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 36)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
@st.cache_data
def load_model():
    model = Net()
    try:
        model.load_state_dict(torch.load("my_best_model.pth", map_location=torch.device('cpu')))
        model.eval()
    except FileNotFoundError:
        st.error("Model file not found! Please check the file path.")
        return None
    return model

net = load_model()


st.markdown(
    "<h1 style='text-align: center;'>🍏🥕 Adenrele's Fruit & Vegetable Image Classifier</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center;'><a href = 'https://adenrele.co.uk'>Adenrele.co.uk</a></h3>", 
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center;'>Upload an image to identify it.</h3>", 
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1])

with col1:
    st.image("results/PerClassAccuracy.png", use_container_width=True, caption= "Accuaracy of the model for each class it was trained on using the provided dataset." )
    st.text("Note that classes with lower accuracy are those for which fewer samples were provided in the training dataset.")

with col2: 
    uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "png", "jpeg"])
    image = None

    # Create two sub-columns inside col2
    sub_col1, sub_col2 = st.columns([1, 1])

    with sub_col1:
       
        if uploaded_file is not None:
            image = Image.open(io.BytesIO(uploaded_file.getvalue()))
            resized_image = image.resize((224, 224))
            st.image(resized_image, caption="Uploaded Image")

    with sub_col2:
        def predict(image, model):
            if model is None or image is None:
                return None

            image = image.convert("RGB")  
            transformed_image = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(transformed_image)

            probabilities = F.softmax(output[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)

            results = [{"Class": classes[int(top5_catid[i])], "Confidence": f"{top5_prob[i].item() * 100:.2f}%"}
                       for i in range(top5_prob.size(0))]
            return results
        
    with sub_col1:   
        if st.button("🔍 Identify Image") and image is not None:
            with st.spinner("🔄 Running inference..."):
                results = predict(image, net)
                if results:
                    with sub_col2:
                        st.table(results)
                else:
                    st.warning("No predictions available. Please upload an image and try again.")

