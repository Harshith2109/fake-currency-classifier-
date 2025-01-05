import torch
from torchvision import transforms
from PIL import Image
from model import EnhancedDeepConvNet
import streamlit as st
import warnings
import joblib 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

model_path = './model/model_full.pth'
model = EnhancedDeepConvNet()
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.eval()
print("Model loaded successfully.")


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

st.title('Fake Note Classifier')
df=st.file_uploader(label="Enter the Image:",type=['jpg','jpeg','png'])

if df:
    try:
        if(df):
            print("image loaded")


    except Exception as e:
        
        image_path = 'D:/fakenote/fake-currency-classifier-/note.jpg'
        image = Image.open(image_path).convert('RGB')  
        input_tensor = transform(image).unsqueeze(0)  

        with torch.no_grad():
            output = model(input_tensor)
            probability = output.item()  
            prediction = "Real" if probability > 0.7 else "Fake"

            st.write(prediction)

    pr=st.button("submit")

    if (pr):

        image = Image.open(df).convert('RGB')  
        input_tensor = transform(image).unsqueeze(0)
        input_tensor = transform(image).unsqueeze(0)  

        with torch.no_grad():
            output = model(input_tensor)
            probability = output.item()  
            prediction = "Real" if probability > 0.7 else "Fake"
            st.write(prediction)
            print(f"Prediction: {prediction} (Probability: {probability:.2f})")

    


# image_path = 'D:/fakenote/fake-currency-classifier-/note.jpg' 
# image = Image.open(image_path).convert('RGB')  
# input_tensor = transform(image).unsqueeze(0)  

# with torch.no_grad():
#     output = model(input_tensor)
#     probability = output.item()  
#     prediction = "Real" if probability > 0.7 else "Fake"

#print(f"Prediction: {prediction} (Probability: {probability:.2f})")


