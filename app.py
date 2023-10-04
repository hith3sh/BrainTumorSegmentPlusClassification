from dependencies import *
from unet import *

app = Flask(__name__)

#For Segmentation
upload_folder = "./static"
device = "cpu"
segment_model = None
path = "./model_state_dict.pt"
data_transforms = None
'''
#for classification
MODEL_PATH = 'modelres50.h5'
classifier_model = load_model(MODEL_PATH)

#for classification

def model_predict(img_path, classifier_model):
    #convert .tif file to .jpg format
    with Image.open(img_path) as img:
        img = img.convert('RGB')
        jpg_path= img_path.split('.')[0] +'jpg'
        img.save(jpg_path, format='JPEG')
    
    #loading the image for classification
    img = image.load_img(img_path, target_size=(200,200)) 
    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255
    preds = classifier_model.predict(img)
    pred = np.argmax(preds,axis = 1)
    str0 = 'Glioma'
    str1 = 'Meningioma'
    str3 = 'pituitary'
    str2 = 'No Tumour'
    if pred[0] == 0:
        return str0
    elif pred[0] == 1:
        return str1
    elif pred[0]==3:
        return str3
    else:
        return str2
'''
#for segmentation
def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=False)
    image = image.unsqueeze(0)
    new_tensor = image.clone()
    
    return new_tensor

def process_image(data_transforms, path_name, image_name, filemodel):
    with torch.no_grad():
        img = image_loader(data_transforms, path_name)
        pred = segment_model(img)
        plt.subplot(1,2,1)                      #original image 
        plt.imshow(np.squeeze(img.cpu().numpy()).transpose(1,2,0))
        plt.title('Original Image')

        plt.subplot(1,2,2)                      #segmented image
        plt.imshow(np.squeeze(pred.cpu()) > .5)
        plt.title('Tumour Prediction')
        plt.savefig("%s/%s-SEGMENTED.png" % (upload_folder, image_name), bbox_inches = "tight")


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:                                                          #if an image was recieved
            image_location = os.path.join(upload_folder,image_file.filename)    #go to ./static folder 
            image_file.save(image_location)                                     #save the image in ./static folder

            image_name = os.path.basename(image_location)  #to get only the path name     
            image_name = image_name.split('.')[0]       #name of image without file extension
            
            #calling for segmentation
            process_image(data_transforms, image_location, image_name, segment_model)
            #calling for classification
            #calling for classification
            #predicted_tumor = model_predict(image_location, classifier_model)

            return render_template("index.html", image_loc = ("%s-SEGMENTED.png" % image_name))
            
    return render_template("index.html", prediction=0, image_loc=None)


if __name__ == "__main__":
    segment_model = UNet().to(device)   #loading the whole model architecture
    segment_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))   #loading the pre trained weights

    segment_model.eval()    #switch the model from training mode to evaluation mode

    data_transforms = transforms.Compose([transforms.Resize(256),transforms.ToTensor()])
    app.run(host="0.0.0.0", port=12000, debug=True)
