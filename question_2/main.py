#Imports
from my_package.model import ImageCaptioningModel
from my_package.data import Dataset, Download
from my_package.data.transforms import FlipImage, RescaleImage, BlurImage, CropImage, RotateImage
import numpy as np
from PIL import Image



def experiment(annotation_file, captioner, transforms, outputs):
    '''
        Function to perform the desired experiments

        Arguments:
        annotation_file: Path to annotation file
        captioner: The image captioner
        transforms: List of transformation classes
        outputs: Path of the output folder to store the images
    '''

    #Create the instances of the dataset, download
    data = Dataset(annotation_file,transforms) 
    DownLoad=Download()

    # #Print image names and their captions from annotation file using dataset object
    for i in range(len(data)):
        dict=data.__getann__(i)
        name=dict["file_name"]
        print(f"Image {i} name: {name}")
        captions=dict["captions"]
        print(f"Captions for image {i}:")
        for j in range(len(captions)):
            b=captions[j]
            capt=b["caption"]
            print(f"Caption {j} :{capt}")
        print("\n")


    #Download images to ./data/imgs/ folder using download object
    for i in range(len(data)):
        dict=data.__getann__(i)
        url=dict["url"]
        name=dict["file_name"]
        path="./data/imgs/"+name
        DownLoad.__call__(path,url)

    #Transform the required image (roll number mod 10) and save it seperately
    my_img=data.__getann__(3)
    a=my_img["file_name"]
    my_img_path="./data/imgs/"+a
   
    img=data.__transformitem__(my_img_path)
    img.save(outputs)
 
    #Get the predictions from the captioner for the above saved transformed image
   
 
    predictions=captioner(my_img_path,3)
   
    for i in range(len(predictions)):
        element= predictions[i]
        print(f"Caption {i+1}: {element}")
        
    print("\n")

def main():
    captioner = ImageCaptioningModel()
    
    experiment('./data/annotations.jsonl', captioner, [], "Output/experiment1.jpg") # Sample arguments to call experiment()
    print("Experiment 1 completed")
    print("\n\n")
    experiment('./data/annotations.jsonl', captioner, [FlipImage('horizontal')], "Output/experiment2.jpg") # Sample arguments to call experiment()
    print("Experiment 2 completed")
    print("\n\n")

    experiment('./data/annotations.jsonl', captioner, [ BlurImage(2)], "Output/experiment3.jpg") # Sample arguments to call experiment()
    print("Experiment 3 completed")
    print("\n\n")
    img=Image.open('./data/imgs/6.jpg')
    w,h=img.size

    experiment('./data/annotations.jsonl', captioner, [RescaleImage((2*w,2*h))], "Output/experiment4.jpg") # Sample arguments to call experiment()
    print("Experiment 4 completed")
    print("\n\n")

    experiment('./data/annotations.jsonl', captioner, [RescaleImage((int(w/2),int(h/2)))], "Output/experiment5.jpg") # Sample arguments to call experiment()
    print("Experiment 5 completed")
    print("\n\n")

    experiment('./data/annotations.jsonl', captioner, [RotateImage(90)], "Output/experiment6.jpg") # Sample arguments to call experiment()
    print("Experiment 6 completed")
    print("\n\n")

    experiment('./data/annotations.jsonl', captioner, [RotateImage(45)], "Output/experiment7.jpg") # Sample arguments to call experiment()
    print("Experiment 7 completed")
    print("\n\n")

if __name__ == '__main__':
    main()
