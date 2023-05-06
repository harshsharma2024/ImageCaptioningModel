from crop import CropImage
from flip import FlipImage
from rotate import RotateImage
from rescale import RescaleImage
from blur import BlurImage
import PIL
from PIL import Image

img = Image.open("./a.jpeg")
img.show()
crop = CropImage((300, 300), "random")
flip = FlipImage('vertical')
rotate = RotateImage(67)
resc = RescaleImage((100,1000))
blur=BlurImage(3)
img2 = blur(img)
img2.show()
