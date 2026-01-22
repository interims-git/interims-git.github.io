from PIL import Image
import matplotlib.pyplot as plt

# Open image with PIL
img = Image.open("/media/barry/56EA40DEEA40BBCD/DATA/studio_test/backgrounds/train/005.jpg")

# Display using matplotlib
plt.imshow(img)
plt.axis("off")  # Hide axes for cleaner look
plt.show()

A = (458, 622)
B = (3819, 511)
C = (375, 2554)
D = (3999,2504)