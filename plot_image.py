import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

# Load your image
image_path = rf"C:\Users\19255\facial_landmarks_infants\data\images\google\google-00.png"
img = mpimg.imread(image_path)

# Assuming you have two lists `x_coords` and `y_coords`
x_coords = [225.0, 234.0, 240.0, 249.0, 269.0, 296.0, 325.0, 353.0, 382.0, 402.0, 422.0, 440.0, 447.0, 449.0, 444.0, 438.0, 429.0, 260.0, 276.0, 294.0, 311.0, 329.0, 380.0, 391.0, 402.0, 411.0, 420.0, 362.0, 367.0, 371.0, 376.0, 353.0, 365.0, 376.0, 385.0, 393.0, 291.0, 305.0, 318.0, 331.0, 322.0, 307.0, 380.0, 391.0, 402.0, 416.0, 409.0, 396.0, 338.0, 356.0, 371.0, 380.0, 387.0, 398.0, 409.0, 400.0, 391.0, 382.0, 373.0, 358.0, 345.0, 371.0, 380.0, 389.0, 402.0, 389.0, 380.0, 371.0]  # Replace with your x coordinates
y_coords = [221.0, 246.0, 270.0, 292.0, 308.0, 312.0, 317.0, 319.0, 317.0, 299.0, 283.0, 268.0, 248.0, 226.0, 203.0, 183.0, 164.0, 190.0, 177.0, 168.0, 166.0, 166.0, 157.0, 152.0, 148.0, 146.0, 148.0, 183.0, 195.0, 206.0, 217.0, 239.0, 239.0, 237.0, 234.0, 230.0, 203.0, 192.0, 188.0, 197.0, 206.0, 208.0, 188.0, 175.0, 172.0, 177.0, 186.0, 190.0, 279.0, 266.0, 257.0, 259.0, 254.0, 257.0, 261.0, 272.0, 277.0, 281.0, 283.0, 283.0, 277.0, 266.0, 266.0, 263.0, 263.0, 266.0, 268.0, 268.0]  # Replace with your y coordinates

# Zip the coordinates
coordinates = zip(x_coords, y_coords)

box = [225.0, 146.0, 449.0, 319.0]

fig, ax = plt.subplots()
ax.imshow(img)

rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='blue', facecolor='none')
ax.add_patch(rect)

for x, y in coordinates:
    ax.scatter(x, y, c='red', marker='o')  # Adjust color and marker as needed

plt.show()
