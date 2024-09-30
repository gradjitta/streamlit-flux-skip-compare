import streamlit as st
from PIL import Image
import os
import numpy as np
import re

def load_image(image_path):
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        return None

def compute_difference(img1, img2):
    if img1.size != img2.size:
        st.error("Image sizes do not match. Cannot compute difference.")
        return None
    
    diff = Image.new('RGB', img1.size)
    for x in range(img1.size[0]):
        for y in range(img1.size[1]):
            r1, g1, b1 = img1.getpixel((x, y))
            r2, g2, b2 = img2.getpixel((x, y))
            diff.putpixel((x, y), (abs(r1-r2), abs(g1-g2), abs(b1-b2)))
    return diff

def get_images(folder_path):
    images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort(key=lambda x: [int(i) for i in re.findall(r'\d+', x)] if re.findall(r'\d+', x) else [])
    return images

def main():
    st.title("Image Comparison: Original vs Skip Layers")

    folder_path = os.path.join(os.getcwd(), 'images')
    original_image_path = os.path.join(folder_path, 'image_original.png')

    img_original = load_image(original_image_path)

    if img_original is None:
        st.error("There is no reference image. Please add 'image_original.png' to the 'images' folder.")
        return

    if os.path.exists(folder_path):
        images = get_images(folder_path)
        images = [img for img in images if img != 'image_original.png']

        if images:
            st.subheader("Select Skip Layer Image")
            selected_skip = st.selectbox("Choose a skip layer image:", images)
            img_skip = load_image(os.path.join(folder_path, selected_skip))

            if img_skip:
                # Side-by-side comparison of original and selected skip layer image
                st.subheader("Original vs Skip Layer Image")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_original, use_column_width=True, caption="Original")
                with col2:
                    st.image(img_skip, use_column_width=True, caption=f"Skip: {selected_skip}")

                # Image difference
                st.subheader("Image Difference")
                diff_img = compute_difference(img_original, img_skip)
                if diff_img:
                    st.image(diff_img, use_column_width=True, caption="Difference")

                    # Compute and display difference statistics
                    diff_array = np.array(diff_img)
                    mean_diff = np.mean(diff_array)
                    max_diff = np.max(diff_array)
                    st.write(f"Mean difference: {mean_diff:.2f}")
                    st.write(f"Max difference: {max_diff}")

                    # Histogram of differences
                    st.subheader("Histogram of Differences")
                    hist_values, hist_bins = np.histogram(diff_array.flatten(), bins=50, range=(0, 255))
                    st.bar_chart(hist_values)
                    st.write("X-axis: Difference value (0-255)")
                    st.write("Y-axis: Frequency")
        else:
            st.warning("No skip layer images found in the 'images' folder.")
    else:
        st.error("The 'images' folder does not exist in the current directory.")

if __name__ == "__main__":
    main()
