# import the necessary packages
import matplotlib
matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch


def Draw3DLines(videopath):
    # MediaPipe Objectron instantiation
    mp_objectron = mp.solutions.objectron
    mp_drawing = mp.solutions.drawing_utils

    objectron = mp_objectron.Objectron(
        static_image_mode=False,
        max_num_objects=5,
        min_detection_confidence=0.5,
        model_name='Chair'
    )

    # Open the video file
    video_path = videopath
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create a VideoWriter object
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Use 'H264' codec for MP4
    out = cv2.VideoWriter('annotated_video.mp4', fourcc, 30, (width, height))

    # Loop through each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with Objectron
        results = objectron.process(image)

        # Draw 3D bounding boxes on the image
        annotated_image = image.copy()
        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=detected_object.landmarks_2d,
                    connections=mp_objectron.BOX_CONNECTIONS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

        # Convert the RGB image back to BGR for saving with OpenCV
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Write the annotated frame to the video
        out.write(annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video and destroy windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Annotated video saved as 'annotated_video.mp4'")


def DrawEdges(img_path):
     # Read image
    image_path = img_path
    image = Image.open(image_path).convert('L') # Convert to grayscale
    image_data = np.array(image)

    # Apply Canny edge detection
    edges = cv2.Canny(image_data, 200, 200)

    # Create a new figure
    fig = plt.figure(figsize=(12, 6))

    # Create a 3D subplot for the edge visualization
    ax1 = fig.add_subplot(121, projection='3d')
    x, y = np.meshgrid(range(edges.shape[1]), range(edges.shape[0]))
    ax1.plot_surface(x, y, edges, cmap='viridis')

    # Create a 2D subplot for the edges without x and y axis
    ax2 = fig.add_subplot(122)
    ax2.imshow(edges, cmap='gray')
    ax2.axis('off') # Turn off the x and y axis

    plt.show()

    # Read image
    image_path = img_path
    image = Image.open(image_path).convert('L') # Convert to grayscale
    image_data = np.array(image)

    # Create a new figure
    fig = plt.figure(figsize=(12, 6))

    # Create a 3D subplot for the visualization
    ax1 = fig.add_subplot(121, projection='3d')
    x, y = np.meshgrid(range(image_data.shape[1]), range(image_data.shape[0]))
    ax1.plot_surface(x, y, image_data, cmap='viridis')

    # Create a 2D subplot for the original image without x and y axis
    ax2 = fig.add_subplot(122)
    ax2.imshow(image, cmap='gray')
    ax2.axis('off') # Turn off the x and y axis

    plt.show()


def Draw3DVizObject(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Explicitly set the DPI for the figure
    fig = plt.figure(figsize=(12, 6), dpi=100)
    canvas_width, canvas_height = fig.canvas.get_width_height()
    plt.close('all')

    # Define the codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_3d = cv2.VideoWriter('3d_vis.mp4', fourcc, fps, (canvas_width, canvas_height))
    out_2d = cv2.VideoWriter('2d_edges.mp4', fourcc, fps, (canvas_width, canvas_height), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Convert frame to grayscale and apply Canny edge detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 200, 200)

        # Create a new figure for 3D plot
        fig = plt.figure(figsize=(12, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(range(edges.shape[1]), range(edges.shape[0]))
        ax.plot_surface(x, y, edges, cmap='viridis')
        plt.tight_layout()

        # Convert 3D plot to an image
        fig.canvas.draw()
        img_3d = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_3d = img_3d.reshape((canvas_height, canvas_width, 3))

        # Resize the 2D edges to match the canvas size
        edges_resized = cv2.resize(edges, (canvas_width, canvas_height))

        # Write the 3D and 2D frames
        out_3d.write(img_3d)
        out_2d.write(edges_resized)

        plt.close('all')

    cap.release()
    out_3d.release()
    out_2d.release()

    print("3D visualization video saved as 3d_vis.mp4.")
    print("2D edge video saved as 2d_edges.mp4.")


def DisparityMap_StereoBM(video_path):
    # Callback function for trackbars, does nothing
    def nothing(x):
        pass

    # Open video file
    cap = cv2.VideoCapture(video_path)

    cv2.namedWindow('Disparity Map')
    cv2.createTrackbar('Num Disparities', 'Disparity Map', 1, 10, nothing) # Max value * 16
    cv2.createTrackbar('Block Size', 'Disparity Map', 5, 50, nothing) # Must be odd

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Splitting left and right images
        height, width = frame.shape[:2]
        left_image = cv2.cvtColor(frame[:, :width//2], cv2.COLOR_BGR2GRAY)
        right_image = cv2.cvtColor(frame[:, width//2:], cv2.COLOR_BGR2GRAY)

        # Get the trackbar values
        num_disp = cv2.getTrackbarPos('Num Disparities', 'Disparity Map') * 16
        block_size = cv2.getTrackbarPos('Block Size', 'Disparity Map')
        block_size = block_size + 1 if block_size % 2 == 0 else block_size

        # Create StereoBM object with the current settings
        stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)

        # Compute the disparity map
        disparity = stereo.compute(left_image, right_image)

        # Normalize the disparity map for visualization
        disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Show the disparity map
        cv2.imshow('Disparity Map', disparity_visual)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def ImageAnalysis(img1, img2):
    # Load the images
    stereo_l = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    stereo_r = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

    # Resize if necessary
    if stereo_l.shape != stereo_r.shape:
        stereo_r = cv2.resize(stereo_r, (stereo_l.shape[1], stereo_l.shape[0]))

    # Create StereoBM and StereoSGBM objects
    stereo_bm = cv2.StereoBM_create()
    stereo_bm.setNumDisparities(128)
    stereo_bm.setBlockSize(21)
    stereo_bm.setSpeckleRange(1)
    stereo_bm.setSpeckleWindowSize(100)
    stereo_bm.setMinDisparity(16)

    stereo_sgbm = cv2.StereoSGBM_create(minDisparity=16,
                                    numDisparities=128,
                                    blockSize=23,
                                    speckleWindowSize=100,
                                    speckleRange=1)

    # Compute disparity maps
    disparity_bm = stereo_bm.compute(stereo_l, stereo_r).astype(np.float32) / 4.0
    disparity_sgbm = stereo_sgbm.compute(stereo_l, stereo_r).astype(np.float32) / 16.0

    # Normalize and apply color maps
    disparity_bm = cv2.normalize(disparity_bm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity_bm_heatmap = cv2.applyColorMap(disparity_bm, cv2.COLORMAP_JET)

    disparity_sgbm = cv2.normalize(disparity_sgbm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity_sgbm_heatmap = cv2.applyColorMap(disparity_sgbm, cv2.COLORMAP_JET)

    # Compute Laplacian
    laplacian = cv2.Laplacian(stereo_l, cv2.CV_64F)

    # Convert to absolute values
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    # Normalize the Laplacian
    minVal, maxVal, _, _ = cv2.minMaxLoc(laplacian_abs)
    laplacian_normalized = (laplacian_abs - minVal) / (maxVal - minVal) * 255
    laplacian_display = laplacian_normalized.astype(np.uint8)

    # Apply histogram equalization to enhance edges
    laplacian_equalized = cv2.equalizeHist(laplacian_display)


    # Compute Sobel filters
    sobel_x = cv2.Sobel(stereo_l, cv2.CV_64F, 1, 0, ksize=7)
    sobel_y = cv2.Sobel(stereo_l, cv2.CV_64F, 0, 1, ksize=7)
    sobel_combined = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    axs[0, 0].imshow(stereo_l, 'gray')
    axs[0, 0].set_title("Left Image")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(stereo_r, 'gray')
    axs[0, 1].set_title("Right Image")
    axs[0, 1].axis('off')

    axs[0, 2].imshow(disparity_bm, 'gray')
    axs[0, 2].set_title("Disparity Map (StereoBM)")
    axs[0, 2].axis('off')

    axs[1, 0].imshow(cv2.cvtColor(disparity_bm_heatmap, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("Heatmap (StereoBM)")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(cv2.cvtColor(disparity_sgbm_heatmap, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title("Heatmap (StereoSGBM)")
    axs[1, 1].axis('off')

    axs[1, 2].imshow(laplacian_equalized, 'gray')
    axs[1, 2].set_title("Laplacian")
    axs[1, 2].axis('off')

    axs[2, 0].imshow(sobel_x, 'gray')
    axs[2, 0].set_title("Sobel X")
    axs[2, 0].axis('off')

    axs[2, 1].imshow(sobel_y, 'gray')
    axs[2, 1].set_title("Sobel Y")
    axs[2, 1].axis('off')

    axs[2, 2].imshow(sobel_combined, 'gray')
    axs[2, 2].set_title("Combined Sobel")
    axs[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig('image_analysis_results.png') # Save the figure to a file


def DPTimageProcessor(img_path):
    # DPTImageProcessor is an advanced image processor for DPT models. It is used to prepare the image for the model.
    url = img_path
    image = Image.open(url).convert('RGB') # Make sure the image is in RGB mode

    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    size = (640, 480) # You can adjust this size according to your needs
    resized_image = image.resize(size)

    # prepare image for the model
    inputs = processor(images=[resized_image], return_tensors="pt")  # Wrap the image in a list

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)

    depth_array = np.array(depth) # Convert the PIL Image to a NumPy array
    laplacian = cv2.Laplacian(depth_array, cv2.CV_64F)

    # Firstly apply the sobel filter to our original image

    # horizontal
    sobelx_original = cv2.Sobel(depth_array, cv2.CV_64F, 1, 0, ksize=5)

    # vertical
    sobely_original = cv2.Sobel(depth_array, cv2.CV_64F, 0, 1, ksize=5)

    sobelxy_original = np.sqrt(np.square(sobelx_original) + np.square(sobely_original))

    # Normalize the result to 0-255
    sobelxy_original = cv2.normalize(sobelxy_original, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Show the original image
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].axis('off')
    axs[0, 0].set_title("Original Image")

    # Show the depth image
    axs[0, 1].imshow(depth, cmap='gray')
    axs[0, 1].axis('off')
    axs[0, 1].set_title("Depth Image")

    # Show the Laplacian image
    laplacian = cv2.Laplacian(depth_array, cv2.CV_64F)
    axs[1, 0].imshow(laplacian.astype(np.uint8), 'gray')
    axs[1, 0].axis('off')
    axs[1, 0].set_title("Laplacian")

    # Show the Sobel combined image
    sobelx_original = cv2.Sobel(depth_array, cv2.CV_64F, 1, 0, ksize=5)
    sobely_original = cv2.Sobel(depth_array, cv2.CV_64F, 0, 1, ksize=5)
    sobelxy_original = np.sqrt(np.square(sobelx_original) + np.square(sobely_original))
    sobelxy_original = cv2.normalize(sobelxy_original, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    axs[1, 1].imshow(sobelxy_original, 'gray')
    axs[1, 1].axis('off')
    axs[1, 1].set_title("Sobel Filter")

    plt.tight_layout()
    plt.show()


def DPTOFObject(videopath):
    # Load Intel's DPT model
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    size = (640, 480)

    # Open the video
    video_path = videopath
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert the frame to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Resize the image
            resized_image = pil_image.resize(size)

            # Prepare image for the model
            inputs = processor(images=[resized_image], return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            )

            # Visualize the prediction
            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            depth_frame = cv2.cvtColor(formatted, cv2.COLOR_GRAY2BGR)

            # Show both original and depth frames
            cv2.imshow('Original', frame)
            cv2.imshow('Depth Estimation', depth_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# calling main function

if __name__ == "__main__":
    while True:
        print("\nSelect an option:")
        print("1: Draw 3D Lines")
        print("2: Draw Edges")
        print("3: Draw 3D Visualization Object")
        print("4: Disparity Map using StereoBM")
        print("5: Image Analysis")
        print("6: DPT Image Processor")
        print("7: DPT Object Detection")
        print("Press 'q' to exit.")

        user_option = input("Enter the option number: ")

        if user_option == 'q':
            print("Exiting the program. Goodbye!")
            exit()
        else:
            user_option = int(user_option)
            if user_option == 1:
                video_path = input("Enter the video path: ")
                Draw3DLines(video_path)
            elif user_option == 2:
                image_path = input("Enter the image path: ")
                DrawEdges(image_path)
            elif user_option == 3:
                video_path = input("Enter the video path: ")
                Draw3DVizObject(video_path)
            elif user_option == 4:
                video_path = input("Enter the video path: ")
                DisparityMap_StereoBM(video_path)
            elif user_option == 5:
                image_path1 = input("Enter the left image path: ")
                image_path2 = input("Enter the right image path: ")
                ImageAnalysis(image_path1, image_path2)
            elif user_option == 6:
                image_path = input("Enter the image path: ")
                DPTimageProcessor(image_path)
            elif user_option == 7:
                video_path = input("Enter the video path: ")
                DPTOFObject(video_path)
            else:
                print("Invalid option selected. Please try again.")

