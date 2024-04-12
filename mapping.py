import cv2

reference_img = cv2.imread("D:/ProjectDataset/OwnDataset/Mango/Train/Bacterial Canker/IMG_20211106_120700 (Custom) (2).jpg")
dataset_img = cv2.imread("D:/ProjectDataset/OwnDataset/Mango/Train/Bacterial Canker/IMG_20211106_121111 (Custom) (2).jpg")

# Check if 2 images are equal.
if reference_img.shape == dataset_img.shape:
    print("The images have the same size and channels")
    difference = cv2.subtract(reference_img, dataset_img)
    b, g, r = cv2.split(difference)
    
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("The images are completely equal.")
    else:
        print("The images are not equal.")

sift = cv2.xfeatures2d.SIFT_create() 
kp_1, desc_1 = sift.detectAndCompute(reference_img, None)
kp_2, desc_2 = sift.detectAndCompute(dataset_img, None)

index_params = dict(algorithm=0, tree=5)
searchParams = dict()
flann = cv2.FlannBasedMatcher(index_params, searchParams)

matches = flann.knnMatch(desc_1, desc_2, k=2)

good_points = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good_points.append(m)

print("Number of good points:", len(good_points))

# Calculate the percentage of similarity
percentage_similarity = (len(kp_1) / len(kp_2)) * 100
print("Percentage similarity:", percentage_similarity)

result = cv2.drawMatches(reference_img, kp_1, dataset_img, kp_2, good_points, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("result", result)
cv2.imshow("Reference", reference_img)
cv2.imshow("Dataset", dataset_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
