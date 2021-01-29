
"""
Whenever a jedi needs help, star detector will be there to navigate them to their homeland
May the force be with you.
Author -- Ahmet Kumaş -- 29.01.2021

"""
import cv2
import numpy as np
import imutils
import time


class StarDetector():
    def __init__(self):
        # Constructor
        self.KAZE = self.load_model()

    def load_model(self):
        # Initiate KAZE
        return cv2.KAZE_create()

    def imshow(self, img):
        # Visualize result
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_matches(self, image1, keypoints1, image2,
                     keypoints2, good_matches):
        # Draw features for both images
        output = cv2.drawMatches(img1=image1,
                                 keypoints1=keypoints1,
                                 img2=image2,
                                 keypoints2=keypoints2,
                                 matches1to2=good_matches,
                                 outImg=None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        self.imshow(output)

    def process(self, galaxy_path, star_path):
        # GALAXY --> image1
        image1 = cv2.imread(galaxy_path, 0)

        # Stars --> image2
        image2 = cv2.imread(star_path, 0)

        # Calculate features
        keypoints1, descriptors1 = self.KAZE.detectAndCompute(image1, None)
        keypoints2, descriptors2 = self.KAZE.detectAndCompute(image2, None)

        FLANN_INDEX_KDTREE = 1
        INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE,
                            trees=5)
        SEARCH_PARAMS = dict(checks=100)

        # Convert to float32
        descriptors1 = np.float32(descriptors1)
        descriptors2 = np.float32(descriptors2)

        FLANN = cv2.FlannBasedMatcher(indexParams=INDEX_PARAMS,
                                      searchParams=SEARCH_PARAMS)

        # Match vectors using FLANN
        matches = FLANN.knnMatch(queryDescriptors=descriptors1,
                                 trainDescriptors=descriptors2,
                                 k=2)
        RATIO_THRESH = 0.4
        good_matches = []

        # Filter matches
        for m, n in matches:
            if m.distance < RATIO_THRESH * n.distance:
                good_matches.append(m)
        print("Number of features:", len(good_matches))

        if len(good_matches):
            src_pts = np.float32(
                [keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                return {"output": None,
                        "points": {"min_x": None,
                                   "min_y": None,
                                   "max_x": None,
                                   "max_y": None}}

            matchesMask = mask.ravel().tolist()
            h, w = image2.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                              [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            image1 = cv2.polylines(
                image1, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            # find top left - bottom right points
            pts = src_pts[mask == 1]
            min_x, min_y = np.int32(pts.min(axis=0))
            max_x, max_y = np.int32(pts.max(axis=0))

            # Draw rectangle
            cv2.rectangle(image1, (min_x, min_y), (max_x, max_y), 255, 1)
            cv2.putText(image1, 'Home*', (min_x, min_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            image1 = imutils.resize(image1, width=800)
            #self.draw_matches(image1, keypoints1, image2, keypoints2, good_matches)
            return {"output": image1,
                    "points": {"min_x": min_x,
                               "min_y": min_y,
                               "max_x": max_x,
                               "max_y": max_y}}

        print("No good match found.")
        return {"output": None,
                "points": {"min_x": None,
                           "min_y": None,
                           "max_x": None,
                           "max_y": None}}


if __name__ == '__main__':
    SD = StarDetector()
    start = time.time()
    # overall process time is less than 2 seconds except for loading images
    # process args(galaxy, star)
    output_dict = SD.process('T5m1VBXg.png', 'hWaJOs4w.png')
    end = time.time()

    print("Points: ", output_dict["points"])
    print("Process time:", round(end - start, 2))
    if output_dict["output"] is not None:
        cv2.imwrite("way_to_home.jpg", output_dict["output"])
        SD.imshow(output_dict["output"])
