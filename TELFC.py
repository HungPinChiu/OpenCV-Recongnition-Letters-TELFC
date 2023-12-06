import cv2
import numpy as np

class CounterRecognition(object):
    def __init__(self, img):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.contours = []
        self.hierarchy = []
        self.binary_img = None

    def get_binary_img(self, lower=(0, 0, 0), upper=(180, 255, 70)):
        self.binary_img = cv2.inRange(self.img, np.array(lower), np.array(upper))
        return self.binary_img
    
    def get_contours(self):
        self.contours, self.hierarchy = cv2.findContours(self.binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.img, self.contours, -1, (0, 255, 0), 3)
        return self.contours, self.hierarchy

    def distinguish_contour(self, img):
        for i, contour in enumerate(self.contours):
            if cv2.contourArea(contour) > 5000:
                child_index = self.hierarchy[0][i][2]
                sides = 0
                sum = 0
                while child_index != -1:
                    child_contour = self.contours[child_index]
                    if cv2.contourArea(child_contour) > 300:
                        epsilon = 0.012 * cv2.arcLength(child_contour, True)
                        approx = cv2.approxPolyDP(child_contour, epsilon, True)
                        sides = len(approx)
                        sum += sides
                        #cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
                    child_index = self.hierarchy[0][child_index][0]
                #print(length)
                self.distinguish_sides(img, contour, sides, sum)
        return img
    
    def distinguish_sides(self, img, contour, sides, sum):
        x,y,w,h = cv2.boundingRect(contour)
        if 6 <= sum <= 7:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3)
            cv2.putText(img, "L", (x-10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        elif 8 <= sum <= 9:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img, "T", (x-10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif 10 <= sum <= 11:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),3)
            cv2.putText(img, "F", (x-10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        elif sum >= 13:
            if sides == sum:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)
                cv2.putText(img, "C", (x-10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                cv2.putText(img, "E", (x-10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return img
    
if __name__ == "__main__":
    cap = cv2.VideoCapture('C:/Users/User/Desktop/OpenCV/input.mov')
    # VideoWriter
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('C:/Users/User/Desktop/OpenCV/area.mov', fourcc, 120, (int(cap.get(3)),int(cap.get(4))))

    while True:
        ret, img = cap.read()
        if not ret:
            break
        # Class
        frame = CounterRecognition(img)
        binary_img = frame.get_binary_img()
        contours = frame.get_contours()
        # Show
        cv2.imshow("Img", img)
        #cv2.imshow("Binary", binary_img)
        cv2.imshow("Contours", frame.distinguish_contour(img))
        # Output with video
        img_result = frame.distinguish_contour(img)
        out.write(img_result)
        out.write(img)
        if cv2.waitKey(10) == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
