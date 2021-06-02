
from custom_functions import *


cap = cv2.VideoCapture(0)

while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == True:
		frame = cv2.resize(frame, (352, 288))
		frame_main = frame.copy()
		frame_, results = object_detection(frame)

		lines, detection_count = object_tracker(results) 
		print(' No. of object detected', detection_count)

		colors = [(0,255,0), (0,255,255), (255,0,0), (255,255,0), (255,0,255), (0, 255,255)] 
		for key, value in lines.items():
			i = list(lines.keys()).index(key)
			value = np.array(value)
			frame_ = cv2.polylines(frame_, [value], isClosed=False, color=colors[i],thickness=1)

		cv2.imshow('Entire frame', frame_)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
			
	else:
		break


cap.release()
cv2.destroyAllWindows()




