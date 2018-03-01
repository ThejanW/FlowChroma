from os import listdir
import cv2
from os.path import join, isfile, isdir


class VideoResizer:

    def __init__(self,source_dir,dest_dir):
        self.source_dir = source_dir
        self.dest_dir = dest_dir

    def resize_video(self,input_file,output_file,size):
        video = cv2.VideoCapture(input_file)
        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, fps , size)

        while video.isOpened():
            ret,frame = video.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame,size)
            out.write(resized_frame)
        video.release()
        out.release()

    def resize_all(self,size):
        for file_name in listdir(self.source_dir):
            input_file = join(self.source_dir,file_name)
            if isfile(input_file):
                self.resize_video(input_file,join(self.dest_dir,file_name.replace(".mp4",".avi")),size)



if __name__ == '__main__':
    video_resizer = VideoResizer('/home/chamath/Projects/FlowChroma/dataset/in','/home/chamath/Projects/FlowChroma/dataset/out')
    video_resizer.resize_all((244,244))
