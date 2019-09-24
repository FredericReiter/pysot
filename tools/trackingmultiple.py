import os, glob, cv2

class Position:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)

    def return_positions(self):
        return str(self.xmin) + '-' + str(self.ymin) + '-' + str(self.xmax) + '-' + str(self.ymax)

    def get_x_y_w_h(self):
        return self.xmin, self.ymin, self.xmax - self.xmin, self.ymax - self.ymin
    def get_x_y_w_h_list(self):
        temp = [self.xmin, self.ymin, self.xmax - self.xmin, self.ymax - self.ymin]
        return [str(x) for x in temp]

class Frame:
    def __init__(self, startframe, stopframe):
        self.startframe = startframe
        self.stopframe = stopframe
        self.positions = []

class Controltool:
    videodict = {}
    videofilelist = []
    wd = 'D:/Inavet'

    def __init__(self):
        self.vidcap = None
        self.import_controldata()
        self.curvid = None

    def import_controldata(self):
        filelist = glob.glob(os.path.join(self.wd, '*.txt'))
        for file in filelist:
            with open(file, 'r') as trackinglistfile:
                data = trackinglistfile.read()
                data = [l.split('\n') for l in data.split('#')]
                data = [list(filter(None, x)) for x in data]
                data = [x for x in data if x]
                for timezone, videodata in enumerate(data): # videodata is the lines of file with information
                    if videodata[0].split('\n')[0] not in self.videodict:
                        self.videodict[videodata[0].split('\n')[0]] = []
                    self.videofilelist.append(videodata[0].split('\n')[0])
                    if 'D' not in videodata[1]:
                        self.videodict[videodata[0]].append(Frame(videodata[1].split('-')[0],videodata[1].split('-')[1]))
                        index = 2
                    else:
                        print('Alert')
                        raise Exception
                    while index < len(videodata):
                        if videodata[index]:
                            positions = videodata[index].split('-')
                            self.videodict[videodata[0]][-1].positions.append(Position(positions[0],positions[1],
                                                                                positions[2],positions[3]))
                            index +=1

                            #TODO multiple timeframes per videodata

    def setup_video(self, videopath, startframe, positions):
        try:
            self.vidcap = cv2.VideoCapture(videopath)
        except cv2.error:
            raise Exception
        if not self.vidcap.isOpened():
            raise Exception

    def get_frame_from_index(self, index):
        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        hasFrames, image = self.vidcap.read()
        if hasFrames:
            return image
        else:
            raise Exception
    
    def get_start_position(self):
        return self.curvid.positions # [0].get_x_y_w_h()
    
    def initialize(self, index):
        self.curvid = self.videodict[self.videofilelist[index]][0]
        self.setup_video(self.videofilelist[index], self.curvid.startframe, self.curvid.positions)

        return int(self.curvid.startframe), int(self.curvid.stopframe), os.path.basename(self.videofilelist[index])


""" if __name__ == "__main__":
    Instance = Controltool()
    print(Instance.initialize(0)) """
    