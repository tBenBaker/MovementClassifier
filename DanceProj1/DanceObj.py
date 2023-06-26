import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, peak_widths
from scipy.signal import savgol_filter


#note aist indexes = 0nose, 1L-ear, 2R-ear, 3L-shoulder, 4R-shoulder, 5L-elbow, 6R-elbow, 7L-wrist, 
# 8R-wrist, 9L-hip, 10R-hip, 11L-knee, 12R-knee, 13L-ankle, 14R-ankle 


class Dance:
    
    def __init__(self, pos, dt):        #pos is a 3d array of joint positions, dt is the time between frames
        self.pos = pos
        self.numjoints = len(self.pos)
        self.dt = dt
        self.frames = range(len(pos[0]))
        self.numframes = len(self.frames)
        self.times = [dt * f for f in self.frames]
        self.moment = 1/dt * 1/4
        self.moments = np.arange(self.moment, self.numframes, self.moment, dtype=int) 
        #quarter-second intervals across frames, for measuring feature stats across moments
        
        self.id = []        #initialize variables to be set below
        self.genre = []       
        self.velocity = []
        self.acceleration = []
        self.jerk = []
        self.movedata = []
        self.sacrum = []
        self.features = {} #initialize dictionary of features

    @staticmethod     
    def smooth_derivative(data, dt):
        derivative = (data[:, 1:] - data[:, :-1]) / dt
        smoothed_derivative = np.empty_like(derivative)

        for index, joint in enumerate(derivative):
            for dim in range(3):
                smoothed_derivative[index, ..., dim] = savgol_filter(joint[:, dim], window_length=45, polyorder=2, mode='nearest')
        smoothed_derivative = np.pad(smoothed_derivative, ((0, 0), (0, 1), (0, 0)), mode='edge')
        
        return smoothed_derivative

    def get_movedata(self):
        # Calculate velocity, acceleration, and jerk using smooth_derivative

        vel = self.smooth_derivative(self.pos, self.dt)  # Calculate smoothed velocity
        acc = self.smooth_derivative(vel, self.dt)  # Calculate smoothed acceleration
        jerk = self.smooth_derivative(acc, self.dt)  # Calculate smoothed jerk
        
        self.velocity = vel
        self.acceleration = acc
        self.jerk = jerk
        self.movedata = [self.pos, self.velocity, self.acceleration, self.jerk]
      

    def get_sacrum(self, hipidxs=[9,10]):       #populate virtual sacrum joint by averaging hip positions.   
                                                                    #idxs of AIST are [9,10]
        Lhip, Rhip = hipidxs
        sacrumpos = np.empty_like(self.pos[0])
        
        for f in range(self.numframes):
            sacrumpos[f] = (self.pos[Lhip][f] + self.pos[Rhip][f]) / 2     
        
        sacrumvel = (sacrumpos[1:, :] - sacrumpos[:-1, :]) / self.dt
        for dim in range(3):                        
            sacrumvel[:, dim] = savgol_filter(sacrumvel[:,dim], window_length=45, polyorder=2, mode='nearest')  
        sacrumvel = np.pad(sacrumvel, ((0,1), (0,0)), mode='edge')   

        sacrumacc = (sacrumvel[1:,:] - sacrumvel[:-1,:]) / self.dt
        for dim in range(3):                        
            sacrumacc[:, dim] = savgol_filter(sacrumacc[:,dim], window_length=45, polyorder=2, mode='nearest')  
        sacrumacc = np.pad(sacrumacc, ((0,1), (0,0)), mode='edge')  

        sacrumjer = (sacrumacc[1:,:] - sacrumacc[:-1,:]) / self.dt
        for dim in range(3):                        
            sacrumjer[:, dim] = savgol_filter(sacrumjer[:,dim], window_length=45, polyorder=2, mode='nearest')  
        sacrumjer = np.pad(sacrumjer, ((0,1), (0,0)), mode='edge')
        
        self.sacrum = [sacrumpos, sacrumvel, sacrumacc, sacrumjer]  

        #standard deviation of sacrum position, avg across 3 dimensions
        sacrumstd = np.std(sacrumpos, axis=0)
        sacrumstd = np.mean(sacrumstd)
        self.features['sacrumstd'] = sacrumstd
        #average absolute value of jerk of sacrum, avg across 3 dimensions
        sacrumjerkmag = np.linalg.norm(sacrumjer, axis=1)
        sacrumjerkmag = np.mean(sacrumjerkmag)
        self.features['sacrumjerkmag'] = sacrumjerkmag
        #get it just in y dimension
        sacrumjerky = sacrumjer[:,1]
        sacrumjerky = np.mean(sacrumjerky)
        self.features['sacrumjerky'] = sacrumjerky

    def get_wrist_ankle_features(self, sparse=False):

        #wrist and ankle acceleration
        Lwrist, Rwrist = [7,8] 
        Lankle, Rankle = [13,14]
        
        wristacc = (np.abs(self.acceleration[Lwrist]) + np.abs(self.acceleration[Rwrist])) / 2   
        ankleacc = (np.abs(self.acceleration[Lankle]) + np.abs(self.acceleration[Rankle])) / 2  

        self.features['wristacceleration'] = wristacc.mean()
        self.features['wristaccstd'] = wristacc.std()           
        self.features['ankleacceleration'] = ankleacc.mean()
        self.features['ankleaccstd'] = ankleacc.std()
        
        #ankle height
        #compute lowest point of ankle
        floor = min(self.pos[Rankle][:,1]) 
        ankleheight = (self.pos[Lankle][:,1] + self.pos[Rankle][:,1] / 2) - floor
        ankleheightm = ankleheight.mean()
        
        self.features['ankleheight'] = ankleheightm
        self.features['ankleheightstd'] = ankleheight.std() #how much is the dancer lifting their feet

    def get_angularmomentum_features(self, sparse=False):
        #angular momentum of each joint, summed over all joints
        #sparse = True returns minimal array of features 
        
        self.get_sacrum()
        angmom = np.empty_like(self.pos[0])
        
        for j in range(self.numjoints):
                angmom = np.cross(self.pos[j] - self.sacrum[0], np.abs(self.velocity[j]))
        angmomm = angmom.mean()
        
        if sparse==True:
            self.features['angularmomentum'] = angmomm
            self.features['angularmomentumstd'] = angmomm.std()

            peaks, properties = find_peaks(angmom, height=0, distance=30, prominence=500, width=10)
            apeaks, aproperties = find_peaks(-angmom, height=0, distance=30, prominence=500, width=10)

            self.features['peaks'] = (len(peaks) + len(apeaks)) / self.numframes
        
        if sparse==False:

            self.features['angularmomentumxz'] = angmom[:,0].mean() + angmom[:,2].mean()  #spinny-ness in horizontal plane
            self.features['angularmomentumy'] = angmom[:,1].mean()
            self.features['angularmomentumxzstd'] = angmom[:,0].std() + angmom[:,2].std()
            self.features['angularmomentumystd'] = angmom[:,1].std()

            ypeaks, yproperties = find_peaks(angmom[:,1], height=0, distance=30, prominence=500, width=10)
            yapeaks, yaproperties = find_peaks(-angmom[:,1], height=0, distance=30, prominence=500, width=10)
            xzpeaks, xzproperties = find_peaks(angmom[:,0] + angmom[:,2], height=0, distance=30, prominence=500, width=10)
            xzapeaks, xzaproperties = find_peaks(-(angmom[:,0] + angmom[:,2]), height=0, distance=30, prominence=500, width=10)

            self.features['ypeaks'] = (len(ypeaks) + len(yapeaks)) / self.numframes
            self.features['xzpeaks'] = (len(xzpeaks) + len(xzapeaks)) / self.numframes


    def get_expandedness(self, sparse=False):   #expandedness ~ distance of joints from sacrum
        
        self.get_sacrum()
        
        Dsfromsacrum = np.empty_like(self.pos)  #inits for diff in pos, vel, and accel from sacrum
                
        for j in range(self.numjoints):                         #DISTANCE of each joint from sacrum                                    
            Dsfromsacrum[j] = np.abs((self.pos[j] - self.sacrum[0]))
        
        expa = Dsfromsacrum.sum(axis=0)                         #sum over joints to get expandedness per frame per dimension
        expa = expa.sum(axis=1)                                 #sum over dimensions to get expandedness over frames
        #calculate absolute value of jerk of expandedness
        expajerk = np.diff(expa, n=2, axis=0)
        expajerk = np.abs(expajerk)

        self.features['Expandedness'] = expa.sum()/self.numframes
        self.features['Expandedness_std'] = expa.std()


    def get_features(self, sparse=False):
        self.features['id'] = self.id
        self.features['Genre'] = self.genre
        self.get_movedata()
        self.get_sacrum()
        self.get_angularmomentum_features(sparse=sparse)
        self.get_wrist_ankle_features(sparse=sparse)
        self.get_expandedness(sparse=sparse)
        
        
     
        
        

        
