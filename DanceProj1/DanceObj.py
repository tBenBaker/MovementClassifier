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
        self.window_size = None       
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
        self.features['sacrum_std'] = sacrumstd
        #average absolute value of jerk of sacrum, avg across 3 dimensions
        sacrumjerkmag = np.linalg.norm(sacrumjer, axis=1)
        sacrumjerkmag = np.mean(sacrumjerkmag)
        self.features['sacrum_jerkiness'] = sacrumjerkmag
        #get it just in y dimension
        sacrumjerky = sacrumjer[:,1]
        sacrumjerky = np.mean(sacrumjerky)
        self.features['sacrum_bounciness'] = sacrumjerky

    def get_wrist_ankle(self, sparse=False):

        #wrist and ankle acceleration
        Lwrist, Rwrist = [7,8] 
        Lankle, Rankle = [13,14]
        
        wristacc = (np.abs(self.acceleration[Lwrist]) + np.abs(self.acceleration[Rwrist])) / 2   
        ankleacc = (np.abs(self.acceleration[Lankle]) + np.abs(self.acceleration[Rankle])) / 2  

        self.features['wrist_acceleration'] = wristacc.mean()
        self.features['wrist_accel_std'] = wristacc.std()           
        self.features['ankle_acceleration'] = ankleacc.mean()
        self.features['ankle_accel_std'] = ankleacc.std()
        
        #ankle height
        #compute lowest point of ankle
        floor = min(self.pos[Rankle][:,1]) 
        ankleheight = (self.pos[Lankle][:,1] + self.pos[Rankle][:,1] / 2) - floor
        ankleheightm = ankleheight.mean()
        
        self.features['ankle_height'] = ankleheightm
        self.features['ankle_height_std'] = ankleheight.std() #how much is the dancer lifting their feet

    def get_angularmomentum(self, sparse=False):
        #angular momentum of each joint, summed over all joints
        #sparse = True returns minimal array of features 
        
        self.get_sacrum()
        angmom = np.empty_like(self.pos[0])
        
        for j in range(self.numjoints):
                angmom = np.cross(self.pos[j] - self.sacrum[0], np.abs(self.velocity[j]))
        angmomm = angmom.mean()
        
        if sparse==True:
            self.features['angularmomentum'] = angmomm
            self.features['angularmomentum_std'] = angmomm.std()

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


    def get_rhythmreg(self, sparse=False):
        # Ensure sacrum and angular momentum are calculated
        self.get_sacrum()

        # Initialize storage for autocorrelations in each dimension
        corrsx = np.empty((self.numjoints, 2*self.numframes-1))
        corrsy = np.empty_like(corrsx)
        corrsz = np.empty_like(corrsx)

        # Calculate autocorrelations for each dimension of each joint's angular momentum
        for j in range(self.numjoints):
            angmom_x = np.cross(self.pos[j] - self.sacrum[0], np.abs(self.velocity[j]))[:, 0]
            angmom_y = np.cross(self.pos[j] - self.sacrum[0], np.abs(self.velocity[j]))[:, 1]
            angmom_z = np.cross(self.pos[j] - self.sacrum[0], np.abs(self.velocity[j]))[:, 2]

            corrsx[j] = np.correlate(angmom_x, angmom_x, mode='full')
            corrsy[j] = np.correlate(angmom_y, angmom_y, mode='full')
            corrsz[j] = np.correlate(angmom_z, angmom_z, mode='full')

        # we average the autocorrelations across all joints before peak detection
        avg_corrsx = np.mean(corrsx, axis=0)
        avg_corrsy = np.mean(corrsy, axis=0)
        avg_corrsz = np.mean(corrsz, axis=0)

        # Find peaks for each averaged autocorrelation
        xpeaks, properties = find_peaks(avg_corrsx, height=0, distance=30, prominence=500, width=10)
        xapeaks, xaproperties = find_peaks(-avg_corrsx, height=0, distance=30, prominence=500, width=10)
        ypeaks, yproperties = find_peaks(avg_corrsy, height=0, distance=30, prominence=500, width=10)
        yapeaks, yaproperties = find_peaks(-avg_corrsy, height=0, distance=30, prominence=500, width=10)
        zpeaks, zproperties = find_peaks(avg_corrsz, height=0, distance=30, prominence=500, width=10)
        zapeaks, zaproperties = find_peaks(-avg_corrsz, height=0, distance=30, prominence=500, width=10)

        # Calculate "peakiness" as number of peaks normalized by the number of frames
        peakiness_x = (len(xpeaks) + len(xapeaks)) / self.numframes
        peakiness_y = (len(ypeaks) + len(yapeaks)) / self.numframes
        peakiness_z = (len(zpeaks) + len(zapeaks)) / self.numframes

        # Store the features
        self.features['rhythmregx'] = peakiness_x
        self.features['rhythmregy'] = peakiness_y
        self.features['rhythmregz'] = peakiness_z

    
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

        self.features['expandedness'] = expa.sum()/self.numframes
        self.features['expandedness_std'] = expa.std()


    def get_features(self, sparse=False):
        self.features['id'] = self.id
        self.features['Genre'] = self.genre
        #self.features['window size'] = self.window_size
        self.get_movedata()
        self.get_sacrum()
        self.get_angularmomentum(sparse=sparse)
        self.get_rhythmreg(sparse=sparse)
        self.get_wrist_ankle(sparse=sparse)
        self.get_expandedness(sparse=sparse)
        
        
     
        
        

        
