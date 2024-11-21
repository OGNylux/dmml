import numpy as np

class WhiteNoise():
    def __init__(self, variance = 1):
        self.variance = variance
        
    def sample(self):
        return np.random.uniform(-self.variance, self.variance)
    
    def gen_samples(self, n_seq = 1, seq_length = 30):
        seq = []
        
        for n in range(n_seq):
            seq.append(np.array([self.sample() for i in range(seq_length)]))
        
        seq = np.array(seq)
        return seq.reshape((n_seq, (seq_length)))

def wn(n_seq = 1, seq_length = 30):
    proc = WhiteNoise()
    return proc.gen_samples(n_seq, seq_length)


class MovingAverage():
    def __init__(self, weights = [0.25, 0.25, 0.25, 0.25]):
        self.history = []
        self.weights = weights
        self.wn = WhiteNoise()
        
    def sample(self):        
        wn_value = self.wn.sample()
        new_sample = sum([h*w for h, w in zip(reversed(self.history), reversed(self.weights))])
        
        self.history.append(wn_value)
        
        if len(self.history) > len(self.weights):
            self.history.pop(0)
        
        return new_sample
    
    def gen_samples(self, n_seq = 2, seq_length = 30):
        seq = []
        
        for n in range(n_seq):
            self.history = []
            seq.append([self.sample() for i in range(seq_length)])
        
        seq = np.array(seq)
        return seq.reshape((n_seq, seq_length))

def ma(n_seq = 1, seq_length = 30):
    proc = MovingAverage()
    return proc.gen_samples(n_seq, seq_length)


class AutoRegressive():
    def __init__(self, weights = [0.25, 0.25, 0.25, 0.25]):
        self.history = []
        self.weights = weights
        self.wn = WhiteNoise()
        
    def sample(self):
        new_value = self.wn.sample()
        
        new_sample = sum([h*w for h, w in zip(reversed(self.history), reversed(self.weights))])
        new_sample += new_value
        self.history.append(new_sample)
        
        if len(self.history) > len(self.weights):
            self.history.pop(0)
            
        return new_sample
    
    def gen_samples(self, n_seq = 2, seq_length = 30):
        seq = []
        
        for n in range(n_seq):
            self.history = []
            seq.append([self.sample() for i in range(seq_length)])
        
        seq = np.array(seq)
        return seq.reshape((n_seq, seq_length))

def ar(n_seq = 1, seq_length = 30):
    proc = AutoRegressive()
    return proc.gen_samples(n_seq, seq_length)


class AutoRegressiveMovingAverage():
    def __init__(self, ar_weights = [0.25, 0.25, 0.25, 0.25], ma_weights = [0.25, 0.25, 0.25, 0.25]):
        self.history_ar = []
        self.history_ma = []
        self.weights_ar = ar_weights
        self.weights_ma = ma_weights
        self.wn = WhiteNoise()
        
    def sample(self):
        wn_value = self.wn.sample()
        self.history_ma.append(wn_value)
        
        new_sample = sum([h*w for h, w in zip(reversed(self.history_ar), reversed(self.weights_ar))])
        new_sample += sum([h*w for h, w in zip(reversed(self.history_ma), reversed(self.weights_ma))])
        new_sample += wn_value
        self.history_ar.append(new_sample)
        
        if len(self.history_ar) > len(self.weights_ar):
            self.history_ar.pop(0)
        if len(self.history_ma) > len(self.weights_ma):
            self.history_ma.pop(0)
        
        return new_sample
    
    def gen_samples(self, n_seq = 2, seq_length = 30):
        seq = []
        
        for n in range(n_seq):
            self.history_ar = []
            self.history_ma = []
            seq.append([self.sample() for i in range(seq_length)])
        
        seq = np.array(seq)
        return seq.reshape((n_seq, seq_length))

def arma(n_seq = 1, seq_length = 30):
    proc = AutoRegressiveMovingAverage()
    return proc.gen_samples(n_seq, seq_length)
