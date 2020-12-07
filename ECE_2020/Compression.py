

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import math
import contextlib
from pylab import*
from scipy.io import wavfile
import pyaudio
import wave,sys
import numpy as numpy
import matplotlib.pyplot as plt

fname = r'C:\Users\91910\Desktop\ECE\r.wav'
outname = 'filtered.wav'

cutOffFrequency = 1000.0


def fft_dis(fname):
    sampFreq, snd = wavfile.read(fname)

    snd = snd / (2.**15) #convert sound array to float pt. values

    #s1 = snd[:,0] #left channel
    print(snd)
    #s2 = snd[:,1] #right channel

    n = len(snd)
    p = fft(snd) # take the fourier transform of left channel

    #m = len(s2) 
    #p2 = fft(s2) # take the fourier transform of right channel

    nUniquePts = int(ceil((n+1)/2.0))
    p = p[0:nUniquePts]
    p = abs(p)

    #mUniquePts = int(ceil((m+1)/2.0))
    #p2 = p2[0:mUniquePts]
    #p2 = abs(p2)
    
    p = p / float(n) # scale by the number of points so that
             # the magnitude does not depend on the length 
             # of the signal or on its sampling frequency  
    p = p**2  # square it to get the power 
# multiply by two (see technical document for details)
# odd nfft excludes Nyquist point
    if n % 2 > 0: # we've got odd number of points fft
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

    freqArray = arange(0, nUniquePts, 1.0) * (sampFreq / n);
    #plt.plot(freqArray/1000, 10*log10(p), color='k')
   # plt.xlabel('Channel_Frequency (kHz)')
  #  plt.ylabel('Channel_Power (dB)')
  #  plt.show()

def run_mean(x, windowSize):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize


def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

    if sample_width == 1:
        dtype = np.uint8 # unsigned char
    elif sample_width == 2:
        dtype = np.int16 # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.fromstring(raw_bytes, dtype=dtype)
    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels
def converter():
    with contextlib.closing(wave.open(fname,'rb')) as spf:
        sampleRate = spf.getframerate()
        ampWidth = spf.getsampwidth()
        nChannels = spf.getnchannels()
        nFrames = spf.getnframes()
    
        # Extract Raw Audio from multi-channel Wav File
        signal = spf.readframes(nFrames*nChannels)
        spf.close()
        channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)
    
        # get window size
        fqRatio = (cutOffFrequency/sampleRate)
        N = int(math.sqrt(0.196196 + fqRatio**2)/fqRatio)
    
        # Use moviung average (only on first channel)
        filt = run_mean(channels[0], N).astype(channels.dtype)
    
        wav_file = wave.open(outname, "w")
        wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
        wav_file.writeframes(filt.tobytes('C'))
        wav_file.close()
    
    
    n = 0
    for n in range (0,2): 
        if n==0:
            fft_dis(fname)
        elif n==1:
            fft_dis(outname)
converter()
def visualize(path: str): 
	raw = wave.open(path) 
	
	signal = raw.readframes(-1) 
	signal = np.frombuffer(signal, dtype ="int16") 
	
	f_rate = raw.getframerate() 

	time = np.linspace( 
		0, # start 
		len(signal) / f_rate, 
		num = len(signal) )
	 

	plt.figure(1) 
	
	plt.title("Sound Wave") 
	
	plt.xlabel("Time") 
	
	plt.plot(time, signal) 
	
	plt.show() 
 

path_output=r'C:\Users\91910\Desktop\ECE\filtered.wav'

path_input=r'C:\Users\91910\Desktop\ECE\r.wav'

visualize(path_input)
visualize(path_output)

