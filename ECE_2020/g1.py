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