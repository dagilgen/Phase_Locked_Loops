'''
This script serves for testing and experimental purposes only

Created on 22.02.2014

@authors: Daniel Gilgen, Fabio Marti
'''
import numpy
import pylab
import ctypes           # Currently only works for Windows OS
from matplotlib import rc


def main():

    # Get screen resolution (Only works with Windows OS)
#     screen = ctypes.windll.user32
#     height = screen.GetSystemMetrics(1)
#     width = screen.GetSystemMetrics(0)
    height = 500
    width = 500
    
    # Enable Latex functionality and fonts
    rc('text',usetex=True)
    rc('font',**{'family':'serif','serif':['Computer Modern']})
    
    # Parameters of exponential decreasing sinusoid
    nOfSamples = 200 
    phase = numpy.pi/4
    frequency = 1
    omega = 2*numpy.pi*frequency
    t = numpy.linspace(2*numpy.pi,0,nOfSamples)
    
    finalAmplitude = 2
    decreaseFactor = 1/1.01
    powers = numpy.arange(nOfSamples)
    amplitude = finalAmplitude*numpy.power(decreaseFactor,powers)
    
    # Computing sinusoid and noise
    samples = amplitude*numpy.sin(omega*t+phase)
    noise = numpy.random.randn(nOfSamples)
    
    # Adjust figure and figure size
    frontcolor = pylab.array([0.8,0.8,0.8])    # Represents bright gray
    dpi = 80
    pylab.figure(num=1, figsize=(width/dpi*0.8, height/dpi*0.8), dpi=dpi, facecolor=frontcolor, edgecolor='k') 
    
     
    # Plot graphs
    plotcolor = pylab.array([0.5,0.5,1])        # Represents bright blue
    pylab.subplot(2,2,1)
    pylab.title('Noisy sinusoid with $\sigma^2=0$')
    pylab.plot(t,samples,color=plotcolor)
#     
    pylab.subplot(2,2,2)
    pylab.title('Noisy sinusoid with $\sigma^2=1/16$')
    plot2 = pylab.plot(t,samples+noise/4,color=plotcolor)
    pylab.plot(t,samples,color='k')
     
    pylab.subplot(2,2,3)
    pylab.title('Noisy sinusoid with $\sigma^2=1/4$')
    plot3 = pylab.plot(t,samples+noise/2,color=plotcolor)
    pylab.plot(t,samples,color='k')
     
    pylab.subplot(2,2,4)
    pylab.title('Noisy sinusoid with $\sigma^2=1$')
    plot4 = pylab.plot(t,samples+noise,color=plotcolor)
    pylab.plot(t,samples,color='k')
    
    pylab.show()

if __name__ == '__main__':
    
    main()
    
