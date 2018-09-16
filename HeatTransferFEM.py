# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:06:29 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def HeatTransferFEM2D(X, Y, TempInitial,BoundaryCondition, NodeEquation, ElementNode,TimeVector,ProblemParameters):
    #Extract Problem Parameters
    VolHeatGen=ProblemParameters[2] #Volumetric Heat Generation [W/m^3]
    ThermalConductivity=ProblemParameters[0] #Thermal Conductivity [W/m/k]
    ThermalDiffusivity=ProblemParameters[1]  #Thermal Diffusivity[m^2/s]
    TemperatureAmbient=ProblemParameters[3]  #Ambient Temperature [C]
    
    #Extract Time Parameters
    if len(TimeVector)<=1:
        SteadyState = 1
        TimeLoop = 1
    else:
        SteadyState = 0
        TimeLoop = len(TimeVector)-1
    
    
    #Set up Temperature Matrix
    Temp = np.zeros((len(TempInitial),TimeLoop+1))
    Temp[:,0]=TempInitial.T
    for i in range(len(NodeEquation)):
        if NodeEquation[i] == 0:
            Temp[i][1:]= TempInitial[i]*np.ones((1,TimeLoop))


    
    #Heat Flux Parameters
    HeatFluxType=BoundaryCondition[:][0]
    HeatFluxValue=BoundaryCondition[:][1]
    
    
    # location matrix
    LocationMatrix = np.zeros((len(ElementNode),3))
    for i in range(len(ElementNode)):
        for j in range(len(ElementNode[0])):
            if NodeEquation[int(ElementNode[i][j])-1] != 0:
                LocationMatrix[i][j]= np.abs(NodeEquation[int(ElementNode[i][j])-1])
    
    #number of equations and elements

    NumberOfEquation = np.max(np.abs(NodeEquation))
    NumberOfElement  = len(ElementNode)
    
   
    for t in range(TimeLoop):
            # assemble system matrix and vector
            SystemMatrix = np.zeros((int(NumberOfEquation),int(NumberOfEquation)))
            SystemVector = np.zeros((int(NumberOfEquation),1))
            
            #Calculate TimeStep
            if SteadyState==0:
                TimeStep= TimeVector[t+1]-TimeVector[t]
                
            for n in range(NumberOfElement):
                
                # element coordinates
                XE=np.zeros((3,1))
                YE=np.zeros((3,1))
                for i in range(0,3):
                    XE[i]= X[int(ElementNode[n,i])-1]
                    YE[i]= Y[int(ElementNode[n,i])-1]
            
            # derivatives of shape functions
    
                Area = 0.5*((XE[1]-XE[0])*(YE[2]-YE[0])-(YE[1]-YE[0])*(XE[2]-XE[0]))
                
    
                NX = [YE[1]-YE[2], YE[2]-YE[0], YE[0]-YE[1]]/(2*Area)
                NY = [XE[2]-XE[1], XE[0]-XE[2], XE[1]-XE[0]]/(2*Area)
                
    
            # element matrix
            
                ElementMatrix = (NX*NX.T+NY*NY.T)*Area*ThermalConductivity
                
                
            # Add contribution due to Transient, Appromiate integral with point on the centriod
            
                if SteadyState==1:
                    TimeElementMatrix=np.zeros((3,3))
                else:
                    TimeElementMatrix=np.ones((3,3))*(1/9*Area)/TimeStep/ThermalDiffusivity*ThermalConductivity 
                    ElementMatrix =ElementMatrix + TimeElementMatrix
                
                ElementVector = 1/3*VolHeatGen*Area*np.ones((3,1))
                
                
            #Check if Element has any boundarys
                if min(NodeEquation[ElementNode[n][0:2]-1])<=0:
                #Evaluate Natural Boundarys first
                    for i in range(3):
                        i2= int(i+1-np.floor((i+1)/3)*3) #Determines next node. If i2(i=0)=1, i2(i=1)=2, i2(i=2)=0
                        NTest= [NodeEquation[int(ElementNode[n][i]-1)],NodeEquation[int(ElementNode[n][i2]-1)]]
                        
                        
                #1)Element will have eithter "Full" or "Half" Natural Boundary
                #Calculate Length of boundary
                        x=[X[int(ElementNode[n][i2]-1)],X[int(ElementNode[n][i]-1)]]
                        y=[Y[int(ElementNode[n][i2]-1)],Y[int(ElementNode[n][i]-1)]]
                        L=np.sqrt((x[0]-x[1])*(x[0]-x[1]) +(y[0]-y[1])*(y[0]-y[1])) #Length of Boundary 
                        
                #1a)Full Natural Boundary: Test if the two points are along the Natural BC
                        if NTest[0]<0 and NTest[1]<0:
                #Determine Average Effective Contribution from Natural Boundary
                #Determine Type of First Node
                            if HeatFluxType[ElementNode[n][i]-1]==1: #Constant
                                FluxValue1=HeatFluxValue[ElementNode[n][i]-1]     
                            elif HeatFluxType[ElementNode[n][i]-1]==2: #Convective
                                FluxValue1=-HeatFluxValue[ElementNode[n][i]-1]*TemperatureAmbient
                #Determine Type of Second Node
                            if HeatFluxType[ElementNode[n][i2]-1]==1: #Constant
                                FluxValue2=HeatFluxValue[ElementNode[n][i2]-1]     
                            elif HeatFluxType[ElementNode[n][i2]-1]==2: #Convective
                                FluxValue2=-HeatFluxValue[ElementNode[n][i2]-1]*TemperatureAmbient
                            AverageFlux= (FluxValue1+ FluxValue2)/2
                            ElementVector[i] = ElementVector[i] -  AverageFlux*L/2 
                            ElementVector[i2] = ElementVector[i2] - AverageFlux*L/2 
                            
                            
                            
                        #Adjust Element Matrix if The boundary is Convective
                            if HeatFluxType[ElementNode[n,i]-1]==2 and HeatFluxType[ElementNode[n,i2]-1]==2:
                                AverageCoefficient=(HeatFluxValue[ElementNode[n,i]-1] + HeatFluxValue[ElementNode[n,i2]-1])/2
                                #ConvectiveMatrixContribution= - L*AverageCoefficient/4
                                ElementMatrix[i,i]=ElementMatrix[i,i] + AverageCoefficient*(L/3)
                                ElementMatrix[i,i2]=ElementMatrix[i,i2] + AverageCoefficient*(L/6)
                                ElementMatrix[i2,i]=ElementMatrix[i2,i] + AverageCoefficient*(L/3)
                                ElementMatrix[i2,i2]=ElementMatrix[i2,i2] + AverageCoefficient*(L/6)
                                
                            
                        # 1b) Half Natural Boundary: 1 essential and 1 natural
                        if (NTest[0]==0 and NTest[1]<0) or (NTest[0]<0 and NTest[1]==0):
                            if NTest[0]==0 and NTest[1]<0:  #Natural on second node
                                ii=1
                            elif NTest[0]<0 and NTest[1]==0: #Natural on first node
                                ii=0          
                            if HeatFluxType[ElementNode[n,ii]-1]==2: #Convective
                                AverageFlux=-HeatFluxValue[ElementNode[n,ii]-1]*TemperatureAmbient     
                            elif HeatFluxType[ElementNode[n,ii]-1]==1: #Constant
                                AverageFlux=HeatFluxValue[ElementNode[n,ii]-1]
                            ElementVector[ii] = ElementVector[ii] + AverageFlux*L/2
                            
                  
                for i in range(3):
                    I=NodeEquation[ElementNode[n][i]-1]
                    if I==0:
                        ElementT = Temp[ElementNode[n,i]-1,t]*ElementMatrix[:,i]
                        ElementVector = ElementVector - ElementT.reshape(3,1)
                           
                        
                    #Insert Transient component to Element Vector

                ElementTV = TimeElementMatrix*Temp[ElementNode[n,:]-1,t]
                ElementVector = ElementVector + ElementTV[:][0].reshape(3,1)
                
                

                    #Local to Global Indexing
                for i in range(3):
                    ii = int(LocationMatrix[n][i])
                    if ii != 0:
                        SystemVector[ii-1] += ElementVector[i]

                        for j in range(3):
                            jj = int(LocationMatrix[n][j])
                            if jj != 0:
                                SystemMatrix[ii-1][jj-1] +=  ElementMatrix[i][j]
            
            
            #Solve system matrix equations and Assign the Solution
            SystemVariable = np.linalg.solve(SystemMatrix,SystemVector)

            # Assign the solution to T

            for i in range(len(NodeEquation)):
                if NodeEquation[i] != 0:
                    Temp[i][t+1] = SystemVariable[np.abs(int(NodeEquation[i]))-1]
                      
            #Temp[NodeEquation !=0][t+1] =  SystemVariable
            if SteadyState==1:
                Temp[:,1]=[];
                break
    return Temp


    
def main():
    #Geometric Parameters
    Length = 10 #m
    Height = 10 #m
    ElementSize = 0.2 #length of the element edge
    
    #Initial condition
    TempInitial = 0
    TimeVector = np.linspace(0,5,6)
    ThermalConductivity = 1 #Thermal Conductivity [W/m/k]
    ThermalDiffusivity= 1 #Thermal Diffusivity[m^2/s]
    VolHeatGen = 2   #Volumetric Heat Generation [W/m^3]
    AmbientAirTemp= 50  #Ambient Temperature [C]
    ProblemParameters=[ThermalConductivity, ThermalDiffusivity,VolHeatGen ,AmbientAirTemp]

    
    #boundary conditions
    
    # First column denotes the value and the second column denotes boundary type:
    # 0: Essential BC, Temperature in Celsius
    # 1: Natural BC, Constant Heat Flux , W/m^2
    # 2: Natural BC, Convective Heat Transfer, Coefficient in W/m^2/K
    # Each Row is a side: 1. Top, 2. Bottom, 3. Left, 4. Right
    BoundaryCondition = [[100,0],  # Top
                         [0,0],    #Bottom
                         [0,0],    #Left
                         [0,0]]   #Right
    
    # Number of elements
    NumberOfElementInX = int(2*(np.floor(0.5*Length/ElementSize)+1))
    NumberOfElementInY = int(np.ceil(Height/ElementSize))
    
    #Generate mash
    NumberOfNode    = (NumberOfElementInX+1)*(NumberOfElementInY+1)
    NumberOfElement = 2*NumberOfElementInX*NumberOfElementInY
    x = np.linspace(0, Length, NumberOfElementInX+1)
    y = np.linspace(0, Height, NumberOfElementInY+1)
    [X,Y] = np.meshgrid(x,y)
    X = X.T
    Y = Y.T
    X=np.reshape(X,-1,1)
    Y=np.reshape(Y,-1,1) 
    
     # element node connectivity

    ElementNode = np.zeros((NumberOfElement,3),dtype='int')

    for j in range(1,NumberOfElementInY+1):
        for i in range(1,NumberOfElementInX+1):
            n = (j-1)*NumberOfElementInX+i
            if i <= NumberOfElementInX/2:
                ElementNode[2*n-2, 0] = int((j-1)*(NumberOfElementInX+1)+i)
                ElementNode[2*n-2, 1] = int(j*(NumberOfElementInX+1)+i+1)
                ElementNode[2*n-2, 2] = int(j*(NumberOfElementInX+1)+i)
                ElementNode[2*n-1, 0]   = int((j-1)*(NumberOfElementInX+1)+i)
                ElementNode[2*n-1, 1]   = int((j-1)*(NumberOfElementInX+1)+i+1)
                ElementNode[2*n-1, 2]   = int(j*(NumberOfElementInX+1)+i+1)
            else:
                ElementNode[2*n-2, 0] = int((j-1)*(NumberOfElementInX+1)+i)
                ElementNode[2*n-2, 1] = int((j-1)*(NumberOfElementInX+1)+i+1)
                ElementNode[2*n-2, 2] = int(j*(NumberOfElementInX+1)+i)
                ElementNode[2*n-1, 0]   = int((j-1)*(NumberOfElementInX+1)+i+1)
                ElementNode[2*n-1, 1]   = int(j*(NumberOfElementInX+1)+i+1)
                ElementNode[2*n-1, 2]   = int(j*(NumberOfElementInX+1)+i)
    
    #boundary condition
    Temperature  = np.zeros((NumberOfNode,1))
    HeatFluxValue = np.zeros((NumberOfNode,1))
    HeatFluxType = np.zeros((NumberOfNode,1),dtype='int')
    NodeEquation = np.ones((NumberOfNode,1),dtype='int')
   
    
    # BCside == 1. Top, 2. Bottom, 3. Left, 4. Right    
    for j  in range(1,NumberOfElementInY+2):
        for i in range(1,NumberOfElementInX+2):
            n = (j-1)*(NumberOfElementInX+1)+i
            BCside=0
            if j == 1:
                BCside =2
            if j == NumberOfElementInY+1:
                BCside =1
            if i == 1:
                BCside =3
            if i == NumberOfElementInX+1:
                BCside =4
    
    #BCSide==0 Means that the node is in the center                        
            if BCside != 0:
                if BoundaryCondition[BCside-1][1] ==0:
                    HeatFluxType[n-1] = 0
                    HeatFluxValue[n-1] = 0
                    Temperature[n-1] = BoundaryCondition[BCside-1][0]
                    NodeEquation[n-1] = 0
                else:
                    HeatFluxType[n-1] = BoundaryCondition[BCside-1][1]
                    HeatFluxValue[n-1] = BoundaryCondition[BCside-1][0]
                    Temperature[n-1] = TempInitial
                    NodeEquation[n-1] = -1 
    HeatFlux=[HeatFluxType,HeatFluxValue]
    
    # number of equations and node corresponding equation
    NumberOfEquation,j = 0,1
    for i in range(0,len(NodeEquation)):
        if NodeEquation[i] == 1 or NodeEquation[i] == -1:
            NodeEquation[i]=j * NodeEquation[i]
            j+=1
            NumberOfEquation+=1

          
    # Solve temperature using the finite element method
    TemperatureTime = HeatTransferFEM2D(X, Y, Temperature, HeatFlux, NodeEquation, ElementNode,TimeVector,ProblemParameters)

    
   #plot & animation 
    fig = plt.figure()
    ax = plt.subplot(111)
    TemperatureDistribusion = TemperatureTime[:,0]
    center_temp = np.array([(x+y+z)/3.0 for x,y,z in TemperatureDistribusion[ElementNode-1]])
    field = ax.tripcolor(X, Y, ElementNode-1, center_temp,cmap=plt.cm.rainbow, edgecolors='k')
    plt.gca().set_aspect('equal')
    plt.title('FEM Solution')

    def make_frame(t):
        TemperatureDistribusion = TemperatureTime[:,t]
        center_temp = np.array([(x+y+z)/3.0 for x,y,z in TemperatureDistribusion[ElementNode-1]])
        field.set_array(center_temp)
        return field,
    
    ani=animation.FuncAnimation(fig,func=make_frame,frames=len(TimeVector),interval=500,blit=True)
    plt.show()
    
    
if __name__ == '__main__':
    main()