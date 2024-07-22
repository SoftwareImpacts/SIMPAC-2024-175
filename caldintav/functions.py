from sympy import *
import numpy as np
import  time, sys
from os import getcwd, system, path
import  inspect
import  matplotlib.pyplot as plt
import  multiprocessing
from numpy.fft import *
from caldintav import trains
from PyQt5.QtWidgets import QApplication
#------------------------------------------
location = path.abspath(path.dirname(trains.__file__))
#------------------------------------------
def update_progress(progress):
    barLength = 10
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if  progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if  progress >=1:
        progress = 1
        status = "Done...\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
def funcionFFT(datos):
    t = datos[:,0]
    y = datos[:,1]
    dt = t[1]-t[0]
    n = len(t)
    # Frecuencia de Nyquist
    if np.remainder(n,2)==0:
        fre = np.zeros(int(n/2))
        for i in range(int(n/2)):
            fre[i]=((i+1)*1./n)*(1/dt)
    else:
        fre = np.zeros(int((n-1)/2))
        for i in range(int((n-1)/2)):
            fre[i]=((i+1)*1./n)*(1/dt)
    m = len(fre)
    # Valor absoluto de FFT de senal
    k = np.abs(fft(y))*dt
    salida = k[1:m+1]
    return fre,salida
#------------------------------------------
class cell_struct():
    """
    Create a cell of structure of data
    """
    pass
#------------------------------------------
class train:
    """
    Clase that determines all characteristics of the train
    """
    def __init__(self):
        return
    def cmovtrain(self,trainname,tra,add_train):
        """
        Function that determines the characteristics of train with train id
        """
        if trainname=='HSLM-A1':
            fid = np.loadtxt(location+'/a01.def');
            self.nombre = 'HSLM-A1';
            self.vmax = 400.;
        elif trainname=='HSLM-A2':
            fid =np.loadtxt(location+'/a02.def');
            self.nombre = 'HSLM-A2';
            self.vmax = 400.;    
        elif trainname == 'HSLM-A3':
            fid = np.loadtxt(location+'/a03.def');
            self.nombre = 'HSLM-A3';
            self.vmax = 400.; 
        elif trainname=='HSLM-A4':
            fid = np.loadtxt(location+'/a04.def');
            self.nombre = 'HSLM-A4';
            self.vmax = 400.;
        elif trainname=='HSLM-A5':
            fid = np.loadtxt(location+'/a05.def');
            self.nombre = 'HSLM-A5';
            self.vmax = 400.;
        elif trainname=='HSLM-A6':
            fid = np.loadtxt(location+'/a06.def');
            self.nombre = 'HSLM-A6';
            self.vmax = 400.; 
        elif trainname=='HSLM-A7':
            fid = np.loadtxt(location+'/a07.def');
            self.nombre = 'HSLM-A7';
            self.vmax = 400.; 
        elif trainname=='HSLM-A8': 
            fid = np.loadtxt(location+'/a08.def');
            self.nombre = 'HSLM-A8';
            self.vmax = 400.; 
        elif trainname=='HSLM-A9':
            fid = np.loadtxt(location+'/a09.def');
            self.nombre = 'HSLM-A9';
            self.vmax = 400.; 
        elif trainname=='HSLM-A10':
            fid = np.loadtxt(location+'/a10.def');
            self.nombre = 'HSLM-A10';
            self.vmax = 400.;
        else:            
            for name in add_train:
                if trainname == name[0]:                    
                    filename = name[1]
                    # print(filename)
                    vmax = float(name[2])
                    #
                    fid = np.loadtxt(filename);
                    self.nombre = trainname;
                    self.vmax=vmax;
        #
        if tra != 0:
            self.dist = np.zeros(3*len(fid))
            self.peso = np.zeros(3*len(fid))
            for i  in range(len(fid)):
                self.dist[i*3] = fid[i,0]
                self.dist[i*3+1] = fid[i,0] + tra
                self.dist[i*3+2] = fid[i,0] + 2*tra
                self.peso[i*3] = fid[i,1]*0.25
                self.peso[i*3+1] = fid[i,1]*0.5
                self.peso[i*3+2] = fid[i,1]*0.25
        else:
            self.dist = fid[:,0]
            self.peso = fid[:,1]
        return
    def intetrain(self,trainname,tra,add_trainInte):
        """
        Function that determines the characteristics of train with train id
        """
        for name in add_trainInte:
            if trainname == name[0]:                    
                filename = name[1]
                # print(filename)
                vmax = float(name[2])
        #
        fid = np.loadtxt(filename);
        self.nombre = trainname;
        self.vmax=vmax;
        #
        if tra != 0:
            self.distv = fid[:,0]
            self.dist = np.zeros(3*len(fid))
            self.peso = np.zeros(3*len(fid))
            for i  in range(len(fid)):
                self.dist[i*3] = fid[i,0]
                self.dist[i*3+1] = fid[i,0] + tra
                self.dist[i*3+2] = fid[i,0] + 2*tra
                self.peso[i*3] = fid[i,1]*0.25
                self.peso[i*3+1] = fid[i,1]*0.5
                self.peso[i*3+2] = fid[i,1]*0.25
        else:
            self.dist = fid[:,0]; self.distv = self.dist
            self.peso = fid[:,1]
        #
        self.mw = fid[:,2] # mass of wheel
        self.mb = fid[:,3] # mass of bogie
        self.k1 = fid[:,4] # stiffness of primary suspension
        self.c1 = fid[:,5] # damping of primary suspension
        return
#--------------------
def phi1(x,param,beta):
    c1 = param[:,[0]]; c2 = param[:,[1]]; c3 = param[:,[2]]; c4 =param[:,[3]]
    b = beta
    y = x*np.cos(b*x)**2*(c1**2/2 + c2**2/2) - x*np.cos(b*x*1j)**2*(c3**2/2 - c4**2/2) - x*(np.cos(b*x)**2 - 1)*(c1**2/2 + c2**2/2) + x*(np.cos(b*x*1j)**2 - 1)*(c3**2/2 - c4**2/2) - (np.cos(b*x)*np.cos(b*x*1j)*(c1*c4 - c2*c3))/b + (np.cos(b*x*1j)*np.sin(b*x)*(c1*c3 + c2*c4))/b + (np.cos(b*x)*np.sin(b*x*1j)*(c1*c3 - c2*c4)*1j)/b - (np.sin(b*x)*np.sin(b*x*1j)*(c1*c4 + c2*c3)*1j)/b - (np.cos(b*x*1j)*np.sin(b*x*1j)*(c3**2 + c4**2)*1j)/(2*b) - (np.cos(b*x)*np.sin(b*x)*(c1**2 - c2**2))/(2*b) - (c1*c2*np.cos(b*x)**2)/b + (c3*c4*np.cos(b*x*1j)**2)/b 
    return y.real
#--------------------
def  phi2(x,param,la):
    c5 = param[:,[0]]
    c6 = param[:,[1]]
    y = (c5**2*x)/2 + (c6**2*x)/2 - (c5**2*np.sin(2*la*x))/(4*la) + (c6**2*np.sin(2*la*x))/(4*la) - (c5*c6*np.cos(2*la*x))/(2*la)
    return y
#--------------------
def SkewModal(estructura):
    EI = estructura.EI
    GJ = estructura.GJ
    alpha = estructura.alpha*np.pi/180
    r = estructura.r
    L = estructura.L
    nmod = estructura.nmod
    R1 = GJ/EI
    m = estructura.m
    #
    wn = symbols('wn')
    b = (m*wn**2/EI)**(1./4)
    la = (m*r**2*wn**2/GJ)**(0.5)
    # Matrix of homogeneous linear system AX=0
    if estructura.boun == 0: 
        # simply-supported beam 
        # print('simply-supported beam')
        A = Matrix([[0,1,0,1,0,0],
            [sin(b*L),cos(b*L),sinh(b*L),cosh(b*L),0,0],
            [-b*sin(alpha),0,-b*sin(alpha),0,0,cos(alpha)],
            [-b*sin(alpha)*cos(b*L),b*sin(alpha)*sin(b*L),-b*sin(alpha)*cosh(b*L),-b*sin(alpha)*sinh(b*L),cos(alpha)*sinh(la*L),cos(alpha)*cosh(la*L)],
            [0,-b**2,0,b**2,R1*la*tan(alpha),0],
            [-b**2*sin(b*L),-b**2*cos(b*L),b**2*sinh(b*L),b**2*cosh(b*L),R1*la*tan(alpha)*cosh(la*L),R1*la*tan(alpha)*sinh(la*L)]])
    elif estructura.boun == 1:
        # fixed beam
        # print('fixed-beam')
        A = Matrix([[0,1,0,1,0,0],
            [sin(b*L),cos(b*L),sinh(b*L),cosh(b*L),0,0],
            [-b*sin(alpha),0,-b*sin(alpha),0,0,cos(alpha)],
            [-b*sin(alpha)*cos(b*L),b*sin(alpha)*sin(b*L),-b*sin(alpha)*cosh(b*L),-b*sin(alpha)*sinh(b*L),cos(alpha)*sinh(la*L),cos(alpha)*cosh(la*L)],
            [b*cos(alpha),0,b*cos(alpha),0,0,sin(alpha)],
            [b*cos(alpha)*cos(b*L), -b*cos(alpha)*sin(b*L),b*cos(alpha)*cosh(b*L), b*cos(alpha)*sinh(b*L),sin(alpha)*sinh(la*L),sin(alpha)*cosh(la*L)]])
    # Calculate the determinant of A
    fun = det(A)
    # Establish the initial value for numerical solving the det(A)=0
    ca = 0.1
    cb = 1.0
    valor1 = fun.subs(wn,ca)
    valor2 = fun.subs(wn,cb)
    freq = np.zeros(nmod)
    beta = np.zeros((nmod,1))
    for i in range(nmod):
        # check the initital value that is closely possible to the solution
        while sign(valor1)==sign(valor2):
            ca =cb
            valor1 = valor2
            cb =cb + 1.0*(i+1)
            valor2 =fun.subs(wn,cb)
	    #
        # print('Valor de ca: %.3f y valor de cb: %3.f' % (ca,cb))
        freq[i] = np.double(nsolve(fun,wn,(ca,cb),solver='bisect',verify=False))
        #a1 = np.double(nsolve(fun,wn,ca,verify=False))
        # a2 = np.double(nsolve(fun,wn,cb,verify=False))
        # if  abs(fun.subs(wn,a1)) <= abs(fun.subs(wn,a2)):
        #    freq[i] = a1
        #else:
        #    freq[i] = a2
        # print('Value of det(A) for mode ',i+1, ' is', np.double(fun.subs(wn,freq[i])))
        beta[i] = (m*freq[i]**2/EI)**(1./4)
        ca =cb
        valor1 = valor2
        cb = cb + 1.*(i+1)
        valor2 =fun.subs(wn,cb)
    #
    estructura.wn = freq
    estructura.beta =beta
    estructura.wd = freq*np.sqrt(1-estructura.xi**2)
    aux = np.zeros((nmod,1)); aux[:,0] = (m*r**2*freq**2/GJ)**(0.5)
    estructura.la = aux 
    # Determine the coefficients of modal shapes
    lista = []
    for i in range(2,7):
        lista.append(var('c'+str(i)))	
    param = np.zeros([nmod,6])
    for i in range(nmod):
        A1 = A.subs(wn,freq[i])
        B = A1[:-1,1:]; n1,m1 = B.shape
        bb = -A1[:-1,0]
        C = B.col_insert(m1,bb)
        Coef = solve_linear_system_LU(C,lista)
        lista1 = [str(kk) for kk in Coef.keys()]; valores= np.array(list(Coef.values())); 
        orden = np.argsort(lista1)
        param[i,0] = 1.0
        param[i,1:] = valores[orden] 
    estructura.param = param 
    return estructura
# -----------------------------------------------------------
def ParaIntegraExact(estructura):
    """
    Function that determines the parameters for the integration with linear interpolation for step size dt
    """
    dt = estructura.dt
    xi = estructura.xi
    wn = estructura.wn
    wd = estructura.wd
    estructura.A = np.exp(-xi*wn*dt)*(xi/np.sqrt(1-xi**2)*np.sin(wd*dt) + np.cos(wd*dt))
    estructura.B = np.exp(-xi*wn*dt)*(np.sin(wd*dt)/wd);
    estructura.C = (1./wn**2)*(2*xi/(wn*dt) + np.exp(-xi*wn*dt)*(((1-2*xi**2)/(wd*dt)-xi/np.sqrt(1-xi**2))*np.sin(wd*dt) - (1 + 2*xi/(wn*dt))*np.cos(wd*dt)));
    estructura.D = (1./wn**2)*(1 - 2*xi/(wn*dt) + np.exp(-xi*wn*dt)*(((2*xi**2-1)/(wd*dt))*np.sin(wd*dt) + (2*xi/(wn*dt))*(np.cos(wd*dt))));
    estructura.A1 = -np.exp(-xi*wn*dt)*((wn/np.sqrt(1-xi**2))*np.sin(wd*dt));
    estructura.B1 = np.exp(-xi*wn*dt)*(np.cos(wd*dt) - (xi/np.sqrt(1-xi**2))*np.sin(wd*dt));
    estructura.C1 = (1./wn**2)*(-1/dt + np.exp(-xi*wn*dt)*((wn/np.sqrt(1-xi**2) + xi/(dt*np.sqrt(1-xi**2)))*np.sin(wd*dt) + (1/dt)*np.cos(wd*dt)));
    estructura.D1 = (1./(wn**2*dt))*(1 - np.exp(-xi*wn*dt)*(xi/np.sqrt(1-xi**2)*np.sin(wd*dt) + np.cos(wd*dt)));
    return estructura
# -----------------------------------------------------------
def SkewMovIntegra(args):
    """
    % function that calculates the dynamic response of bridge due to a moving
    % load at a distance x from the start point
    % User form: res = SkewMovIntegra(estructura,vel,x,P)
    % Inputs: 
    %   estructura:  structures that contains all information about structure
    %   vel: moving load velocity
    %   x:  Distance at a point from start point 
    %   tren.dist:  vector that contains the sistance between axle loading [1xn_eje]
    %   tren.peso:  vector that contains the train loadings [1xn_eje]
    % Outputs: 
    %   res(:,1): times
    %   res(:,2): displacement at distance x from the start point of span
    %   res(:,3): velocity at distance x from the start point of span
    %   res(:,4): acceleration at distance x from the start point of span
    %   res(:,5): rotational angle at distance x
    %   res(:,6): velocity of rotational angle at distance x
    % Created by Khanh Nguyen Gia
    % ========================================================================
    """
    #
    estructura, vel, x, tren, qq = args
    # Total time integration
    total_time = (tren.dist[-1]+estructura.L)/vel + 1.0
    t = np.arange(0,total_time+estructura.dt,estructura.dt)
    # Initial conditions at t = 0
    b = np.zeros(estructura.nmod)
    db = np.zeros(estructura.nmod)
    c = np.zeros(estructura.nmod)
    dc = np.zeros(estructura.nmod)
    P0 = np.zeros(estructura.nmod)
    T0 = np.zeros(estructura.nmod)
    #
    param = estructura.param; beta = estructura.beta; la = estructura.la; nmod=estructura.nmod
    alpha1 = (90-estructura.alpha)*np.pi/180
    k1 = float(estructura.L/2*np.tan(alpha1)/(1+(estructura.EI/estructura.GJ)*np.tan(alpha1)**2))
    # Modal mass
    Mi = estructura.m*(phi1(estructura.L,param[:,:4],beta) - phi1(0.,param[:,:4],beta))
    # Modal rotational mass
    Mti = estructura.m*estructura.r**2*(phi2(estructura.L,param[:,4:6],la) - phi2(0.,param[:,4:6],la)); 
    # modes shape at distance x
    phi1_x = param[:,[0]]*np.sin(beta*x) + param[:,[1]]*np.cos(beta*x) + param[:,[2]]*np.sinh(beta*x) + param[:,[3]]*np.cosh(beta*x);
    phi2_x = param[:,[4]]*np.sin(la*x) + param[:,[5]]*np.cos(la*x);
    # Loop for integration
    res = np.zeros([len(t),6])
    for i in range(1,len(t)):
        dist = vel*t[i] - tren.dist; dist[dist>estructura.L]=0.; dist[dist<0]=0.; 
        # Displacement and velocity at instance t_{n-1}
        q = b; dq= db;
        p = c; dp = dc;
        # Modal force at instance t_n
        aa1 = beta*dist
        Pn = -(param[:,[0]]*np.sin(aa1) + param[:,[1]]*np.cos(aa1)+param[:,[2]]*np.sinh(aa1) + param[:,[3]]*np.cosh(aa1)).dot(tren.peso) / Mi[:,0]; 
        # Modal torsional force at t_n
        if  estructura.alpha == 0 and tren.excentricidad ==0:
            Tn = T0
        else:
            aux2 = -k1*tren.peso*(dist/estructura.L - dist**2/estructura.L**2) + tren.peso*tren.excentricidad
            aa2 = la*dist
            Tn = (param[:,[4]]*np.sin(aa2)+param[:,[5]]*np.cos(aa2)).dot(aux2) / Mti[:,0]
        # Integration based on the linear interpolation of excitation
        b = estructura.A*q + estructura.B*dq + estructura.C*P0 + estructura.D*Pn;
        db = estructura.A1*q + estructura.B1*dq + estructura.C1*P0 + estructura.D1*Pn
        ddb = Pn - 2*estructura.xi*estructura.wn*db-estructura.wn**2*b
        c = estructura.A*p + estructura.B*dp + estructura.C*T0 + estructura.D*Tn
        dc = estructura.A1*p + estructura.B1*dp + estructura.C1*T0 + estructura.D1*Tn
        # Update for next time step
        P0 = Pn
        T0 = Tn
        # 
        res[i,0] = t[i]
        res[i,1] = b.dot(phi1_x)
        res[i,2] = db.dot(phi1_x)
        res[i,3] = ddb.dot(phi1_x)
        res[i,4] = c.dot(phi2_x)
        res[i,5] = dc.dot(phi2_x)
        # update_progress(float(i+1)/len(t))
	#
    qq.put(1)
    return res
# -----------------------------------------------------------
def SkewStatic(estructura,tren):
    """
    Function that determines the maximum static displaxcement generated by a convoy of loads
    Input:
        estructura: a class of data of structure
        tren: a class of data of analysed train
    Output: 
        estructura.u_max:  maximum static displacement
        estructura.u_static: vector of static displacement
        estructura.phi1: vector of torsional angle at suuport 1
        estructura.phi2: vector of torsional 
    """
    EI = estructura.EI; GJ = estructura.GJ; k1 = EI/GJ; alpha1 = (90-estructura.alpha)*np.pi/180.;
    L =estructura.L; x = L/2
    # number of discretized bridge span
    n = 100
    # increment in distance
    ds = L/n
    # number of increment
    Np = int(np.ceil((tren.dist[-1]+L)/ds))
    res = np.zeros(Np)
    res1 = np.zeros(Np)
    res2 = np.zeros(Np)
    ft1, ft2 = 1./L, 1/(2*(1+k1*np.tan(alpha1)**2))
    for i in range(Np):
        distance = i*ds - tren.dist
        distance[distance>L] = 0.; distance[distance<0] = 0.
        aux2 = np.where(distance>L/2)[0]
        distance[aux2] = L - distance[aux2]
        aux3 = np.where(distance>0.)[0]
        P = tren.peso[aux3]; dist = distance[aux3]
        if  estructura.boun == 0:
            Ma = P*dist*(1-dist*ft1)*ft2
            A = (0.5*Ma*L**2 - P*(L-dist)*(L**2-(L-dist)**2)/6)/L
            a1 = (P*(1-dist/L)*x**3/6-Ma*x**2/2 + A*x - P*((x-dist)**3)/6)/EI
            b1 = A/EI; b2 = (P*dist*(L-dist)/2 + Ma*L + A)/EI
            res[i] = a1.sum()
            res1[i] = b1.sum()
            res2[i] = b2.sum()
        elif estructura.boun == 1:
            a1 = P*dist**2 * (2*(L-dist)-L/2.) /(24.*EI)
            res[i] = a1.sum()
            res1[i] = 0
            res2[i] = 0
        #update_progress(float(i+1)/Np)
    # 
    estructura.u_max = np.max(abs(res))
    estructura.u_static = res
    estructura.phi1 = res1; estructura.phi1_max = np.max(abs(res1))
    estructura.phi2 = res2; estructura.phi2_max = np.max(abs(res2))
    return estructura
#---------------------------
#
def  SkewInteIntegra(args):
    """
    % function that calculates the dynamic response of bridge due to a lumped mass 
    % (1/4 bogie model) at a distance x from the start point, solving with
    % beta-Newmark method
    % User form: res = SkewInteIntegra(estructura,vel,x,tren)
    % Inputs: 
    %   estructura:  structures that contains all information about structure
    %   vel: moving load velocity
    %   x:  Distance at a point from start point 
    %   tren.dist:  vector that contains the distance between axle loading [1xn], if tra !=0, n!=n_eje
    %   tren.distv:  vector that contains the distance between axle loading [1xn_eje]
    %   tren.mb:  lumped mass [1xn_eje]
    %   tren.mw:   no lumped mass [1xn_eje]
    %   tren.k1: stiffness of primary suspension [1xn_eje]
    %   tren.c1: damping of primary suspension [1xn_eje] 
    % Outputs: 
    %   res(:,1): times
    %   res(:,2): displacement at distance x from the start point of span
    %   res(:,3): velocity at distance x from the start point of span
    %   res(:,4): acceleration at distance x from the start point of span
    %   res(:,5): displacement of  lumped mass
    %   res(:,6): velocity of  lumped mass
    %   res(:,7): acceleration of lumped mass
    % Created by Khanh Nguyen Gia
    % ========================================================================
    """
    estructura, vel, x, tren, qq = args
    # Total time integration
    total_time =(tren.dist[-1] + estructura.L )/vel + 1.0;
    t = np.arange(0,total_time+estructura.dt,estructura.dt);
    dt=estructura.dt;
    e = tren.excentricidad;
    param = estructura.param; beta = estructura.beta; la = estructura.la;
    alpha1 = (90 - estructura.alpha)*np.pi/180;
    if estructura.alpha == 0:
        # Matrix of Modal mass of bridge for bending
        Mi = np.zeros((estructura.nmod,estructura.nmod))
        mii = estructura.m*(phi1(estructura.L,param[:,:4],beta) - phi1(0.,param[:,:4],beta))
        # Matrix of modal damping of bridge for bending
        Ci = np.zeros((estructura.nmod,estructura.nmod));
        # Matrix of stiffness of bridge for bending
        Ki = np.zeros((estructura.nmod,estructura.nmod));
        #
        for i in range(estructura.nmod):
            Mi[i,i] = mii[i,0]
            Ci[i,i] = 2*estructura.xi*estructura.wn[i]*Mi[i,i];
            Ki[i,i] = estructura.wn[i]**2*Mi[i,i];
        # number of axle loads
        ne = len(tren.distv);
        # Mass matrix of lumped mass
        Mbb = np.zeros((ne,ne));
        # Mass matrix of no lumped mass
        Mw = np.zeros((ne,ne));
        # Damping matrix  of vehicle
        Cv = np.zeros((ne,ne));
        # Stiffness matrix of vehicle
        Kv = np.zeros((ne,ne));
        for i in range(ne):
            Mbb[i,i] = tren.mb[i];
            Mw[i,i] = tren.mw[i];
            Cv[i,i] = tren.c1[i];
            Kv[i,i] = tren.k1[i];
        # Mass matrix of coupled system
        M0 = np.zeros((estructura.nmod+ne,estructura.nmod+ne));
        M0[estructura.nmod:,estructura.nmod:] = Mbb;
        # Stiffness matrix of coupled system
        K0 = np.zeros((estructura.nmod+ne,estructura.nmod+ne));
        K0[estructura.nmod:,estructura.nmod:] = Kv;
        # Damping matrix of coupled system
        C0 = np.zeros((estructura.nmod+ne,estructura.nmod+ne));
        C0[estructura.nmod:,estructura.nmod:] = Cv;
        # Initital data
        b = np.zeros(estructura.nmod+ne);
        db = np.zeros(estructura.nmod+ne);
        ddb = np.zeros(estructura.nmod+ne);
        Fn = np.zeros(estructura.nmod+ne);
        # modes shape at distance x;
        phi1_x = param[:,[0]]*np.sin(beta*x) + param[:,[1]]*np.cos(beta*x) + param[:,[2]]*np.sinh(beta*x) + param[:,[3]]*np.cosh(beta*x);
        phi2_x = param[:,[4]]*np.sinh(la*x) + param[:,[5]]*np.cosh(la*x);
        #
        res = np.zeros((len(t),10));
        # Newmark parameters for integration
        gamma = 0.5; be = 0.25;
        # determinar los coeficientes del metodo de newmark
        c1 = 1/(be*dt**2); c2 = 1/(be*dt); c3 = 1/(2*be) -1;
        c4 = (1-gamma/be); c5 = dt*(1-gamma/(2*be)); c6 = gamma/(be*dt);            
        # Start bucle to calculate the dynamic responses
        nmod = estructura.nmod
        for i in range(1,len(t)):
            # Determinate the axles running in the bridge
            distv = vel*t[i] - tren.distv; 
            distv[distv > estructura.L] = 0.;  distv[distv < 0.] = 0.;
            distp = vel*t[i] - tren.dist;
            distp[distp > estructura.L] = 0.; distp[distp < 0.] = 0.;
            # Displacement and velocity, acceleration at instance t_{n-1}
            q = b; dq = db; ddq = ddb;
            # Determinate the matrix of modal shape at vt
            aa1 = beta*distv; aa2 = beta*distp; 
            Phi1 = param[:,[0]]*np.sin(aa1) + param[:,[1]]*np.cos(aa1) + param[:,[2]]*np.sinh(aa1) + param[:,[3]]*np.cosh(aa1);
            #
            phi1p = param[:,[0]]*np.sin(aa2) + param[:,[1]]*np.cos(aa2) + param[:,[2]]*np.sinh(aa2) + param[:,[3]]*np.cosh(aa2);
            # Mass matrix
            M = np.zeros((estructura.nmod+ne,estructura.nmod+ne)); M[:]= M0[:]; M[:estructura.nmod,:estructura.nmod] = Mi+(Phi1.dot(Mw)).dot(Phi1.T);
            M[:estructura.nmod, estructura.nmod:] = Phi1.dot(Mbb);
            #
            K = np.zeros((estructura.nmod+ne,estructura.nmod+ne)); K[:]= K0[:]; K[:estructura.nmod,:estructura.nmod] = Ki;
            K[estructura.nmod:,:estructura.nmod] = -Kv.dot(Phi1.T);
            #
            C = np.zeros((estructura.nmod+ne,estructura.nmod+ne)); C[:]=C0[:]; C[:estructura.nmod,:estructura.nmod] = Ci;
            C[estructura.nmod:,:estructura.nmod] = -Cv.dot(Phi1.T);
            #
            #Fn[:estructura.nmod] = -phi1p.dot(Fw); Fn[estructura.nmod:estructura.nmod*2] = -phi2p.dot(Fw*auxp);
            #Fn[estructura.nmod*2:] = -Fb;
            #
            Fn[:estructura.nmod] = -phi1p.dot(tren.peso); 
            FF = Fn + (c1*M+c6*C).dot(q)+(c2*M-c4*C).dot(dq) + (c3*M-c5*C).dot(ddq);
            KK = c1*M+c6*C+K;
            b = np.linalg.solve(KK,FF);
            ddb = c1*(b-q) - c2*dq - c3*ddq;
            db = dq + dt*(1-gamma)*ddq + gamma*dt*ddb;
            res[i,0] = t[i];
            res[i,1] = b[:estructura.nmod].dot(phi1_x);
            res[i,2] = db[:estructura.nmod].dot(phi1_x);
            res[i,3] = ddb[:estructura.nmod].dot(phi1_x);
            res[i,4] = 0.
            res[i,5] = 0.
            res[i,6] = 0.
            res[i,7] = b[estructura.nmod];
            res[i,8] = db[estructura.nmod];
            res[i,9] = ddb[estructura.nmod];
            #update_progress(float(i+1)/len(t))
    else:
        k1 = estructura.L/2 *np.tan(alpha1)/(1 + (estructura.EI/estructura.GJ)*np.tan(alpha1)**2);
        # Matrix of Modal mass of bridge for bending
        Mi = np.zeros((estructura.nmod,estructura.nmod))
        mii = estructura.m*(phi1(estructura.L,param[:,:4],beta) - phi1(0.,param[:,:4],beta))
        # Matrix of modal mass of bridge for torsion
        Mti = np.zeros((estructura.nmod,estructura.nmod))
        mtii = estructura.m*estructura.r**2*(phi2(estructura.L,param[:,4:6],la)-phi2(0.,param[:,4:6],la))
        # Matrix of modal damping of bridge for bending
        Ci = np.zeros((estructura.nmod,estructura.nmod));
        # Matrix of modal damping of bridge for torsion
        Cti = np.zeros((estructura.nmod,estructura.nmod));
        # Matrix of stiffness of bridge for bending
        Ki = np.zeros((estructura.nmod,estructura.nmod));
        # Matrix of modal stiffness of bridge for torsion
        Kti = np.zeros((estructura.nmod,estructura.nmod));
        #
        for i in range(estructura.nmod):
            Mi[i,i] = mii[i,0]
            Mti[i,i] = mtii[i,0]
            Ci[i,i] = 2*estructura.xi*estructura.wn[i]*Mi[i,i];
            Cti[i,i] = 2*estructura.xi*estructura.wn[i]*Mti[i,i];
            # Cti[i,i] = 0.0;
            Ki[i,i] = estructura.wn[i]**2*Mi[i,i];
            Kti[i,i] = estructura.wn[i]**2*Mti[i,i];
        # number of axle loads
        ne = len(tren.distv);
        # Mass matrix of lumped mass
        Mbb = np.zeros((ne,ne));
        # Mass matrix of no lumped mass
        Mw = np.zeros((ne,ne));
        # Damping matrix  of vehicle
        Cv = np.zeros((ne,ne));
        # Stiffness matrix of vehicle
        Kv = np.zeros((ne,ne));
        for i in range(ne):
            Mbb[i,i] = tren.mb[i];
            Mw[i,i] = tren.mw[i];
            Cv[i,i] = tren.c1[i];
            Kv[i,i] = tren.k1[i];
        # Mass matrix of coupled system
        M0 = np.zeros((estructura.nmod*2+ne,estructura.nmod*2+ne));
        M0[estructura.nmod*2:,estructura.nmod*2:] = Mbb;
        # Stiffness matrix of coupled system
        K0 = np.zeros((estructura.nmod*2+ne,estructura.nmod*2+ne));
        K0[estructura.nmod*2:,estructura.nmod*2:] = Kv;
        # Damping matrix of coupled system
        C0 = np.zeros((estructura.nmod*2+ne,estructura.nmod*2+ne));
        C0[estructura.nmod*2:,estructura.nmod*2:] = Cv;
        # Initital data
        b = np.zeros(estructura.nmod*2+ne);
        db = np.zeros(estructura.nmod*2+ne);
        ddb = np.zeros(estructura.nmod*2+ne);
        Fn = np.zeros(estructura.nmod*2+ne);
        # modes shape at distance x;
        phi1_x = param[:,[0]]*np.sin(beta*x) + param[:,[1]]*np.cos(beta*x) + param[:,[2]]*np.sinh(beta*x) + param[:,[3]]*np.cosh(beta*x);
        phi2_x = param[:,[4]]*np.sinh(la*x) + param[:,[5]]*np.cosh(la*x);
        #
        res = np.zeros((len(t),10));
        # Newmark parameters for integration
        gamma = 0.5; be = 0.25;
        # determinar los coeficientes del metodo de newmark
        c1 = 1/(be*dt**2); c2 = 1/(be*dt); c3 = 1/(2*be) -1;
        c4 = (1-gamma/be); c5 = dt*(1-gamma/(2*be)); c6 = gamma/(be*dt);            
        # Start bucle to calculate the dynamic responses
        nmod = estructura.nmod
        for i in range(1,len(t)):
            # Determinate the axles running in the bridge
            distv = vel*t[i] - tren.distv; 
            distv[distv > estructura.L] = 0.;  distv[distv < 0.] = 0.;
            distp = vel*t[i] - tren.dist;
            distp[distp > estructura.L] = 0.; distp[distp < 0.] = 0.;
            # Displacement and velocity, acceleration at instance t_{n-1}
            q = b; dq = db; ddq = ddb;
            # Determinate the matrix of modal shape at vt
            aa1 = beta*distv; bb1 = la*distv; aa2 = beta*distp; bb2 = la*distp
            Phi1 = param[:,[0]]*np.sin(aa1) + param[:,[1]]*np.cos(aa1) + param[:,[2]]*np.sinh(aa1) + param[:,[3]]*np.cosh(aa1);
            Phi2 = param[:,[4]]*np.sinh(bb1) + param[:,[5]]*np.cosh(bb1);
            #
            phi1p = param[:,[0]]*np.sin(aa2) + param[:,[1]]*np.cos(aa2) + param[:,[2]]*np.sinh(aa2) + param[:,[3]]*np.cosh(aa2);
            phi2p = param[:,[4]]*np.sinh(bb2) + param[:,[5]]*np.cosh(bb2);
            # 
            aux = k1*(distv/estructura.L - distv**2/estructura.L**2)+e; 
            auxp = k1*(distp/estructura.L - distp**2/estructura.L**2)+e;
            # Mass matrix
            M = np.zeros((estructura.nmod*2+ne,estructura.nmod*2+ne)); M[:]= M0[:]; M[:estructura.nmod,:estructura.nmod] = Mi+(Phi1.dot(Mw)).dot(Phi1.T);
            M[:estructura.nmod, estructura.nmod*2:] = Phi1.dot(Mbb);
            M[estructura.nmod:estructura.nmod*2,:estructura.nmod] = (Phi2.dot(Mw*aux)).dot(Phi1.T);
            M[estructura.nmod:estructura.nmod*2,estructura.nmod:estructura.nmod*2] = Mti;
            M[estructura.nmod:estructura.nmod*2,estructura.nmod*2:] = Phi2.dot(Mbb*aux);
            #
            K =np.zeros((estructura.nmod*2+ne,estructura.nmod*2+ne)); K[:]= K0[:]; K[:estructura.nmod,:estructura.nmod] = Ki;
            K[estructura.nmod:estructura.nmod*2,estructura.nmod:estructura.nmod*2] = Kti;
            K[estructura.nmod*2:,:estructura.nmod] = -Kv.dot(Phi1.T);
            #
            C = np.zeros((estructura.nmod*2+ne,estructura.nmod*2+ne)); C[:]=C0[:]; C[:estructura.nmod,:estructura.nmod] = Ci;
            C[estructura.nmod:estructura.nmod*2,estructura.nmod:estructura.nmod*2] = Cti;
            C[estructura.nmod*2:,:estructura.nmod] = -Cv.dot(Phi1.T);
            #
            Fn[:estructura.nmod] = -phi1p.dot(tren.peso); Fn[estructura.nmod:estructura.nmod*2] = -phi2p.dot((tren.peso)*auxp);
            FF = Fn + (c1*M+c6*C).dot(q)+(c2*M-c4*C).dot(dq) + (c3*M-c5*C).dot(ddq);
            KK = c1*M+c6*C+K;
            b = np.linalg.solve(KK,FF);
            ddb = c1*(b-q) - c2*dq - c3*ddq;
            db = dq + dt*(1-gamma)*ddq + gamma*dt*ddb;
            res[i,0] = t[i];
            res[i,1] = b[:estructura.nmod].dot(phi1_x);
            res[i,2] = db[:estructura.nmod].dot(phi1_x);
            res[i,3] = ddb[:estructura.nmod].dot(phi1_x);
            res[i,4] = b[estructura.nmod:estructura.nmod*2].dot(phi2_x);
            res[i,5] = db[estructura.nmod:estructura.nmod*2].dot(phi2_x);
            res[i,6] = ddb[estructura.nmod:estructura.nmod*2].dot(phi2_x);
            res[i,7] = b[estructura.nmod*2];
            res[i,8] = db[estructura.nmod*2];
            res[i,9] = ddb[estructura.nmod*2];
            # update_progress(float(i+1)/len(t))
        #
    qq.put(1)
    return res
# ------------ Spring model ----------------------------------
def SpringModal(estructura):
    """	
    % ==================================================================
    % Function that determines the frequency and modal shape parameters
    % User form: SpringModal(estructura)
    % Inputs:
    %	estructura: a class that contains all information of structure
    % Outputs:
    %       estructura.beta:
    %       estructura.wn:  frequency in rad/s
    %       estructura.param: parameter of modal shape
    %       estructura.wd: amortiguated frequency in rad/s
    % ==================================================================
    % Created by Khanh Nguyen Gia Jul-2016
    """
    wn = symbols('wn');
    EI = estructura.EI;
    GJ = estructura.GJ;
    m = estructura.m;
    b = (m*wn**2/EI)**(1./4);
    alpha = estructura.alpha;
    alpha1 = (90-alpha)*np.pi/180;
    L = estructura.L;
    nmod = estructura.nmod;
    k = 2*GJ/(L*tan(alpha1)**2);
    B = Matrix([[0,1,0,1],[sin(b*L),cos(b*L),sinh(b*L),cosh(b*L)],[-k*b/EI, -b**2, -k*b/EI, b**2], [k*b*cos(b*L)/EI-b**2*sin(b*L), -k*b*sin(b*L)/EI-b**2*cos(b*L),k*b*cosh(b*L)/EI+b**2*sinh(b*L), k*b*sinh(b*L)/EI+b**2*cosh(b*L)]]);
    # determinate de matriz B
    fun = det(B);
    # establece unos valores iniciales para el bucle de hallar solucion numerica
    ca = 1.e-02; valor1=fun.subs(wn,ca);
    cb = 0.1; valor2 = fun.subs(wn,cb);
    freq = np.zeros(nmod);
    beta = np.zeros(nmod);
    for i in range(nmod):
        while sign(valor1)==sign(valor2):
            ca=cb;
            valor1=valor2;
            cb=cb+1.0;
            valor2 = fun.subs(wn,cb);
        a1=np.double(nsolve(fun,wn,cb,verify=False));
        a2=np.double(nsolve(fun,wn,cb,verify=False));
        if  abs(fun.subs(wn,a1)) <= abs(fun.subs(wn,a2)):
            freq[i] = a1
        else:
            freq[i] = a2
        # print('Value of det(A) for mode ',i+1, ' is', np.double(fun.subs(wn,freq[i])))
        beta[i] = (m*freq[i]**2/EI)**(1./4);
        ca =cb;
        cb=cb+1.0*(i+1);
        valor1=valor2;
        valor2=fun.subs(wn,cb);
	#
    # Determine the coefficicnets of modal shape
    param = np.zeros((nmod,4));
    var('c2,c3,c4')
    for i in range(nmod):
        AA = B.subs(wn,freq[i]);
        B1 = AA[:-1,1:]; n1,m1 = B1.shape
        bb = -AA[:-1,0]
        C = B1.col_insert(m1,bb)
        Coef = solve_linear_system(C,c2,c3,c4)
        param[i,0] = 1.0
        param[i,1] = Coef[c2]
        param[i,2] = Coef[c3]
        param[i,3] = Coef[c4]
	#
    estructura.beta = beta; estructura.wn=freq; estructura.param = param;
    estructura.wd = freq*np.sqrt(1-estructura.xi**2);
    return estructura
# -----------------------------------
def SpringStatic(estructura,tren):
    """
    % ==================================================================
    % Function that determines the maximum static deflection of bridge
    % due to the train loadings
    % User form: estructura = SpringStatic(estructura,tren)
    % Inputs:
    %	estructura: object that contains all information of structure
    %   tren:   object that contains information of vehicles
    % Outputs:
    %       estructura.u_max: maximum static displacement
    % ==================================================================
    """
    n = 100;
    # increment in distance
    ds = estructura.L/n;
    # number of increment
    Np = int(np.ceil((tren.dist[-1]+estructura.L)/ds));
    res = np.zeros(Np);
    #
    alpha1 = (90 - estructura.alpha)*np.pi/180;
    k0 = (2*estructura.GJ)/(estructura.L*np.tan(alpha1)**2); L =estructura.L; EI = estructura.EI;
    #
    for i in range(Np):
        # fprintf('%s%i\n','Step ',i);
        distance = i*ds - tren.dist;
        distance[distance > estructura.L] = 0.; distance[distance < 0.] = 0.;
        aux2 = np.where(distance > estructura.L/2);
        distance[aux2] = estructura.L - distance[aux2];
        aux3 = np.where(distance>0.0); P =tren.peso[aux3]; a = distance[aux3]
        A = -(EI*(P*k0*L**3*a - 2*P*k0*L**2*a**2 + 4*EI*P*L**2*a + P*k0*L*a**3 - 6*EI*P*L*a**2 + 2*EI*P*a**3))/(L*(12*EI**2 + 8*EI*L*k0 + L**2*k0**2))
        Ma = -k0*A/EI
        Mb = -(k0*(- P*k0*L**2*a**2 - 2*EI*P*L**2*a + P*k0*L*a**3 + 2*EI*P*a**3))/(L*(12*EI**2 + 8*EI*L*k0 + L**2*k0**2))
        x = L/2
        u2 = -((L**3*((P*(a/L - 1))/6 - ((k0*(- P*k0*L**2*a**2 - 2*EI*P*L**2*a + P*k0*L*a**3 + 2*EI*P*a**3))/(L*(12*EI**2 + 8*EI*L*k0 + L**2*k0**2)) + (k0*(P*k0*L**3*a - 2*P*k0*L**2*a**2 + 4*EI*P*L**2*a + P*k0*L*a**3 - 6*EI*P*L*a**2 + 2*EI*P*a**3))/(L*(12*EI**2 + 8*EI*L*k0 + L**2*k0**2)))/(6*L)))/8 + (P*(L/2 - a)**3)/6 + (EI*(P*k0*L**3*a - 2*P*k0*L**2*a**2 + 4*EI*P*L**2*a + P*k0*L*a**3 - 6*EI*P*L*a**2 + 2*EI*P*a**3))/(2*(12*EI**2 + 8*EI*L*k0 + L**2*k0**2)) + (L*k0*(P*k0*L**3*a - 2*P*k0*L**2*a**2 + 4*EI*P*L**2*a + P*k0*L*a**3 - 6*EI*P*L*a**2 + 2*EI*P*a**3))/(8*(12*EI**2 + 8*EI*L*k0 + L**2*k0**2)))/EI
        res[i] = np.sum(u2)
    estructura.u_max = np.max(abs(res));
    estructura.u_static = res;
    return estructura
#-----------------------------------------
def SpringMovIntegra(estructura,vel,x,tren):
    """		
    % function that calculates the dynamic response of bridge due to a moving 
    % load at a distance x from the start point
    % User form: SpringMovIntegra(estructura,vel,x,tren)
    % Inputs: 
    %   estructura:  structures that contains all information about structure
    %   vel: moving load velocity (m/s)
    %   x:  Distance at a point from start point 
    %   tren.dist:  vector that contains the sistance between axle loading [1xn_eje]
    %   tren.peso:  vector that contains the train loadings [1xn_eje]
    % Outputs: 
    %   res(:,1): times
    %   res(:,2): displacement at distance x from the start point of span
    %   res(:,3): velocity at distance x from the start point of span
    %   res(:,4): acceleration at distance x from the start point of span
    % Created by Khanh Nguyen Gia
    % ========================================================================
    """
    # Total time integration
    total_time =(tren.dist[-1] + estructura.L )/ vel + 1.0;
    t = np.arange(0,total_time+estructura.dt,estructura.dt);
    # Initital conditions at t=0
    b = np.zeros(estructura.nmod);
    db = np.zeros(estructura.nmod);
    P0 = np.zeros(estructura.nmod);
    # Modal mass
    Mi = np.zeros(estructura.nmod);
    param = estructura.param; beta = estructura.beta;
    var('a')
    for i in range(estructura.nmod):
        f2 = param[i,0]*sin(beta[i]*a)+param[i,1]*cos(beta[i]*a)+param[i,2]*sinh(beta[i]*a)+param[i,3]*cosh(beta[i]*a);	
        f3 = f2**2;
        Mi[i] = estructura.m*np.double(integrate(f3,(a,0,estructura.L)));
    # modes shape at distance x;
    phi_x = param[:,0]*np.sin(beta*x) + param[:,1]*np.cos(beta*x) + param[:,2]*np.sinh(beta*x) + param[:,3]*np.cosh(beta*x);
    # bucle for integration
    res = np.zeros((len(t),4));
    nmod = estructura.nmod
    for i in range(1,len(t)):
        dist = vel*t[i] - tren.dist; 
        dist[dist > estructura.L] = 0.;  dist[dist < 0.] = 0.;
        # Displacement and velocity at instance t_{n-1}
        q = b; dq = db;
        # Modal forces at instance t_{n}
        aux = (param[:,0].reshape(nmod,1)*np.sin(beta.reshape(nmod,1)*dist.reshape(1,len(dist))) + param[:,1].reshape(nmod,1)*np.cos(beta.reshape(nmod,1)*dist.reshape(1,len(dist))) + param[:,2].reshape(nmod,1)*np.sinh(beta.reshape(nmod,1)*dist.reshape(1,len(dist))) + param[:,3].reshape(nmod,1)*np.cosh(beta.reshape(nmod,1)*dist.reshape(1,len(dist)))).dot(tren.peso); 
        Pn = -aux/Mi;
        # integration based on the linear interpolation of excitation
        b = estructura.A * q + estructura.B * dq + estructura.C * P0 + estructura.D * Pn;
        db = estructura.A1 * q + estructura.B1 * dq + estructura.C1 * P0 + estructura.D1 * Pn;
        ddb = Pn - 2*estructura.xi*estructura.wn * db - estructura.wn**2*b;
        # update the modal forces for next increment
        P0 = Pn;
        res[i,0] = t[i];
        res[i,1] = np.dot(phi_x,b);
        res[i,2] = np.dot(phi_x,db);
        res[i,3] = np.dot(phi_x,ddb);
    return res
# -------------------------------------------
def SpringInteIntegra(estructura,vel,x,tren):
    """
    % function that calculates the dynamic response of bridge due to a lumped mass 
    % (1/4 bogie model) at a distance x from the start point, solving with
    % beta-Newmark method
    % User form: res = SpringInteIntegra(estructura,vel,x,P)
    % Inputs: 
    %   estructura:  structures that contains all information about structure
    %   vel: moving load velocity
    %   x:  Distance at a point from start point 
    %   tren.dist:  vector that contains the sistance between axle loading [1xn_eje]
    %   tren.ms:  lumped mass [1xn_eje]
    %   tren.m:   no lumped mass [1xn_eje]
    %   tren.k1: stiffness of primary suspension [1xn_eje]
    %   tren.c1: damping of primary suspension [1xn_eje] 
    % Outputs: 
    %   res(:,1): times
    %   res(:,2): displacement at distance x from the start point of span
    %   res(:,3): velocity at distance x from the start point of span
    %   res(:,4): acceleration at distance x from the start point of span
    %   res(:,5): displacement of  lumped mass
    %   res(:,6): velocity of  lumped mass
    %   res(:,7): acceleration of lumped mass
    % Created by Khanh Nguyen Gia
    % ========================================================================
    """
    #
    # Total time integration
    total_time =(tren.dist[-1] + estructura.L )/ vel + 1.0;
    t = np.arange(0,total_time+estructura.dt,estructura.dt)
    dt=estructura.dt;
    # Matrix of Modal mass of bridge
    Mi = np.zeros((estructura.nmod,estructura.nmod));
    # Matrix of damping of bridge
    Ci = np.zeros((estructura.nmod,estructura.nmod));
    # Matrix of stiffness of bridge
    Ki = np.zeros((estructura.nmod,estructura.nmod));
    #
    param = estructura.param; beta = estructura.beta;
    var('a')
    nmod = estructura.nmod	
    for i in range(nmod):
        f2 = param[i,0]*sin(beta[i]*a)+param[i,1]*cos(beta[i]*a)+param[i,2]*sinh(beta[i]*a)+param[i,3]*cosh(beta[i]*a);
        f3 = f2**2;
        Mi[i,i] = estructura.m*np.double(integrate(f3,(a,0,estructura.L)));
        Ci[i,i] = 2*estructura.xi*estructura.wn[i]*Mi[i,i];
        Ki[i,i] = estructura.wn[i]**2*Mi[i,i];
    #
    # number of axle loads
    ne = len(tren.distv);
    # Mass matrix of lumped mass
    Mbb = np.zeros((ne,ne));
    # Mass matrix of no lumped mass
    Mw = np.zeros((ne,ne));
    # Damping matrix  of vehicle
    Cv = np.zeros((ne,ne));
    # Stiffness matrix of vehicle
    Kv = np.zeros((ne,ne));
    for i in range(ne):
        Mbb[i,i] = tren.mb[i];
        Mw[i,i] = tren.mw[i];
        Cv[i,i] = tren.c1[i];
        Kv[i,i] = tren.k1[i];
    # 
    if len(tren.distv) == len(tren.dist):
        Fw = 9.8*tren.mw;
    else:
        Fw = np.zeros(3*ne);
        for i in range(ne):
            Fw[i*3+2] = tren.mw[i]*9.8*0.25;
            Fw[i*3+1] = tren.mw[i]*9.8*0.5;
            Fw[i*3] = tren.mw[i]*9.8*0.25;
    #	
    Fb = 9.8*(tren.mb+tren.mc);
    # Mass matrix of coupled system
    M0 = np.zeros((estructura.nmod+ne,estructura.nmod+ne));
    M0[estructura.nmod:,estructura.nmod:] = Mbb;
    # Stiffness matrix of coupled system
    K0 = np.zeros((estructura.nmod+ne,estructura.nmod+ne));
    K0[estructura.nmod:,estructura.nmod:] = Kv;
    # Damping matrix of coupled system
    C0 = np.zeros((estructura.nmod+ne,estructura.nmod+ne));
    C0[estructura.nmod:,estructura.nmod:] = Cv;
    # Initital data
    b = np.zeros(estructura.nmod+ne);
    db = np.zeros(estructura.nmod+ne);
    ddb = np.zeros(estructura.nmod+ne);
    Fn = np.zeros(estructura.nmod+ne);
    # modes shape at distance x;
    phi_x = param[:,0]*np.sin(beta*x) + param[:,1]*np.cos(beta*x) + param[:,2]*np.sinh(beta*x) + param[:,3]*np.cosh(beta*x);
    #
    res = np.zeros((len(t),7));
    # Newmark parameters for integration
    gamma = 0.5; be = 0.25;
    # Start loop to calculate the dynamic responses
    for i in range(1,len(t)):
        # Determinate the axles running in the bridge
        distv = vel*t[i] - tren.distv;
        distv[distv>estructura.L] = 0.; distv[distv <0] = 0.;
        distp = vel*t[i] - tren.dist; 
        distp[distp > estructura.L] =0.; distp[distp < 0.] = 0.;
        # Displacement and velocity, acceleration at instance t_{n-1}
        q = b; dq = db; ddq = ddb;
        # Determinate the matrix of modes shape at vt
        phi = param[:,0].reshape(nmod,1)*np.sin(beta.reshape(nmod,1)*distv.reshape(1,len(distv))) + param[:,1].reshape(nmod,1)*np.cos(beta.reshape(nmod,1)*distv.reshape(1,len(distv))) 	+ param[:,2].reshape(nmod,1)*np.sinh(beta.reshape(nmod,1)*distv.reshape(1,len(distv))) + param[:,3].reshape(nmod,1)*np.cosh(beta.reshape(nmod,1)*distv.reshape(1,len(distv)));
        phip = param[:,0].reshape(nmod,1)*np.sin(beta.reshape(nmod,1)*distp.reshape(1,len(distp))) + param[:,1].reshape(nmod,1)*np.cos(beta.reshape(nmod,1)*distp.reshape(1,len(distp))) + param[:,2].reshape(nmod,1)*np.sinh(beta.reshape(nmod,1)*distp.reshape(1,len(distp))) + param[:,3].reshape(nmod,1)*np.cosh(beta.reshape(nmod,1)*distp.reshape(1,len(distp)));
        # Mass matrix
        M = M0; M[:estructura.nmod,:estructura.nmod] = Mi+(phi.dot(Mw)).dot(phi.T);
        K = K0; K[:estructura.nmod,:estructura.nmod] = Ki+(phi.dot(Kv)).dot(phi.T);
        K[:estructura.nmod,estructura.nmod:] = phi.dot(-Kv);
        K[estructura.nmod:,:estructura.nmod] = -Kv.dot(phi.T);
        C = C0; C[:estructura.nmod,:estructura.nmod] = Ci+(phi.dot(Cv)).dot(phi.T);
        C[:estructura.nmod,estructura.nmod:] = phi.dot(-Cv);
        C[estructura.nmod:,:estructura.nmod] = -Cv.dot(phi.T);
        # determinar los coeficientes del metodo de newmark
        c1 = 1/(be*dt**2); c2 = 1/(be*dt); c3 = 1/(2*be) -1;
        c4 = (1-gamma/be); c5 = dt*(1-gamma/(2*be)); c6 = gamma/(be*dt);
    	# 
        Fn[:estructura.nmod] = -phip.dot(Fw); Fn[estructura.nmod:] = -Fb;
        FF = Fn + (c1*M+c6*C).dot(q)+(c2*M-c4*C).dot(dq) + (c3*M-c5*C).dot(ddq);
        KK = c1*M+c6*C+K;
        b = np.linalg.solve(KK,FF);
        ddb = c1*(b-q) - c2*dq - c3*ddq;
        db = dq + dt*(1-gamma)*ddq + gamma*dt*ddb;
        res[i,0] = t[i]
        res[i,1] = np.dot(phi_x,b[:estructura.nmod]);
        res[i,2] = np.dot(phi_x,db[:estructura.nmod]);
        res[i,3] = np.dot(phi_x,ddb[:estructura.nmod]);
        res[i,4] = b[estructura.nmod];
        res[i,5] = db[estructura.nmod];
        res[i,6] = ddb[estructura.nmod];
    return res
# ------------ Skew hiper bridges ----------------------------
def SkewHiperModal(estructura):
    """
    Function that determines the modal parameters of the structure: frequency,
    coefficients of the modal shape of each mode.
    User command: SkewHiperModal(estructura)

    Input: 
        estructura:  structure that contains all necessary information about the structure
        For example:
        estructura.EI = [1xnv] a vector of bending stiffness of hiperstatic structure
        estructura.GJ = [1xnv] a vector of torsion stiffness
        estructura.m = [1xnv] a vector of mass per length
        estructura.xi: damping coefficient
        estructura.nv: numbers of span
        estructura.alpha: skew angle
    Output: 
        estructura.wn: angular frequencies
        estructura.wd damped angular frequencies
        estructura.beta: matrix of value of beta [nmodxnv]
        estructura.lambda: matrix of value of lambda [nmodxnv]
        estructura.param:  a cell of matrix of paramters of modal shape 
    """
    var('wn')
    alpha = estructura.alpha*np.pi/180.
    R1 = estructura.GJ / estructura.EI
    beta = (estructura.m/estructura.EI)**(1./4)*(wn)**(0.5)
    la = (estructura.m*estructura.r**2/estructura.GJ)**(0.5)*wn
    nv = estructura.nv
    A = zeros(6*nv,6*nv)
    # For first span
    A[0:2,0:6] = [[0,1,0,1,0,0],[-beta[0]*sin(alpha),0,-beta[0]*sin(alpha),0,0,cos(alpha)]]
    if estructura.boun == 0:
        A[2,0:6] = [[0, -beta[0]**2, 0, beta[0]**2,R1[0]*la[0]*tan(alpha),0]]
    elif estructura.boun == 1:
        A[2,0:6] = [[beta[0]*cos(alpha),0,beta[0]*cos(alpha),0,0,sin(alpha)]]
    # 
    # For the subsequent spans
    if nv >= 2:
        for i in range(1,nv):
            # u_{i-1}(L_{i-1}) = 0
            A[6*i-3,6*i-6:6*i] = [[sin(beta[i-1]*estructura.L[i-1]), cos(beta[i-1]*estructura.L[i-1]),sinh(beta[i-1]*estructura.L[i-1]),cosh(beta[i-1]*estructura.L[i-1]),0,0]]
            # theta_{i-1}(L_{i-1}) = 0
            A[6*i-2,6*i-6:6*i] = [[-beta[i-1]*sin(alpha)*cos(beta[i-1]*estructura.L[i-1]), beta[i-1]*sin(alpha)*sin(beta[i-1]*estructura.L[i-1]), -beta[i-1]*sin(alpha)*cosh(beta[i-1]*estructura.L[i-1]),-beta[i-1]*sin(alpha)*sinh(beta[i-1]*estructura.L[i-1]),cos(alpha)*sinh(la[i-1]*estructura.L[i-1]),cos(alpha)*cosh(la[i-1]*estructura.L[i-1])]]
            # u_i(0) = 0
            A[6*i-1,6*i:6*(i+1)] = [[0,1,0,1,0,0]]
            # theta_i(0) = 0
            A[6*i,6*i:6*(i+1)] = [[-beta[i]*sin(alpha), 0, -beta[i]*sin(alpha), 0, 0, cos(alpha)]]
            # du_[i-1](L_{i-1}) = du_i(0)
            A[6*i+1,6*(i-1):6*(i+1)] = [[beta[i-1]*cos(alpha)*cos(beta[i-1]*estructura.L[i-1]), -beta[i-1]*cos(alpha)*sin(beta[i-1]*estructura.L[i-1]), beta[i-1]*cos(alpha)*cosh(beta[i-1]*estructura.L[i-1]), beta[i-1]*cos(alpha)*sinh(beta[i-1]*estructura.L[i-1]), sin(alpha)*sinh(la[i-1]*estructura.L[i-1]), sin(alpha)*cosh(la[i-1]*estructura.L[i-1]), -beta[i]*cos(alpha), 0, -beta[i]*cos(alpha), 0, 0, -sin(alpha)]]
            # M_{i-1}(L_{i-1}) = M_i(0)
            A[6*i+2,6*(i-1):6*(i+1)] = [[-estructura.EI[i-1]*beta[i-1]**2*sin(beta[i-1]*estructura.L[i-1]), -estructura.EI[i-1]*beta[i-1]**2*cos(beta[i-1]*estructura.L[i-1]),estructura.EI[i-1]*beta[i-1]**2*sinh(beta[i-1]*estructura.L[i-1]),estructura.EI[i-1]*beta[i-1]**2*cosh(beta[i-1]*estructura.L[i-1]),estructura.GJ[i-1]*la[i-1]*tan(alpha)*cosh(la[i-1]*estructura.L[i-1]),-estructura.GJ[i-1]*la[i-1]*tan(alpha)*sinh(la[i-1]*estructura.L[i-1]), 0, estructura.EI[i]*beta[i]**2, 0, -estructura.EI[i]*beta[i]**2, -estructura.GJ[i]*la[i]*tan(alpha),0]]
    #
    # For the final span
    # u_{nv}(L_{nv}) = 0 
    A[6*nv-3,6*(nv-1):6*nv] = [[sin(beta[-1]*estructura.L[-1]), cos(beta[-1]*estructura.L[-1]),sinh(beta[-1]*estructura.L[-1]),cosh(beta[-1]*estructura.L[-1]), 0, 0]]
    # theta_{nv}(L_{nv}) = 0
    A[6*nv-2,6*(nv-1):6*nv] = [[-beta[-1]*sin(alpha)*cos(beta[-1]*estructura.L[-1]), beta[-1]*sin(alpha)*sin(beta[-1]*estructura.L[-1]),-beta[-1]*sin(alpha)*cosh(beta[-1]*estructura.L[-1]),-beta[-1]*sin(alpha)*sinh(beta[-1]*estructura.L[-1]),cos(alpha)*sinh(la[-1]*estructura.L[-1]),cos(alpha)*cosh(la[-1]*estructura.L[-1])]]
    # 
    if estructura.boun == 0:
        A[6*nv-1,6*(nv-1):6*nv] = [[-beta[-1]**2*sin(beta[-1]*estructura.L[-1]),-beta[-1]**2*cos(beta[-1]*estructura.L[-1]),beta[-1]**2*sinh(beta[-1]*estructura.L[-1]),beta[-1]**2*cosh(beta[-1]*estructura.L[-1]), R1[-1]*la[-1]*tan(alpha)*cosh(la[-1]*estructura.L[-1]), - R1[-1]*la[-1]*tan(alpha)*sinh(la[-1]*estructura.L[-1])]]
    elif estructura.boun == 1:
        A[6*nv-1,6*(nv-1):6*nv] = [[beta[-1]*cos(alpha)*cos(beta[-1]*estructura.L[-1]),-beta[-1]*cos(alpha)*sin(beta[-1]*estructura.L[-1]), beta[-1]*cos(alpha)*cosh(beta[-1]*estructura.L[-1]),beta[-1]*cos(alpha)*sinh(beta[-1]*estructura.L[-1]), sin(alpha)*sinh(la[-1]*estructura.L[-1]),sin(alpha)*cosh(la[-1]*estructura.L[-1])]]
    # 
    # Determine the determinant of matrix A
    fun = det_quick(A)
    # Establish th initial value for numerical solving the det(A)=0
    ca = 0.1
    cb = 1.0
    valor1 = fun.subs(wn,ca)
    valor2 = fun.subs(wn,cb)
    nmod = estructura.nmod
    freq = np.zeros(nmod)
    beta = np.zeros(nmod)
    for i in range(nmod):
        # check the initital value that is closely possible to the solution
        while sign(valor1)==sign(valor2):
            ca =cb
            valor1 = valor2
            cb =cb + 1.0
            valor2 =fun.subs(wn,cb)
	    #
        freq[i] = np.double(nsolve(fun,wn,(ca,cb),solver='bisect',verify=False))
        ca =cb
        valor1 = valor2
        cb = cb + 1.*(i+1)
        valor2 =fun.subs(wn,cb)
    #
    estructura.wn = freq
    beta = np.zeros([nmod,nv]); la = np.zeros([nmod,nv])
    for i in range(nmod):
        for j in range(nv):
            beta[i,j] = (estructura.m[j]*freq[i]**2/estructura.EI[j])**(1./4)
            la[i,j] = (estructura.m[j]*estructura.r[j]**2*freq[i]**2/estructura.GJ[j])**(0.5)
    estructura.wd = freq*np.sqrt(1-estructura.xi**2)
    estructura.la = la
    estructura.beta = beta
    # Determine the coefficients of modal shapes
    lista = []
    for i in range(1,6*nv):
        lista.append(var('c_'+str(i)))
    #
    param = np.zeros([nmod,6*nv])
    for i in range(nmod):
        A1 = A.subs(wn,freq[i])
        B = A1[:-1,1:]; n1,m1 = B.shape
        bb = -A1[:-1,0]
        C = B.col_insert(m1,bb)
        Coef = solve_linear_system_LU(C,lista)
        lista1 = [str(kk) for kk in Coef.keys()]; valores= np.array(list(Coef.values())); 
        num = []
        for kk in lista1:
            a1,a2 = kk.split("_",1)
            num.append(float(a2)) 
        orden = np.argsort(num)
        param[i,0] = 1.0
        param[i,1:] = valores[orden]    
    estructura.param = param 
    # 
    return estructura
#--------------------------------------------------------------------------------------------
#
def SkewHiperMovIntegra(args):
    r"""
    % function that calculates the dynamic response of continous skewbridge due 
    % to a moving load at a distance x from the start point
    % User form: res = SkewHiperMovIntegra(estructura,vel,x,P)
    % Inputs: 
        %   estructura:  structures that contains all information about structure
        %   vel: moving load velocity
        %   x:  Distance at a point from start point 
        %   tren.dist:  vector that contains the distance between axle loading [1xn_eje]
        %   tren.peso:  vector that contains the train loadings [1xn_eje]
    % Outputs: 
        %   res(:,1): times
        %   res(:,2): displacement at distance x from the start point of span
        %   res(:,3): velocity at distance x from the start point of span
        %   res(:,4): acceleration at distance x from the start point of span
        %   res(:,5): rotational angle at distance x
        %   res(:,6): velocity of rotational angle at distance x
        % Created by Khanh Nguyen Gia
    """
    #
    estructura, vel, x, tren, qq = args
    #--------
    if  estructura.portico == 1:
        total_time = (tren.dist[-1]+estructura.L[1])/vel + 1.0
    elif estructura.portico == 0:
        total_time = (tren.dist[-1]+np.sum(estructura.L))/vel + 1.0
    #--------
    nmod = estructura.nmod
    nv = estructura.nv
    t = np.arange(0,total_time+estructura.dt,estructura.dt)
    # Initital conditions at t=0
    b = np.zeros(nmod)
    db = np.zeros(nmod)
    c = np.zeros(nmod)
    dc = np.zeros(nmod)
    P0 = np.zeros(nmod)
    T0 = np.zeros(nmod)
    #  Modal mass
    Mi = np.zeros((nmod,1))
    # Modal rotational mass
    Mti = np.zeros((nmod,1))
    #--------
    param = estructura.param; beta = estructura.beta; la = estructura.la
    alpha1 = (90-estructura.alpha)*np.pi/180.
    k1 = estructura.L/2 * np.tan(alpha1) /(1. + (estructura.EI/estructura.GJ)*np.tan(alpha1)**2)
    #--------
    for j in range(nv):
        Mi += estructura.m[j]*(phi1(estructura.L[j],param[:,6*j:6*j+4],beta[:,[j]]) - phi1(0.,param[:,6*j:6*j+4],beta[:,[j]]))
        Mti += estructura.m[j]*estructura.r[j]**2*(phi2(estructura.L[j],param[:,6*j+4:6*j+6],beta[:,[j]]) - phi2(0.,param[:,6*j+4:6*j+6],beta[:,[j]]))
    #--------
    # Mode shape at distance x
    # look for the recorded point in what span is located
    if  estructura.portico:
        span_num = 2
    else:
        for i in range(nv):
            if np.sum(estructura.L[:i]) <= x and x <= np.sum(estructura.L[:i+1]):
                span_num = i;
    #--------
    # Modal shape at distance x
    # phi1_x = array[nmodx1]
    phi1_x = param[:,[6*span_num]] * np.sin(beta[:,[span_num]]*(x-np.sum(estructura.L[:span_num]))) + param[:,[6*span_num+1]] * np.cos(beta[:,[span_num]]*(x-np.sum(estructura.L[:span_num]))) + param[:,[6*span_num+2]] * np.sinh(beta[:,[span_num]]*(x-np.sum(estructura.L[:span_num]))) + param[:,[6*span_num+3]]*np.cosh(beta[:,[span_num]]*(x-np.sum(estructura.L[:span_num])))
    # phi2_x = array[nmodx1]
    phi2_x = param[:,[6*span_num+4]] * np.sin(la[:,[span_num]]*(x-np.sum(estructura.L[:span_num]))) + param[:,[6*span_num+5]]*np.cos(la[:,[span_num]]*(x-np.sum(estructura.L[:span_num])))
    # Loop for integration
    res = np.zeros([len(t),6])
    distance = tren.dist; loads = tren.peso; L = estructura.L; excentricidad = tren.excentricidad
    for i in range(1,len(t)):
	# print i
        dist = vel*t[i] - distance;  
        if estructura.portico:
            dist[dist > L[1]] = 0.;  dist[dist<0] = 0.
            aa1 = beta[:,[1]]*dist; aa2 = la[:,[1]]*dist
            aux1 = param[:,[6]]*np.sin(aa1) + param[:,[7]]*np.cos(aa1) + param[:,[8]]*np.sinh(aa1) + param[:,[9]]*np.cosh(aa1)
            aux2 = -loads * (k1[1]*(dist/L[1] - dist**2/L[1]**2) + excentricidad)
            aux3 = param[:,[10]]*np.sin(aa2) + param[:,[11]]*np.cos(aa2)
        else:
            dist[dist>np.sum(L)] = 0.;  dist[dist<0] = 0.
            # Modal forces at instance t_{n}
            # determine the loads on corresponding span
            aux1 = np.zeros((nmod,len(dist)))
            aux2 = np.zeros(len(dist))
            aux3 = np.zeros((nmod,len(dist)))
            for j in range(nv):
                kk1 = np.intersect1d(np.where(np.sum(L[:j]) <= dist)[0], np.where(dist<=np.sum(L[:j+1]))[0])
                if  len(kk1) != 0:
                    aa0 = dist[kk1] - np.sum(L[:j])
                    a01 = beta[:,[j]]*aa0; a02 = la[:,[j]]*aa0
                    aa1 = param[:,[6*j]]*np.sin(a01) + param[:,[6*j+1]]*np.cos(a01) + param[:,[6*j+2]]*np.sinh(a01) + param[:,[6*j+3]]*np.cosh(a01)
                    aa2 = param[:,[6*j+4]]*np.sin(a02) + param[:,[6*j+5]]*np.cos(a02)
                    aa3 = -loads[kk1] * (k1[j]*(aa0/L[j] - aa0**2/L[j]**2) + tren.excentricidad)
                    aux1[:,kk1] = aa1
                    aux2[kk1] = aa3
                    aux3[:,kk1] = aa2
        #-----
        # Displacement and velocity at instance t_{n-1}
        q = b; dq =db;
        p = c; dp = dc;
        aux0 = aux1.dot(tren.peso)
        aux4 = aux3.dot(aux2)
        # Modal forces
        Pn = -aux0/Mi[:,0]
        if  estructura.alpha == 0:
            Tn = aux4*0
        else:
            Tn = aux4/Mti[:,0]
        # integration based on the linear interpolation of excitation
        b = estructura.A * q + estructura.B * dq + estructura.C * P0 + estructura.D *Pn
        db = estructura.A1 * q + estructura.B1 * dq + estructura.C1 * P0 + estructura.D1 * Pn
        ddb = Pn - 2*estructura.xi*estructura.wn*db - estructura.wn**2 * b
        c = estructura.A * p + estructura.B * dp + estructura.C * T0 + estructura.D * Tn
        dc = estructura.A1*p + estructura.B1*dp + estructura.C1*T0 + estructura.D1*Tn
        # update the modal forces for next increment
        P0 = Pn; T0 =Tn
        res[i,0] = t[i]
        res[i,1] = b.dot(phi1_x)
        res[i,2] = db.dot(phi1_x)
        res[i,3] = ddb.dot(phi1_x)
        res[i,4] = c.dot(phi2_x)
        res[i,5] = dc.dot(phi2_x)
        #update_progress(float(i+1)/len(t))
    qq.put(1)
    return res
# ----------------
#------------------------------
# ----------------
def  SkewHiperInteIntegra(args):
    r"""
    % function that calculates the dynamic response of bridge due to a lumped mass 
    % (1/4 bogie model) at a distance x from the start point, solving with
    % beta-Newmark method
    % User form: res = SkewHiperInteIntegra(estructura,vel,x,P)
    % Inputs: 
        %   estructura:  structures that contains all information about structure
        %   vel: moving load velocity
        %   x:  Distance at a point from start point 
        %   tren.dist:  vector that contains the sistance between axle loading [1xn_eje]
        %   tren.ms:  lumped mass [1xn_eje]
        %   tren.m:   no lumped mass [1xn_eje]
        %   tren.k1: stiffness of primary suspension [1xn_eje]
        %   tren.c1: damping of primary suspension [1xn_eje] 
    % Outputs: 
        %   res(:,1): times
        %   res(:,2): displacement at distance x from the start point of span
        %   res(:,3): velocity at distance x from the start point of span
        %   res(:,4): acceleration at distance x from the start point of span
        %   res(:,5): displacement of  lumped mass
        %   res(:,6): velocity of  lumped mass
        %   res(:,7): acceleration of lumped mass
        % Created by Khanh Nguyen Gia
    % ========================================================================
    """
    # 
    estructura, vel, x, tren, qq = args
    #--------
    if  estructura.portico == 1:
        total_time = (tren.dist[-1]+estructura.L[1])/vel + 0.5
    elif estructura.portico == 0:
        total_time = (tren.dist[-1]+np.sum(estructura.L))/vel + 0.5
    #--------
    nmod = estructura.nmod
    nv = estructura.nv
    dt = estructura.dt
    t = np.arange(0,total_time+dt,dt)
    #--------
    # Matrix of modal mass of bridge for bending
    Mi = np.zeros([nmod,nmod])
    # Matrix of modal mass of bridge for torsion
    Mti = np.zeros([nmod,nmod])
    # Matrix of modal damping of bridge for bending
    Ci = np.zeros([nmod,nmod])
    # Matrix of modal damping of bridge for torsion
    Cti = np.zeros([nmod,nmod])
    # Matrix of modal stiffness of bridge for bending
    Ki = np.zeros([nmod,nmod])
    # matrix of modal stiffness of bridge for torsion
    Kti = np.zeros([nmod,nmod])
    #
    mii = np.zeros((nmod,1))
    mti = np.zeros((nmod,1))
    #--------
    param = estructura.param; beta = estructura.beta; la = estructura.la
    alpha1 = (90-estructura.alpha)*np.pi/180.
    for j in range(nv):
        mii += estructura.m[j]*(phi1(estructura.L[j],param[:,6*j:6*j+4],beta[:,[j]]) - phi1(0.,param[:,6*j:6*j+4],beta[:,[j]]))
        mti += estructura.m[j]*estructura.r[j]**2*(phi2(estructura.L[j],param[:,6*j+4:6*j+6],beta[:,[j]]) - phi2(0.,param[:,6*j+4:6*j+6],beta[:,[j]]))
    for i in range(nmod):
        Mi[i,i] = mii[i,0]
        Mti[i,i] = mti[i,0]
        Ci[i,i] = 2*estructura.xi*estructura.wn[i]*Mi[i,i]
        Ki[i,i] = estructura.wn[i]**2*Mi[i,i]
        Kti[i,i] = estructura.wn[i]**2*Mti[i,i]
    #--------
    # number of axle
    ne = len(tren.distv)
    # Mass matrix of lumped mass
    Mbb = np.zeros([ne,ne])
    # Mass matrix of no lumped mass
    Mw = np.zeros([ne,ne])
    # Damping matrix of vehicle
    Cv = np.zeros([ne,ne])
    # Stiffness matrix of vehicle
    Kv = np.zeros([ne,ne])
    for i in range(ne):
        Mbb[i,i] = tren.mb[i]
        Mw[i,i] = tren.mw[i]
        Cv[i,i] = tren.c1[i]
        Kv[i,i] = tren.k1[i]
    #
    if  estructura.alpha == 0:  
        #print("here")
        # Mass matrix of coupled system
        M0 = np.zeros([nmod+ne,nmod+ne])
        M0[nmod:,nmod:] = Mbb
        # Stiffness matrix of coupled system
        K0 = np.zeros([nmod+ne,nmod+ne])
        K0[nmod:,nmod:] = Kv
        # Damping matrix of coupled system
        C0 = np.zeros([nmod+ne,nmod+ne])
        C0[nmod:,nmod:] = Cv
        #--------
        # Initial conditions  
        b = np.zeros(nmod+ne)
        db = np.zeros(nmod+ne)
        ddb = np.zeros(nmod+ne)
        Fn = np.zeros(nmod+ne)
        #--------
        # mode shapes at distance x
        # look for the recorded point in which span is located
        if  estructura.portico == 1:
            span_num = 2
        else:
            for i in range(nv):
                if np.sum(estructura.L[:i]) <= x and x <= np.sum(estructura.L[:i+1]):
                    span_num = i;
        #--------
        # Modal shape at distance x
        # phi1_x = array[nmodx1]
        phi1_x = param[:,[6*span_num]] * np.sin(beta[:,[span_num]]*(x-np.sum(estructura.L[:span_num]))) + param[:,[6*span_num+1]] * np.cos(beta[:,[span_num]]*(x-np.sum(estructura.L[:span_num]))) + param[:,[6*span_num+2]] * np.sinh(beta[:,[span_num]]*(x-np.sum(estructura.L[:span_num]))) + param[:,[6*span_num+3]]*np.cosh(beta[:,[span_num]]*(x-np.sum(estructura.L[:span_num])))
        # phi2_x = array[nmodx1]
        phi2_x = param[:,[6*span_num+4]] * np.sin(la[:,[span_num]]*(x-np.sum(estructura.L[:span_num]))) + param[:,[6*span_num+5]]*np.cos(la[:,[span_num]]*(x-np.sum(estructura.L[:span_num])))
        # dynamic responses
        res = np.zeros([len(t),10])
        # Newmark parameters for integration
        gamma = 0.5; be = 0.25
        # determinar los coeficientes del metodo de newmark
        c1 = 1/(be*dt**2); c2 = 1/(be*dt); c3 = 1/(2*be) -1;
        c4 = (1-gamma/be); c5 = dt*(1-gamma/(2*be)); c6 = gamma/(be*dt);      
        #  excentricidad de carga
        e = tren.excentricidad
        # Start loop for integration
        for i in range(1,len(t)):
            distv = vel*t[i] - tren.distv
            distp = vel*t[i] - tren.dist
            if estructura.portico ==1:
                # determines the axles running on the bridge
                distv[distv > estructura.L[1]] = 0.;  distv[distv<0] = 0.
                distp[distp > estructura.L[1]] = 0.;  distp[distp<0] = 0.
                # determines the matrix of modal shape at vt
                a01 = beta[:,[1]]*distv; a02 = beta[:,[1]]*distp; a03 = la[:,[1]]*distv; a04 = la[:,[1]]*distp
                Phi1 = param[:,[6]]*np.sin(a01)+param[:,[7]]*np.cos(a01)+param[:,[8]]*np.sinh(a01)+param[:,[9]]*np.cosh(a01)
                phi1p = param[:,[6]]*np.sin(a02)+param[:,[7]]*np.cos(a02)+param[:,[8]]*np.sinh(a02)+param[:,[9]]*np.cosh(a02)
            else:
                # determines the axles running on the bridge
                distv[distv > np.sum(estructura.L)] = 0.;  distv[distv<0.] = 0.
                distp[distp > np.sum(estructura.L)] = 0.;  distp[distp<0.] = 0.
                # Determines the matrix of modal shape at vt
                Phi1 = np.zeros([nmod,ne])
                phi1p = np.zeros([nmod,len(distp)])
                #---
                for j in range(nv):
                    kk1 = np.intersect1d(np.where(np.sum(estructura.L[:j]) <= distv)[0], np.where(distv<=np.sum(estructura.L[:j+1]))[0])
                    kk2 = np.intersect1d(np.where(np.sum(estructura.L[:j]) <= distp)[0], np.where(distp<=np.sum(estructura.L[:j+1]))[0])
                    if  len(kk1) != 0:
                        aa0 = distv[kk1] - np.sum(estructura.L[:j]); a01 = beta[:,[j]]*aa0;
                        aa1 = param[:,[6*j]]*np.sin(a01) + param[:,[6*j+1]]*np.cos(a01)+param[:,[6*j+2]]*np.sinh(a01) + param[:,[6*j+3]]*np.cosh(a01)
                        Phi1[:,kk1] = aa1
                    if  len(kk2) != 0:
                        aa0p = distp[kk2] - np.sum(estructura.L[:j])
                        a03 = beta[:,[j]]*aa0p
                        aa1p = param[:,[6*j]]*np.sin(a03) + param[:,[6*j+1]]*np.cos(a03)+param[:,[6*j+2]]*np.sinh(a03)+param[:,[6*j+3]]*np.cosh(a03)
                        phi1p[:,kk2] = aa1p
            #---
            # Displacement, velocity and acceleration at instance t_{n-1}
            q = b; dq = db; ddq = ddb;
            # Update the matrices of coupled system
            M = np.zeros((nmod+ne,nmod+ne)); M[:] = M0[:]
            M[:nmod,:nmod] = Mi+(Phi1.dot(Mw)).dot(Phi1.T);
            M[:nmod,nmod:] = Phi1.dot(Mbb)
            #
            K = np.zeros((nmod+ne,nmod+ne)); K[:] = K0[:]
            K[:nmod,:nmod] = Ki; K[nmod:,:nmod] = -Kv.dot(Phi1.T) 
            #
            C = np.zeros((nmod+ne,nmod+ne)); C[:] = C0[:]
            C[:nmod,:nmod] = Ci; C[nmod:,:nmod] = -Cv.dot(Phi1.T)
            # 
            Fn[:nmod] = -phi1p.dot(tren.peso)
            FF = Fn + (c1*M+c6*C).dot(q)+(c2*M-c4*C).dot(dq) + (c3*M-c5*C).dot(ddq);
            KK = c1*M+c6*C+K;
            b = np.linalg.solve(KK,FF);
            ddb = c1*(b-q) - c2*dq - c3*ddq;
            db = dq + dt*(1-gamma)*ddq + gamma*dt*ddb;
            res[i,0] = t[i];
            res[i,1] = b[:nmod].dot(phi1_x);
            res[i,2] = db[:nmod].dot(phi1_x);
            res[i,3] = ddb[:nmod].dot(phi1_x);
            res[i,4] = 0.0;
            res[i,5] = 0.0;
            res[i,6] = 0.0;
            res[i,7] = b[nmod];
            res[i,8] = db[nmod];
            res[i,9] = ddb[nmod];
            update_progress(float(i+1)/len(t))
    elif estructura.alpha != 0:    
        k1 = estructura.L/2 * np.tan(alpha1) /(1. + (estructura.EI/estructura.GJ)*np.tan(alpha1)**2)
        # Mass matrix of coupled system
        M0 = np.zeros([nmod*2+ne,nmod*2+ne])
        M0[2*nmod:,2*nmod:] = Mbb
        # Stiffness matrix of coupled system
        K0 = np.zeros([2*nmod+ne,2*nmod+ne])
        K0[2*nmod:,2*nmod:] = Kv
        # Damping matrix of coupled system
        C0 = np.zeros([2*nmod+ne,2*nmod+ne])
        C0[2*nmod:,2*nmod:] = Cv
        #--------
        # Initial conditions  
        b = np.zeros(2*nmod+ne)
        db = np.zeros(2*nmod+ne)
        ddb = np.zeros(2*nmod+ne)
        Fn = np.zeros(2*nmod+ne)
        #--------
        # mode shapes at distance x
        # look for the recorded point in which span is located
        if  estructura.portico == 1:
            span_num = 2
        else:
            for i in range(nv):
                if np.sum(estructura.L[:i]) <= x and x <= np.sum(estructura.L[:i+1]):
                    span_num = i;
        #--------
        # Modal shape at distance x
        # phi1_x = array[nmodx1]
        phi1_x = param[:,[6*span_num]] * np.sin(beta[:,[span_num]]*(x-np.sum(estructura.L[:span_num]))) + param[:,[6*span_num+1]] * np.cos(beta[:,[span_num]]*(x-np.sum(estructura.L[:span_num]))) + param[:,[6*span_num+2]] * np.sinh(beta[:,[span_num]]*(x-np.sum(estructura.L[:span_num]))) + param[:,[6*span_num+3]]*np.cosh(beta[:,[span_num]]*(x-np.sum(estructura.L[:span_num])))
        # phi2_x = array[nmodx1]
        phi2_x = param[:,[6*span_num+4]] * np.sin(la[:,[span_num]]*(x-np.sum(estructura.L[:span_num]))) + param[:,[6*span_num+5]]*np.cos(la[:,[span_num]]*(x-np.sum(estructura.L[:span_num])))
        # dynamic responses
        res = np.zeros([len(t),10])
        # Newmark parameters for integration
        gamma = 0.5; be = 0.25
        # determinar los coeficientes del metodo de newmark
        c1 = 1/(be*dt**2); c2 = 1/(be*dt); c3 = 1/(2*be) -1;
        c4 = (1-gamma/be); c5 = dt*(1-gamma/(2*be)); c6 = gamma/(be*dt);      
        #  excentricidad de carga
        e = tren.excentricidad
        # Start loop for integration
        for i in range(1,len(t)):
            distv = vel*t[i] - tren.distv
            distp = vel*t[i] - tren.dist
            if estructura.portico ==1:
                # determines the axles running on the bridge
                distv[distv > estructura.L[1]] = 0.;  distv[distv<0] = 0.
                distp[distp > estructura.L[1]] = 0.;  distp[distp<0] = 0.
                # determines the matrix of modal shape at vt
                a01 = beta[:,[1]]*distv; a02 = beta[:,[1]]*distp; a03 = la[:,[1]]*distv; a04 = la[:,[1]]*distp
                Phi1 = param[:,[6]]*np.sin(a01)+param[:,[7]]*np.cos(a01)+param[:,[8]]*np.sinh(a01)+param[:,[9]]*np.cosh(a01)
                phi1p = param[:,[6]]*np.sin(a02)+param[:,[7]]*np.cos(a02)+param[:,[8]]*np.sinh(a02)+param[:,[9]]*np.cosh(a02)
                Phi2 = param[:,[10]]*np.sinh(a03) + param[:,[11]]*np.cosh(a03)
                phi2p = param[:,[10]]*np.sinh(a04) + param[:,[11]]*np.cosh(a04)
                aux2 = k1[1]*(distv/estructura.L[1] - distv**2/estructura.L[1]**2) + e
                aux2p = k1[1]*(distp/estructura.L[1] - distp**2/estructura.L[1]**2) + e
            else:
                # determines the axles running on the bridge
                distv[distv > np.sum(estructura.L)] = 0.;  distv[distv<0.] = 0.
                distp[distp > np.sum(estructura.L)] = 0.;  distp[distp<0.] = 0.
                # Determines the matrix of modal shape at vt
                Phi1 = np.zeros([nmod,ne])
                phi1p = np.zeros([nmod,len(distp)])
                aux2 = np.zeros(ne)
                aux2p = np.zeros(len(distp))
                Phi2 = np.zeros([nmod,ne])
                phi2p = np.zeros([nmod,len(distp)])
                #---
                for j in range(nv):
                    kk1 = np.intersect1d(np.where(np.sum(estructura.L[:j]) <= distv)[0], np.where(distv<=np.sum(estructura.L[:j+1]))[0])
                    kk2 = np.intersect1d(np.where(np.sum(estructura.L[:j]) <= distp)[0], np.where(distp<=np.sum(estructura.L[:j+1]))[0])
                    if  len(kk1) != 0:
                        aa0 = distv[kk1] - np.sum(estructura.L[:j]); a01 = beta[:,[j]]*aa0; a02 = la[:,[j]]*aa0
                        aa1 = param[:,[6*j]]*np.sin(a01) + param[:,[6*j+1]]*np.cos(a01)+param[:,[6*j+2]]*np.sinh(a01) + param[:,[6*j+3]]*np.cosh(a01)
                        aa2 = param[:,[6*j+4]]*np.sinh(a02)+ + param[:,[6*j+5]]*np.cosh(a02)
                        aa3 = k1[j]*(aa0/estructura.L[j] - aa0**2/estructura.L[j]**2) + e
                        Phi1[:,kk1] = aa1
                        Phi2[:,kk1] = aa2
                        aux2[kk1] = aa3
                    if  len(kk2) != 0:
                        aa0p = distp[kk2] - np.sum(estructura.L[:j])
                        a03 = beta[:,[j]]*aa0p; a04 = la[:,[j]]*aa0p
                        aa1p = param[:,[6*j]]*np.sin(a03) + param[:,[6*j+1]]*np.cos(a03)+param[:,[6*j+2]]*np.sinh(a03)+param[:,[6*j+3]]*np.cosh(a03)
                        aa2p = param[:,[6*j+4]]*np.sinh(a04) + param[:,[6*j+5]]*np.cosh(a04)
                        aa3p = k1[j]*(aa0p/estructura.L[j] - aa0p**2/estructura.L[j]**2) + e
                        phi1p[:,kk2] = aa1p
                        phi2p[:,kk2] = aa2p
                        aux2p[kk2] = aa3p
            #---
            # Displacement, velocity and acceleration at instance t_{n-1}
            q = b; dq = db; ddq = ddb;
            # Update the matrices of coupled system
            M = np.zeros((2*nmod+ne,2*nmod+ne)); M[:nmod,:nmod] = Mi+(Phi1.dot(Mw)).dot(Phi1.T);
            M[:estructura.nmod,2*estructura.nmod:] = Phi1.dot(Mbb)
            M[estructura.nmod:2*estructura.nmod,:estructura.nmod] = (Phi2.dot(Mw*aux2)).dot(Phi1.T);
            M[estructura.nmod:2*estructura.nmod,estructura.nmod:2*estructura.nmod] = Mti;
            M[estructura.nmod:2*estructura.nmod, 2*estructura.nmod:] = Phi2.dot(Mbb*aux2)
            #
            K = np.zeros((2*nmod+ne,2*nmod+ne)); K[:] = K0[:]; 
            K[:estructura.nmod,:estructura.nmod] = Ki 
            K[estructura.nmod:estructura.nmod*2,estructura.nmod:estructura.nmod*2] = Kti
            K[estructura.nmod*2:,:estructura.nmod] = -Kv.dot(Phi1.T);
            #
            C = np.zeros((2*nmod+ne,2*nmod+ne)); C[:] = C0[:];
            C[:estructura.nmod,:estructura.nmod] = Ci
            C[estructura.nmod*2:,:estructura.nmod] = -Cv.dot(Phi1.T);
            # 
            # 
            Fn[:estructura.nmod] = -phi1p.dot(tren.peso); Fn[estructura.nmod:estructura.nmod*2] = -phi2p.dot(tren.peso*aux2p);
            #
            FF = Fn + (c1*M+c6*C).dot(q)+(c2*M-c4*C).dot(dq) + (c3*M-c5*C).dot(ddq);
            KK = c1*M+c6*C+K;
            b = np.linalg.solve(KK,FF);
            ddb = c1*(b-q) - c2*dq - c3*ddq;
            db = dq + dt*(1-gamma)*ddq + gamma*dt*ddb;
            res[i,0] = t[i];
            res[i,1] = b[:estructura.nmod].dot(phi1_x);
            res[i,2] = db[:estructura.nmod].dot(phi1_x);
            res[i,3] = ddb[:estructura.nmod].dot(phi1_x);
            res[i,4] = b[estructura.nmod:estructura.nmod*2].dot(phi2_x);
            res[i,5] = db[estructura.nmod:estructura.nmod*2].dot(phi2_x);
            res[i,6] = ddb[estructura.nmod:estructura.nmod*2].dot(phi2_x);
            res[i,7] = b[estructura.nmod*2];
            res[i,8] = db[estructura.nmod*2];
            res[i,9] = ddb[estructura.nmod*2];
            #update_progress(float(i+1)/len(t))
    #
    qq.put(1)
    return  res
#-----------------
# Additional functions for determining the static response of skew multi-span bridge
def element_matrix(E,G,Iz,Iy,J,L,m,Ip,A):
    """
    Function that returns the element mass matrix and stiffness matrix

    User form: element_matrix(E,G,Iz,Iy,J,L,m,Ip,A)     ^ y
                                                        |
    Input:                                              |
        E:  elastic modulus                         o---|---o
        G: shear modulus                            |   |   |
        Iz: moment of inertia about the z axis      |   |   |
        Iy:  moment of inertia about the y axis    -|---+---|----> z
        J: Torsional constant                       |   |   |
        Ip: polar moment                            |   |   |
        L:  element length                          o---|---o
        m: mass per unit length
        A: cross section
    
    Output:
        mb, kb: elemental mass and stiffness matrix (ndarray)

    """
    m1 = np.diag([140,156,156,140*Ip/A,4*L**2,4*L**2])
    m1[4,2] = -22*L; m1[2,4] = m1[4,2]
    m1[5,1] = 22*L; m1[1,5] = m1[5,1]

    m2 = np.diag([70,54,54,70*Ip/A,-3*L**2,-3*L**2])
    m2[4,2] = 13*L; m2[2,4] = -13*L
    m2[5,1] = -13*L; m2[1,5] = 13*L
    m12 = m2; m21 = m2.T

    m3 = np.diag([140,156,156,140*Ip/A,4*L**2,4*L**2])
    m3[4,2] = 22*L; m3[2,4] = m3[4,2]
    m3[5,1] = -22*L; m3[1,5] = m3[5,1]

    mb = np.array(np.bmat('m1 m21; m12 m3'))
    #-----
    k1 = np.diag([E*A/L,12*E*Iz/L**3,12*E*Iy/L**3,G*J/L,4*E*Iy/L,4*E*Iz/L])
    k1[4,2] = -6*E*Iy/L**2; k1[2,4] = k1[4,2]
    k1[5,1] = 6*E*Iz/L**2; k1[1,5] = k1[5,1]

    k12 = np.diag([-E*A/L, -12*E*Iz/L**3,-12*E*Iy/L**3, -G*J/L, 2*E*Iy/L, 2*E*Iz/L])
    k12[4,2] = 6*E*Iy/L**2; k12[2,4] = -k12[4,2]
    k12[5,1] = -6*E*Iz/L**2; k12[1,5] = -k12[5,1]
    
    k21 = np.diag([-E*A/L, -12*E*Iz/L**3, -12*E*Iy/L**3, -G*J/L, 2*E*Iy/L, 2*E*Iz/L])
    k21[4,2] = -6*E*Iy/L**2; k21[2,4] = -k21[4,2]
    k21[5,1] = 6*E*Iz/L**2; k21[1,5] = - k21[5,1]

    k3 = np.diag([E*A/L,12*E*Iz/L**3,12*E*Iy/L**3, G*J/L, 4*E*Iy/L, 4*E*Iz/L])
    k3[4,2] = 6*E*Iy/L**2; k3[2,4] = k3[4,2]
    k3[5,1] = -6*E*Iz/L**2; k3[1,5] = k3[5,1]

    kb = np.array(np.bmat('k1 k12; k21 k3'))

    return mb, kb
# Assembly matrix
def  KM_assembly(estructura):
    """
    Function that returns the assemblied mass and stiffness matrix of the model

    Input: 
       estructura: clase object

    Output:
       M, K:  mass and stiffness matrix (ndarray)
    """
    E = np.zeros(estructura.nv); E[:] = 3.2e10
    G = E/(2*(1+0.25)); Iz = estructura.EI/E; J=estructura.GJ/G;
    m = estructura.m; A = estructura.m/2500.; Iy = 5*Iz; Ip = Iz + Iy
    L = np.sum(estructura.L)    # total length of brige
    Nelems = estructura.nv*100  # total number of elements to be discretized
    spans = {'span'+str(i): np.linspace(np.sum(estructura.L[:i]),np.sum(estructura.L[:i+1]),100+1) for i in range(estructura.nv)}
    Lelems = [spans['span'+str(i)][1]-spans['span'+str(i)][0] for i in range(estructura.nv)]
    longitud = list(spans['span0']); 
    for i in range(1,estructura.nv):
        longitud.extend(x for x in spans['span'+str(i)] if x not in longitud)
    Nnodes = len(longitud)      # number of ndoes
    dofN = 6                    # degree of freedom of each node
    dofs = dofN*Nnodes          # Total degree of freedom
    M = np.zeros((dofs,dofs))   # global mass matrix
    K = np.zeros((dofs,dofs))   # global stiffness matrix
    for i in range(Nelems):
        node1 = i; node2 = i+1
        nspan = int(i/100)
        M_el, K_el = element_matrix(E[nspan],G[nspan],Iz[nspan],Iy[nspan],J[nspan],Lelems[nspan],m[nspan],Ip[nspan],A[nspan])
        t1 = dofN*(node1+1); t2 = dofN*(node2+1)
        # DOF order of nodes
        bg = list(range(t1-dofN,t1)); en = list(range(t2-dofN,t2)); bg.extend(en)
        # Assembly matrix
        M[np.ix_(bg,bg)] += M_el
        K[np.ix_(bg,bg)] += K_el

    return M, K
def BC(estructura):
    Nelems = estructura.nv*100
    Nnodes = Nelems +1
    dofN = 6
    dofs =dofN*Nnodes
    L0 = list(range(dofs))
    # boundary condition at beginning of bridge
    if estructura.boun == 0:
        boun = {'node1': [1, 1, 1, 1, 0, 0, 0]}
    elif  estructura.boun == 1:
        boun = {'node1' : [1, 1, 1, 1, 1, 1, 1]}
    # boundary condition at adjacent supports 
    num_supports = estructura.nv - 1 
    if  num_supports >= 1:
        for i in range(num_supports):
            boun.update({'node'+str(100*(i+1)+1): [100*(i+1)+1, 1, 1, 1, 0, 0, 0]})
    # boundary condition at end of bridge
    if estructura.boun == 0:
        boun.update({'node'+str(estructura.nv*100+1): [estructura.nv*100+1, 1,1,1, 0, 0, 0]})
    elif estructura.boun == 1:
        boun.update({'node'+str(estructura.nv*100+1): [estructura.nv*100+1, 1,1,1, 1, 1, 1]})
    LL = []
    for k,v  in boun.items():
        for j in range(dofN):
            if v[1+j] == 1:
                LL.append(v[0]*dofN - (dofN-j))

    L = [x for x in L0 if x not in LL]
    return L
#-----------------
def SkewHiperStatic(estructura,tren):
    """
    ----------------------------------------------------------------------------
    Function that returns the maximum static response at point x of bridge under 
    all combinations of train loading positions
    
    Input: 
       estructura

    Output:
       estructura.u_max
       estructura.u_static
    ----------------------------------------------------------------------------   
    """
    # total number of elements to be discretized
    Nelems = estructura.nv*100  
    spans = {'span'+str(i): np.linspace(np.sum(estructura.L[:i]),np.sum(estructura.L[:i+1]),100+1) for i in range(estructura.nv)}
    Lelems = [spans['span'+str(i)][1]-spans['span'+str(i)][0] for i in range(estructura.nv)]
    longitud = list(spans['span0']); 
    for i in range(1,estructura.nv):
        longitud.extend(x for x in spans['span'+str(i)] if x not in longitud)
    Nnodes = len(longitud)      # number of ndoes
    dofN = 6    # degree of freedom per node
    dofs = dofN*Nnodes  # total degree of freedom 
    # find node that corresponds to estructura.x (recorded point)
    n1 = [i for i in range(len(longitud)) if longitud[i]<=estructura.x][-1]
    n2 = n1+1; index0 = [n1,n2]; xp = [longitud[n1],longitud[n2]]
    # Build mass and stiffness matrix
    M, K = KM_assembly(estructura)
    # set boundary conditions
    index1 = BC(estructura)
    # increment in posistion of train loads
    ds = min(estructura.L/100)
    # number of increment
    if estructura.portico:
        Np = int(np.ceil((tren.dist[-1]+estructura.L[1])/ds))
    else:
        Np = int(np.ceil((tren.dist[-1]+np.sum(estructura.L))/ds))
    # static response for all combination of trian loading positions
    res = np.zeros(Np)
    # introduce the additional baoundary conditions due to the skewness
    alpha = estructura.alpha*np.pi/180
    A1 = np.zeros((estructura.nv+1,dofs))
    for kk in range(estructura.nv+1):
        n1, n2 = 100*kk*dofN+3, 100*kk+5
        A1[kk,n1] = 1.; A1[kk,n2] = np.tan(alpha)
    # calculation
    KK = np.vstack((K[np.ix_(index1,index1)],A1[:,index1]))
    # pseudo-inverse matrix
    KK1 = np.linalg.pinv(KK)
    for ii in range(Np):
        y0 = np.zeros(dofs)
        distance = ii*ds - tren.dist
        if estructura.portico == 0:
            distance[distance>np.sum(estructura.L)]=0.; distance[distance<0] = 0.
        elif  estructura.portico==1:
            distance[distance>estructura.L[1]] = 0.; distance[distance<0] = 0.
        aux1 = np.where(distance>0.)[0]
        dist = distance[aux1]; P = -tren.peso[aux1]
        if len(dist) > 0:
            F = np.zeros(dofs)
            for j  in range(len(dist)):
                if estructura.portico:
                    node1 = np.where(longitud < estructura.L[0] + dist[j])[0][-1]
                    node2 = node1 + 1
                    Lelem = longitud[node2] - longitud[node1]
                    aa = estructura.L[0] + dist[j] - longitud[node1]
                    bb = longitud[node2] - (estructura.L[0]+dist[j])
                else:
                    node1 = np.where(longitud < dist[j])[0][-1]
                    node2 = node1 + 1
                    Lelem = longitud[node2] - longitud[node1]
                    aa = dist[j] - longitud[node1]
                    bb = longitud[node2] - dist[j]
                # factor of load distribution to node1 and node2
                factor1 = bb/Lelem; factor2 = aa/Lelem; 
                F[dofN*node1+1] = P[j]*factor1
                F[dofN*node2+1] = P[j]*factor2
            FF = np.append(F[index1], np.zeros(estructura.nv+1))
            # solve linear system of equations
            y0[index1] = KK1.dot(FF)
            fp = y0[dofN*np.array(index0)+1]
            res[ii] = np.interp(estructura.x, xp, fp)
        update_progress(float(ii+1)/Np)
    estructura.u_max = np.max(abs(res))
    estructura.u_static = res
    return estructura
# some additional functions for saving and plotting data
#-----------------
def cmovprintfhistoria(estructura,tren,vel,tra,res,tdmax,tvmax,tamax, dmax,vmax,amax,project_name):
    path = getcwd()
    if  tra != 0:
        name = '-mld-'
    else:
        name = '-ml-'
    datfile = path + '/' + project_name + '/' + tren.nombre+ name + 'skew'+str(estructura.alpha)+'-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-m'+str(estructura.m/1.e03)+'t-nm'+str(estructura.nmod) +'-v'+str(vel)+'.dat'
    fid = open(datfile,'w')
    fid.write("# Case %s\n" % (datfile))
    fid.write("# Train = %s\n" % (tren.nombre))
    fid.write("# v = %.1f km/h\n" % (vel))
    fid.write("# L = %s m\n" % (str(estructura.L)))
    fid.write("# EI = %s N.m2\n" % (str(estructura.EI)))
    fid.write("# GJ = %s N.m2\n" % (str(estructura.GJ)))
    fid.write("# rho = %s t/m\n" % (str(estructura.m/1.e03)))
    fid.write("# r = %s m\n" % (str(estructura.r)))
    fid.write("# skew angle = %.2f\n" % (estructura.alpha))
    fid.write("# zeta = %.2f \n" % (estructura.xi))
    fid.write("# f0 = %.3f Hz\n" % (estructura.wn[0]/2/np.pi))
    fid.write("# Number of modes = %i\n" % (estructura.nmod))
    fid.write("# Time increment = %.3f (s)\n" % (estructura.dt))
    fid.write("# \n")
    fid.write("# dmax = %12.5g at %12.5g s\n" % (dmax,tdmax))
    fid.write("# vmax = %12.5g at %12.5g s\n" % (vmax,tvmax))
    fid.write("# amax = %12.5g at %12.5g s\n" % (amax,tamax))
    fid.write("# \n")
    fid.write("#    time    d   v   a\n")
    for i in range(len(res)):
        fid.write("%5.4g  %5.4g   %5.4g   %5.4g   \n" % (res[i,0],res[i,1],res[i,2],res[i,3]))
    fid.close()
    return
#----------
def cmovprintfbarrido(estructura,tren,velo,tra,res,sta,vdmax,vvmax,vamax,dmax,vmax,amax,project_name):
    path = getcwd()
    if  tra != 0:
        name = '-mld-'
    else:
        name = '-ml-'
    if  velo.final <= tren.vmax:
        datfile = path + '/' + project_name + '/' + tren.nombre+'-envelope'+ name + 'skew'+str(estructura.alpha)+'-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-m'+str(estructura.m/1.e03)+'t-nm'+str(estructura.nmod) +'-v'+str(velo.ini)+'-'+str(velo.final) + '.dat'
    else:
        datfile = path + '/' + project_name + '/' + tren.nombre+'-envelope'+ name + 'skew'+str(estructura.alpha)+'-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-m'+str(estructura.m/1.e03)+'t-nm'+str(estructura.nmod) +'-v'+str(velo.ini)+'-'+str(tren.vmax) + '.dat'
    #
    fid = open(datfile,'w')
    fid.write("# Envelope: %s\n" %(datfile))
    fid.write("# Train = %s\n" % (tren.nombre))
    fid.write("# v initial = %.2f km/h \n" % (velo.ini))
    if  velo.final <= tren.vmax:
        fid.write("# v final = %.2f km/h \n" % (velo.final))
    else:
        fid.write("# v final = %.2f km/h \n" % (tren.vmax))
    fid.write("# speed increment = %.2f km/h \n" %(velo.inc))
    fid.write("# rho = %s t/m\n" % (str(estructura.m/1.e03)))
    fid.write("# r = %s m\n" % (str(estructura.r)))
    fid.write("# skew angle = %.2f\n" % (estructura.alpha))
    fid.write("# zeta = %.2f \n" % (estructura.xi))
    fid.write("# f0 = %.3f Hz\n" % (estructura.wn[0]/2/np.pi))
    fid.write("# Number of modes = %i\n" % (estructura.nmod))
    fid.write("# Time increment = %.3f (s)\n" % (estructura.dt))
    fid.write("# \n")
    fid.write("# dmax = %12.5g at %12.5g km/h\n" % (dmax,vdmax))
    fid.write("# vmax = %12.5g at %12.5g km/h\n" % (vmax,vvmax))
    fid.write("# amax = %12.5g at %12.5g km/h\n" % (amax,vamax))
    if sta != 0.:
        fid.write("# dmax static = %12.5g m\n" % (sta))
    fid.write("# \n")
    fid.write("# vel    dmax    vmax    amax    DAF\n")
    for i in range(len(res)):
       fid.write("%8.4g    %8.4g   %8.4g   %8.4g    %8.4g\n" % (res[i,0],res[i,1],res[i,2],res[i,3],res[i,1]/sta))
    fid.close()
    return
#----------
def cmovprintfstatic(estructura,tren,tra,ressta, project_name):
    path = getcwd()
    if  tra != 0:
        name = '-mld-'
    else:
        name = '-ml-'
    datfile = path + '/' + project_name + '/' + tren.nombre+ name + 'skew'+str(estructura.alpha)+'-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-m'+str(estructura.m/1.e03)+'t-nm'+str(estructura.nmod) +'-static.dat'
    fid = open(datfile,'w')
    fid.write("# Case %s\n" % (datfile))
    fid.write("# Train = %s\n" % (tren.nombre))
    fid.write("# L = %s m\n" % (str(estructura.L)))
    fid.write("# EI = %s N.m2\n" % (str(estructura.EI)))
    fid.write("# GJ = %s N.m2\n" % (str(estructura.GJ)))
    fid.write("# rho = %s t/m\n" % (str(estructura.m/1.e03)))
    fid.write("# r = %s m\n" % (str(estructura.r)))
    fid.write("# skew angle = %.2f\n" % (estructura.alpha))
    fid.write("# zeta = %.2f \n" % (estructura.xi))
    fid.write("# f0 = %.3f Hz\n" % (estructura.wn[0]/2/np.pi))
    for i in range(len(ressta.u_static)):
        fid.write('%i\t%5.4g\n' % (i, ressta.u_static[i]))
    fid.close()
    return
#-----------
def cmovplothistoria(estructura,tren,vel,res,tdmax,tamax,dmax,amax,project_name):
    #
    path = getcwd()
    basename = path + '/' + project_name + '/' +tren.nombre+'-cmov-skew'+str(estructura.alpha)+'-L'+str(estructura.L)+'-v'+str(vel)
    titlename = tren.nombre + '-cmov-skew'+str(estructura.alpha) + '-L' + str(estructura.L) + '-v' + str(vel)
    epsfile = basename + '.eps'
    plt.figure()
    plt.subplot(211)
    plt.plot(res[:,0],res[:,3])
    if amax == max(res[:,3]):
        plt.plot(tamax,amax,'or',ms = 5.)
    else:
        plt.plot(tamax,-amax,'or',ms = 5.)
    plt.title(titlename)
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'Acceleration (m/s$^2$)')
    leyenda = r'amax = %.3f m/s$^2$ at t = %.3f s' % (amax,tamax)
    tmedio = 0.5*(res[0,0]+res[-1,0])
    if tamax > tmedio:
        if amax == max(res[:,3]):
            plt.text(tamax, amax,leyenda, horizontalalignment='right')
        else:
            plt.text(tamax,-amax,leyenda, horizontalalignment='right')
    else:
        if amax == max(res[:,3]):
            plt.text(tamax, amax,leyenda, horizontalalignment='left')
        else:
            plt.text(tamax,-amax,leyenda, horizontalalignment='left')
    #---            
    plt.subplot(212)
    plt.plot(res[:,0],1000*res[:,1])
    if  dmax == max(res[:,1]):
        plt.plot(tdmax,1000*dmax,'or',ms=5.)
    else:
        plt.plot(tdmax,-1000*dmax,'or',ms=5.)
    # plt.title(titlename + ': desplazamiento en el centro del vano')
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'Displacement (mm)')
    leyenda = r'dmax = %.3f mm at t = %.3f s' % (1000*dmax,tdmax)
    if tdmax > tmedio:
        if dmax == max(res[:,1]):
            plt.text(tdmax, dmax*1000,leyenda, horizontalalignment='right')
        else:
            plt.text(tdmax,-dmax*1000,leyenda, horizontalalignment='right')
    else:
        if dmax == max(res[:,1]):
            plt.text(tdmax, dmax*1000,leyenda, horizontalalignment='left')
        else:
            plt.text(tdmax,-dmax*1000,leyenda, horizontalalignment='left')
    # 
    plt.savefig(epsfile)  
    return
#------------
def cmovplotbarrido(estructura,tren,velo,tra,res, vdmax, vamax, dmax, amax,project_name):
    path = getcwd()
    if  tra != 0:
        name = '-mld-'
    else:
        name = '-ml-'
    if velo.final <= tren.vmax:
        basename = path + '/' + project_name + '/' + tren.nombre+'-envelope'+ name + 'skew'+str(estructura.alpha)+'-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-m'+str(estructura.m/1.e03)+'t-nm'+str(estructura.nmod) +'-v'+str(velo.ini)+'-'+str(velo.final) 
        titlename = tren.nombre+'-envelope'+name+'skew' + str(estructura.alpha) + '-L' + str(estructura.L)+'-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-v' + str(velo.ini) + '-' + str(velo.final)
    else:
        basename = path + '/' + project_name + '/' + tren.nombre+'-envelope'+ name + 'skew'+str(estructura.alpha)+'-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-m'+str(estructura.m/1.e03)+'t-nm'+str(estructura.nmod) +'-v'+str(velo.ini)+'-'+str(tren.vmax) 
        titlename = tren.nombre+'-envelope'+name +'skew' + str(estructura.alpha) + '-L' + str(estructura.L)+'-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-v' + str(velo.ini) + '-' + str(tren.vmax)
    #
    epsfile = basename + '.eps'
    plt.figure()
    plt.subplot(211)
    plt.plot(res[:,0],res[:,3])
    plt.plot(vamax,amax,'or',ms=10.)
    plt.xlabel(r'Velocity (km/h)')
    plt.ylabel(r'Acceleration (m/s$^2$)')
    plt.title(titlename)
    leyenda = r'amax = %.3f m/s$^2$ at v = %.1f km/h' %(amax,vamax)
    vmedio = 0.5*(res[0,0]+res[-1,0])
    if  vamax > vmedio:
        plt.text(vamax,amax,leyenda,horizontalalignment='right')
    else:
        plt.text(vamax,amax,leyenda,horizontalalignment='left')
    plt.subplot(212)
    plt.plot(res[:,0],1.e03*res[:,1])
    plt.plot(vdmax,dmax*1e03,'or',ms=10.)
    plt.xlabel(r'Velocity (km/h)')
    plt.ylabel(r'Displacement (mm)')
    # plt.title(titlename+': maximo desplazamiento')
    leyenda = r'dmax = %.3f mm at v = %.1f km/h' %(dmax*1.e3,vdmax)
    if vdmax > vmedio:
        plt.text(vdmax,dmax*1000,leyenda,horizontalalignment='right')
    else:
        plt.text(vdmax,dmax*1000,leyenda,horizontalalignment='left')
    plt.savefig(epsfile)
    return

#--------------
def cmovisoe(trenid,estructuras,dt, velo, tra, sta, vb, cp , gr, nc, e, project_name, add_train, progressbar):
    r"""
    % Function that calculates the dynamic responseof bridge due to moving load
    % for general case of bridge including the skew bridges (with introducing 
            % the skew angle)
    % User form:  cmovisoe(trenid, estructura, dt, velocidad, tra, sta, vb, cp, gr, nc, e)
    % Input data:
    %   trenid: identificative number of train 
    %   estructura: Structure that contains all information about the bridges
    %       estructura.EI:  bending stiffness
    %       estructura.GJ:  torsion stiffness
    %       estructura.alpha: skew angle
    %       estructura.m: mass per unit length
    %       estructura.r: radius of gyration
    %       estructura.L: Length of bridge
    %       estructura.nmod: number of modes of vibration considered in
    %       calculation
    %       estructura.nf: indicator of frequency level: 'h'-high, 'm'-media,
    %       'l' - low
    %       estructura.nm: indicator of mass level: 'h'-high, 'm'-media,
    %       'l'-low
    %       estructura.nv: number of spans of the bridge
    %       estructura.xi: structural damping
    %       estructura.simply=1 use simplification model for skew bridge, =0 use 
    %       full analytical model bridge
    %   dt: time increment
    %   velo: structure that contains information about the range of
    %   velocity that trains runs on the bridges.
    %       if there is not range of vehicle velocity:
    %           velocity contains only the value of a vehicle velocity in km/h
    %       if there is range of vehicle velocity:
    %           velocidad.inicial: initial velocity
    %           velocidad.final: final velocity
    %           velocidad.inc:  increment of speed km/h
    %   tra: distance between sleepers in m
    %   sta: if sta =1, the static response will be calculated, elseif sta = 0
    %   is not.
    %   vb: verbose mode. if vb =1 the results is saved for each velocity,
    %   elseif vb = 0 is not
    %   cp: if cp=1 compress the file of results, else is not
    %   gr: graphic mode, if gr =1 is yes, else is not
    %   nc: number of core to be used in parallel computing
    %   estructura.x: distance of the point that we want to obtain the dynamic response
    %   e: excentricity of load respect to the center line
    % Output data: files of results of displacement, velocity, acceleration 
    % Created by Jose M. Goicolea abr 2008, updated by P. Antolin
    % Ultimate update and modified by Khanh Nguyen Gia Jul 2016
    % =========================================================================
    """
    # Check the velo is a class or only one value
    if not isinstance(velo,cell_struct):
        # print('is calculating in this point')
        if isinstance(estructuras,list) and len(estructuras) >1:
            #print('Se definio mas de una estructura para un caso sencillo')
            return
        elif isinstance(estructuras,cell_struct):
            estructura = estructuras
        if isinstance(trenid,list) and len(trenid) > 1:
            #print('More than one train is defined')
            return
        elif isinstance(trenid,int):
            #print('is calculating in this point')
            tren = train(); tren.cmovtrain(trenid,tra,add_train); tren.excentricidad = e
        else:
            #print('is calculating in this point')
            tren = train(); tren.cmovtrain(trenid[0],tra, add_train); tren.excentricidad = e
        estructura.dt = dt
        vel = velo/3.6
        # Determine the modal parameters and the responses for the bridge
        mm = multiprocessing.Manager(); qq = mm.Queue()
        if estructura.nv == 1:
            if estructura.simply ==1:
                SpringModal(estructura)
                ParaIntegraExact(estructura)
                res = SpringMovIntegra(estructura,vel,estructura.x, tren)
            elif estructura.simply == 0:
                SkewModal(estructura)
                ParaIntegraExact(estructura)
                res = SkewMovIntegra((estructura,vel,estructura.x,tren,qq))
        elif estructura.nv > 1:
            SkewHiperModal(estructura)
            ParaIntegraExact(estructura)
            res = SkewHiperMovIntegra((estructura,vel,estructura.x,tren,qq))
        # Determine the maximum responses
        # Displacement
        dmax = np.max(abs(res[:,1]))
        id1 = np.argmax(abs(res[:,1])); tdmax = res[id1,0]
        # Velocity
        vmax = np.max(abs(res[:,2]))
        id2 = np.argmax(abs(res[:,2])); tvmax = res[id2,0]
        # Acceleration
        amax = np.max(abs(res[:,3]))
        id3 = np.argmax(abs(res[:,3])); tamax = res[id3,0]
        # save the results
        cmovprintfhistoria(estructura,tren,vel,tra,res,tdmax,tvmax,tamax, dmax,vmax,amax, project_name)
        # plot the results
        if gr:
            cmovplothistoria(estructura,tren,vel,res,tdmax,tamax,dmax,amax)
        #
    else:
        v0 = np.arange(velo.ini,velo.final+velo.inc,velo.inc)
        v0 = v0/3.6
        if  isinstance(trenid,int):
            num_trains=1
            trenes = train(); trenes.cmovtrain(trenid,tra, add_train); trenes.excentricidad = e
        elif isinstance(trenid,list):
            num_trains = len(trenid)
            trenes = [train() for i in range(num_trains)]
            for i in range(num_trains):
                trenes[i].cmovtrain(trenid[i],tra, add_train)
                trenes[i].excentricidad = e
        # Determine cell of structures
        if isinstance(estructuras,list):
            num_struct = len(estructuras)
            for i in range(len(estructuras)):
                if  estructuras[i].nv == 1:
                    if estructuras[i].simply == 1:
                        SpringModal(estructuras[i]); estructuras[i].dt = dt
                        ParaIntegraExact(estructuras[i])
                    else:
                        SkewModal(estructuras[i]); estructuras[i].dt = dt
                        ParaIntegraExact(estructuras[i])
                else:
                    SkewHiperModal(estructuras[i]); estructuras[i].dt = dt
                    ParaIntegraExact(estructuras[i])
        elif isinstance(estructuras,cell_struct):
            num_struct = 1
            if  estructuras.nv == 1:
                if estructuras.simply == 1:
                    SpringModal(estructuras); estructuras.dt = dt
                    ParaIntegraExact(estructuras)
                else:
                    SkewModal(estructuras); estructuras.dt = dt
                    ParaIntegraExact(estructuras)
            else:
                SkewHiperModal(estructuras); estructuras.dt = dt
                ParaIntegraExact(estructuras)
        #---
        # May be that all trains do not run woth the same velocity. So we need 
        # determine the total number cases that will be calculated
        ncases = 0
        for i in range(num_trains):
            for j in range(num_struct):
                for k in range(len(v0)):
                    if isinstance(trenid,list):
                        if  v0[k] > trenes[i].vmax/3.6:
                            break
                    elif isinstance(trenid,int):
                        if  v0[k] > trenes.vmax/3.6:
                            break
                    ncases += 1
        # print(ncases)
        #---
        # Create an argument that contains all informations of all cases to be calculated
        argumentos = [cell_struct() for i in range(ncases)]
        n = 0
        for i in range(num_trains):
            for j in range(num_struct):
                for k in range(len(v0)):
                    if isinstance(trenid,list):
                        if  v0[k] > trenes[i].vmax/3.6:
                            break
                    elif isinstance(trenid,int):
                        if  v0[k] > trenes.vmax/3.6:
                            break
                    n += 1
                    if isinstance(trenid,list):
                        argumentos[n-1].tren = trenes[i]
                    elif  isinstance(trenid,int):
                        argumentos[n-1].tren = trenes
                    if isinstance(estructuras,list):
                        argumentos[n-1].estructura = estructuras[j]
                        argumentos[n-1].x = estructuras[j].x
                    else:
                        argumentos[n-1].estructura = estructuras
                        argumentos[n-1].x = estructuras.x
                    argumentos[n-1].vel = v0[k]
        #  Determine the number of core used in the parallel calculation
        if  nc > multiprocessing.cpu_count():
            num_cores = multiprocessing.cpu_count()
        else:
            num_cores = nc
        # print(num_cores)
        if  num_cores == 1:
            # Create a cell 1xncases that contains all responses of all cases
            res =[cell_struct() for i in range(ncases)]
            progressbar.setMinimum(1)
            progressbar.setMaximum(ncases)
            mm = multiprocessing.Manager(); qq = mm.Queue()
            for i in range(ncases):
                if argumentos[i].estructura.nv == 1:
                    res[i] = SkewMovIntegra((argumentos[i].estructura,argumentos[i].vel,argumentos[i].x,argumentos[i].tren,qq))
                    progressbar.setValue(float(i+1))
                    QApplication.processEvents()
                else:
                    res[i] = SkewHiperMovIntegra((argumentos[i].estructura,argumentos[i].vel,argumentos[i].x,argumentos[i].tren,qq))
                    progressbar.setValue(float(i+1))
                    QApplication.processEvents()
        else:
            progressbar.setMinimum(1)
            progressbar.setMaximum(ncases)
            mm = multiprocessing.Manager(); qq = mm.Queue()
            if estructuras.nv == 1:
                mypool = multiprocessing.Pool(num_cores)
                kk = [(argumentos[i].estructura,argumentos[i].vel,argumentos[i].x,argumentos[i].tren, qq)  for i in range(ncases)]
                output = mypool.map_async(SkewMovIntegra, kk)
                while not output.ready():
                    progressbar.setValue(qq.qsize())
                    QApplication.processEvents()
                res = output.get()
                mypool.close()
                #res = Parallel(n_jobs=num_cores)(delayed(SkewMovIntegra) (argumentos[i].estructura,argumentos[i].vel,argumentos[i].x,argumentos[i].tren)  for i in range(ncases))
                #progressbar.setValue(ncases)
                #QApplication.processEvents()
            else:
                mypool = multiprocessing.Pool(num_cores)
                kk = [(argumentos[i].estructura,argumentos[i].vel,argumentos[i].x,argumentos[i].tren, qq)  for i in range(ncases)]
                output = mypool.map_async(SkewHiperMovIntegra, kk)
                while not output.ready():
                    progressbar.setValue(qq.qsize())
                    QApplication.processEvents()
                res = output.get()
                mypool.close()
                #res = Parallel(n_jobs=num_cores)(delayed(SkewHiperMovIntegra) (argumentos[i].estructura,argumentos[i].vel,argumentos[i].x,argumentos[i].tren) for i in range(ncases))
                #progressbar.setValue(ncases)
                #QApplication.processEvents()
        #------
        # Determine the static responses if is requested
        if sta:
            if  num_cores == 1:
                ressta = [cell_struct() for i in range(num_trains)]
                if estructuras.nv == 1:
                    if estructuras.simply == 1:
                        for i in range(num_trains):
                            SpringStatic(estructuras,trenes[i])
                            ressta[i].u_max = estructuras.u_max
                            ressta[i].u_static = estructuras.u_static
                            ressta[i].phi1 = estructuras.phi1
                            ressta[i].phi2 = estructuras.phi2
                            ressta[i].phi1_max = estructuras.phi1_max
                            ressta[i].phi2_max = estructuras.phi2_max
                            cmovprintfstatic(estructuras, trenes[i], tra, ressta[i], project_name)
                    else:
                        for i in range(num_trains):
                            SkewStatic(estructuras,trenes[i])
                            ressta[i].u_max = estructuras.u_max
                            ressta[i].u_static = estructuras.u_static
                            ressta[i].phi1 = estructuras.phi1
                            ressta[i].phi2 = estructuras.phi2
                            ressta[i].phi1_max = estructuras.phi1_max
                            ressta[i].phi2_max = estructuras.phi2_max
                            cmovprintfstatic(estructuras, trenes[i], tra, ressta[i], project_name)
                elif estructuras.nv > 1:
                    for i in range(num_trains):
                        SkewHiperStatic(estructuras,trenes[i])
                        ressta[i].u_max = estructuras.u_max
                        ressta[i].u_static = estructuras.u_static
                        cmovprintfstatic(estructuras, trenes[i], tra, ressta[i], project_name)
            else:
                if estructuras.nv == 1:
                    if estructuras.simply == 1:
                        mypool = multiprocessing.Pool(num_cores)
                        output = [mypool.apply_async(SpringStatic, args=(estructuras,trenes[i])) for i in range(num_trains)]
                        ressta = [p.get() for p in output]
                        mypool.close()
                        #ressta = Parallel(n_jobs=num_cores)(delayed(SpringStatic) (estructuras,trenes[i]) for i in range(num_trains))
                        for i in range(num_trains):
                            cmovprintfstatic(estructuras, trenes[i], tra, ressta[i], project_name)
                    else:
                        mypool = multiprocessing.Pool(num_cores)
                        output = [mypool.apply_async(SkewStatic, args=(estructuras,trenes[i])) for i in range(num_trains)]
                        ressta = [p.get() for p in output]
                        mypool.close()
                        #ressta = Parallel(n_jobs=num_cores)(delayed(SkewStatic) (estructuras,trenes[i]) for i in range(num_trains))
                        for i in range(num_trains):
                            cmovprintfstatic(estructuras, trenes[i], tra, ressta[i], project_name)
                else:
                    mypool = multiprocessing.Pool(num_cores)
                    output = [mypool.apply_async(SkewHiperStatic, args=(estructuras,trenes[i])) for i in range(num_trains)]
                    ressta = [p.get() for p in output]
                    mypool.close()
                    #ressta = Parallel(n_jobs=num_cores)(delayed(SkewHiperStatic) (estructuras,trenes[i]) for i in range(num_trains))
                    for i in range(num_trains):
                        cmovprintfstatic(estructuras, trenes[i], tra, ressta[i], project_name)
        #------------
        # postprocess
        #------------
        n = 0
        bmax = np.zeros([len(v0),4])
        for i in range(num_trains):
            for j in range(num_struct):
                for k in range(len(v0)):
                    if isinstance(trenid,list):
                        if  v0[k] > trenes[i].vmax/3.6:
                            break
                    elif isinstance(trenid,int):
                        if  v0[k] > trenes.vmax/3.6:
                            break
                    n +=1
                    # maximum responses
                    # displacement
                    dmax = np.max(abs(res[n-1][:,1]))
                    id1 = np.argmax(abs(res[n-1][:,1]))
                    tdmax = res[n-1][id1,0]
                    # velocity
                    vmax = np.max(abs(res[n-1][:,2]))
                    id2 = np.argmax(abs(res[n-1][:,2]))
                    tvmax = res[n-1][id2,0]
                    # acceleration
                    amax = np.max(abs(res[n-1][:,3]))
                    id3 = np.argmax(abs(res[n-1][:,3]))
                    tamax = res[n-1][id3,0]
                    bmax[k,0] = v0[k]*3.6
                    bmax[k,1] = dmax 
                    bmax[k,2] = vmax
                    bmax[k,3] = amax
                    if  vb:
                        cmovprintfhistoria(argumentos[n-1].estructura,argumentos[n-1].tren,v0[k]*3.6,tra,res[n-1],tdmax,tvmax,tamax,dmax,vmax,amax, project_name)
                # 
                if k!= 0:
                    if k < len(v0)-1:
                        dmax = np.max(bmax[:k,1])
                        id1 = np.argmax(bmax[:k,1])
                        vdmax = bmax[id1,0]
                        vmax = np.max(bmax[:k,2])
                        id2 = np.argmax(bmax[:k,2])
                        vvmax = bmax[id2,0]
                        amax = np.max(bmax[:k,3])
                        id3 = np.argmax(bmax[:k,3])
                        vamax = bmax[id3,0]
                        if sta:
                            cmovprintfbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k,:],ressta[i].u_max,vdmax,vvmax,vamax,dmax,vmax,amax,project_name)
                        else:
                            cmovprintfbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k,:],0.0,vdmax,vvmax,vamax,dmax,vmax,amax,project_name)
                        if gr:
                            cmovplotbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k,:], vdmax, vamax, dmax, amax,project_name)
                    else:
                        dmax = np.max(bmax[:k+1,1])
                        id1 = np.argmax(bmax[:k+1,1])
                        vdmax = bmax[id1,0]
                        vmax = np.max(bmax[:k+1,2])
                        id2 = np.argmax(bmax[:k+1,2])
                        vvmax = bmax[id2,0]
                        amax = np.max(bmax[:k+1,3])
                        id3 = np.argmax(bmax[:k+1,3])
                        vamax = bmax[id3,0]
                        if sta:
                            cmovprintfbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k+1,:],ressta[i].u_max,vdmax,vvmax,vamax,dmax,vmax,amax,project_name)
                        else:
                            cmovprintfbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k+1,:],0.0,vdmax,vvmax,vamax,dmax,vmax,amax,project_name)
                        if gr:
                            cmovplotbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k+1,:], vdmax, vamax, dmax, amax,project_name)

                else: 
                    dmax = np.max(bmax[:k+1,1])
                    id1 = np.argmax(bmax[:k+1,1])
                    vdmax = bmax[id1,0]
                    vmax = np.max(bmax[:k+1,2])
                    id2 = np.argmax(bmax[:k+1,2])
                    vvmax = bmax[id2,0]
                    amax = np.max(bmax[:k+1,3])
                    id3 = np.argmax(bmax[:k+1,3])
                    vamax = bmax[id3,0]
                    if sta:
                        cmovprintfbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k,:],ressta[i].u_max,vdmax,vvmax,vamax,dmax,vmax,amax,project_name)
                    else:
                        cmovprintfbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k,:],0.0,vdmax,vvmax,vamax,dmax,vmax,amax,project_name)
                    if gr:
                        cmovplotbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k,:], vdmax, vamax, dmax, amax,project_name)
    #------------------------------------
    return
#--------- functions for interaction option -----
def inteiso(trenid,estructuras,dt, velo, tra, sta, vb, cp , gr, nc, e, project_name, add_trainInte, progressbar):
    r"""
    % Function that calculates the dynamic responseof bridge due to moving load
    % for general case of bridge including the skew bridges (with introducing 
            % the skew angle)
    % User form:  cmovisoe(trenid, estructura, dt, velocidad, tra, sta, vb, cp, gr, nc, e)
    % Input data:
    %   trenid: identificative number of train 
    %   estructura: Structure that contains all information about the bridges
    %       estructura.EI:  bending stiffness
    %       estructura.GJ:  torsion stiffness
    %       estructura.alpha: skew angle
    %       estructura.m: mass per unit length
    %       estructura.r: radius of gyration
    %       estructura.L: Length of bridge
    %       estructura.nmod: number of modes of vibration considered in
    %       calculation
    %       estructura.nf: indicator of frequency level: 'h'-high, 'm'-media,
    %       'l' - low
    %       estructura.nm: indicator of mass level: 'h'-high, 'm'-media,
    %       'l'-low
    %       estructura.nv: number of spans of the bridge
    %       estructura.xi: structural damping
    %       estructura.simply=1 use simplification model for skew bridge, =0 use 
    %       full analytical model bridge
    %   dt: time increment
    %   velo: structure that contains information about the range of
    %   velocity that trains runs on the bridges.
    %       if there is not range of vehicle velocity:
    %           velocity contains only the value of a vehicle velocity in km/h
    %       if there is range of vehicle velocity:
    %           velocidad.inicial: initial velocity
    %           velocidad.final: final velocity
    %           velocidad.inc:  increment of speed km/h
    %   tra: distance between sleepers in m
    %   sta: if sta =1, the static response will be calculated, elseif sta = 0
    %   is not.
    %   vb: verbose mode. if vb =1 the results is saved for each velocity,
    %   elseif vb = 0 is not
    %   cp: if cp=1 compress the file of results, else is not
    %   gr: graphic mode, if gr =1 is yes, else is not
    %   nc: number of core to be used in parallel computing
    %   estructura.x: distance of the point that we want to obtain the dynamic response
    %   e: excentricity of load respect to the center line
    % Output data: files of results of displacement, velocity, acceleration 
    % Created by Jose M. Goicolea abr 2008, updated by P. Antolin
    % Ultimate update and modified by Khanh Nguyen Gia Jul 2016
    % =========================================================================
    """
    # Check the velo is a class or only one value
    if not isinstance(velo,cell_struct):
        # print('is calculating in this point')
        if isinstance(estructuras,list) and len(estructuras) >1:
            #print('Se definio mas de una estructura para un caso sencillo')
            return
        elif isinstance(estructuras,cell_struct):
            estructura = estructuras
        if isinstance(trenid,list) and len(trenid) > 1:
            #print('More than one train is defined')
            return
        elif isinstance(trenid,int):
            #print('is calculating in this point')
            tren = train(); tren.intetrain(trenid,tra,add_trainInte); tren.excentricidad = e
        else:
            #print('is calculating in this point')
            tren = train(); tren.intetrain(trenid[0],tra, add_trainInte); tren.excentricidad = e
        estructura.dt = dt
        vel = velo/3.6
        # Determine the modal parameters and the responses for the bridge
        mm = multiprocessing.Manager(); qq = mm.Queue()
        if estructura.nv == 1:
            if estructura.simply ==1:
                SpringModal(estructura)
                ParaIntegraExact(estructura)
                res = SpringInteIntegra(estructura,vel,estructura.x, tren)
            elif estructura.simply == 0:
                SkewModal(estructura)
                ParaIntegraExact(estructura)
                res = SkewInteIntegra((estructura,vel,estructura.x,tren,qq))
        elif estructura.nv > 1:
            SkewHiperModal(estructura)
            ParaIntegraExact(estructura)
            res = SkewHiperInteIntegra((estructura,vel,estructura.x,tren,qq))
        # Determine the maximum responses
        # Displacement
        dmax = np.max(abs(res[:,1]))
        id1 = np.argmax(abs(res[:,1])); tdmax = res[id1,0]
        # Velocity
        vmax = np.max(abs(res[:,2]))
        id2 = np.argmax(abs(res[:,2])); tvmax = res[id2,0]
        # Acceleration
        amax = np.max(abs(res[:,3]))
        id3 = np.argmax(abs(res[:,3])); tamax = res[id3,0]
        # save the results
        inteprintfhistoria(estructura,tren,vel,tra,res,tdmax,tvmax,tamax, dmax,vmax,amax, project_name)
        # plot the results
        if gr:
            inteplothistoria(estructura,tren,vel,res,tdmax,tamax,dmax,amax)
        #
    else:
        v0 = np.arange(velo.ini,velo.final+velo.inc,velo.inc)
        v0 = v0/3.6
        if  isinstance(trenid,int):
            num_trains=1
            trenes = train(); trenes.intetrain(trenid,tra, add_trainInte); trenes.excentricidad = e
        elif isinstance(trenid,list):
            num_trains = len(trenid)
            trenes = [train() for i in range(num_trains)]
            for i in range(num_trains):
                trenes[i].intetrain(trenid[i],tra, add_trainInte)
                trenes[i].excentricidad = e
        # Determine cell of structures
        if isinstance(estructuras,list):
            num_struct = len(estructuras)
            for i in range(len(estructuras)):
                if  estructuras[i].nv == 1:
                    if estructuras[i].simply == 1:
                        SpringModal(estructuras[i]); estructuras[i].dt = dt
                        ParaIntegraExact(estructuras[i])
                    else:
                        SkewModal(estructuras[i]); estructuras[i].dt = dt
                        ParaIntegraExact(estructuras[i])
                else:
                    SkewHiperModal(estructuras[i]); estructuras[i].dt = dt
                    ParaIntegraExact(estructuras[i])
        elif isinstance(estructuras,cell_struct):
            num_struct = 1
            if  estructuras.nv == 1:
                if estructuras.simply == 1:
                    SpringModal(estructuras); estructuras.dt = dt
                    ParaIntegraExact(estructuras)
                else:
                    SkewModal(estructuras); estructuras.dt = dt
                    ParaIntegraExact(estructuras)
            else:
                SkewHiperModal(estructuras); estructuras.dt = dt
                ParaIntegraExact(estructuras)
        #---
        # May be that all trains do not run woth the same velocity. So we need 
        # determine the total number cases that will be calculated
        ncases = 0
        for i in range(num_trains):
            for j in range(num_struct):
                for k in range(len(v0)):
                    if isinstance(trenid,list):
                        if  v0[k] > trenes[i].vmax/3.6:
                            break
                    elif isinstance(trenid,int):
                        if  v0[k] > trenes.vmax/3.6:
                            break
                    ncases += 1
        # print(ncases)
        #---
        # Create an argument that contains all informations of all cases to be calculated
        argumentos = [cell_struct() for i in range(ncases)]
        n = 0
        for i in range(num_trains):
            for j in range(num_struct):
                for k in range(len(v0)):
                    if isinstance(trenid,list):
                        if  v0[k] > trenes[i].vmax/3.6:
                            break
                    elif isinstance(trenid,int):
                        if  v0[k] > trenes.vmax/3.6:
                            break
                    n += 1
                    if isinstance(trenid,list):
                        argumentos[n-1].tren = trenes[i]
                    elif  isinstance(trenid,int):
                        argumentos[n-1].tren = trenes
                    if isinstance(estructuras,list):
                        argumentos[n-1].estructura = estructuras[j]
                        argumentos[n-1].x = estructuras[j].x
                    else:
                        argumentos[n-1].estructura = estructuras
                        argumentos[n-1].x = estructuras.x
                    argumentos[n-1].vel = v0[k]
        #  Determine the number of core used in the parallel calculation
        if  nc > multiprocessing.cpu_count():
            num_cores = multiprocessing.cpu_count()
        else:
            num_cores = nc
        # print(num_cores)
        if  num_cores == 1:
            # Create a cell 1xncases that contains all responses of all cases
            res =[cell_struct() for i in range(ncases)]
            progressbar.setMinimum(1)
            progressbar.setMaximum(ncases)
            mm = multiprocessing.Manager(); qq = mm.Queue()
            for i in range(ncases):
                if argumentos[i].estructura.nv == 1:
                    res[i] = SkewInteIntegra((argumentos[i].estructura,argumentos[i].vel,argumentos[i].x,argumentos[i].tren,qq))
                    progressbar.setValue(float(i+1))
                    QApplication.processEvents()
                else:
                    res[i] = SkewHiperInteIntegra((argumentos[i].estructura,argumentos[i].vel,argumentos[i].x,argumentos[i].tren,qq))
                    progressbar.setValue(float(i+1))
                    QApplication.processEvents()
        else:
            progressbar.setMinimum(1)
            progressbar.setMaximum(ncases)
            mm = multiprocessing.Manager(); qq = mm.Queue()
            if estructuras.nv == 1:
                mypool = multiprocessing.Pool(num_cores)
                kk = [(argumentos[i].estructura,argumentos[i].vel,argumentos[i].x,argumentos[i].tren, qq)  for i in range(ncases)]
                output = mypool.map_async(SkewInteIntegra, kk)
                while not output.ready():
                    progressbar.setValue(qq.qsize())
                    QApplication.processEvents()
                res = output.get()
                mypool.close()
                #res = Parallel(n_jobs=num_cores)(delayed(SkewInteIntegra) (argumentos[i].estructura,argumentos[i].vel,argumentos[i].x,argumentos[i].tren)  for i in range(ncases))
                #progressbar.setValue(ncases)
                #QApplication.processEvents()
            else:
                mypool = multiprocessing.Pool(num_cores)
                kk = [(argumentos[i].estructura,argumentos[i].vel,argumentos[i].x,argumentos[i].tren, qq)  for i in range(ncases)]
                output = mypool.map_async(SkewHiperInteIntegra, kk)
                while not output.ready():
                    progressbar.setValue(qq.qsize())
                    QApplication.processEvents()
                res = output.get()
                mypool.close()
                #res = Parallel(n_jobs=num_cores)(delayed(SkewHiperInteIntegra) (argumentos[i].estructura,argumentos[i].vel,argumentos[i].x,argumentos[i].tren) for i in range(ncases))
                #progressbar.setValue(ncases)
                #QApplication.processEvents()
        #------
        # Determine the static responses if is requested
        if sta:
            if  num_cores == 1:
                ressta = [cell_struct() for i in range(num_trains)]
                if estructuras.nv == 1:
                    if estructuras.simply == 1:
                        for i in range(num_trains):
                            SpringStatic(estructuras,trenes[i])
                            ressta[i].u_max = estructuras.u_max
                            ressta[i].u_static = estructuras.u_static
                            ressta[i].phi1 = estructuras.phi1
                            ressta[i].phi2 = estructuras.phi2
                            ressta[i].phi1_max = estructuras.phi1_max
                            ressta[i].phi2_max = estructuras.phi2_max
                            inteprintfstatic(estructuras,trenes[i],tra,ressta[i], project_name)
                    else:
                        for i in range(num_trains):
                            SkewStatic(estructuras,trenes[i])
                            ressta[i].u_max = estructuras.u_max
                            ressta[i].u_static = estructuras.u_static
                            ressta[i].phi1 = estructuras.phi1
                            ressta[i].phi2 = estructuras.phi2
                            ressta[i].phi1_max = estructuras.phi1_max
                            ressta[i].phi2_max = estructuras.phi2_max
                            inteprintfstatic(estructuras,trenes[i],tra,ressta[i], project_name)
                elif estructuras.nv > 1:
                    for i in range(num_trains):
                        ressta[i] = SkewHiperStatic(estructuras,trenes[i])
                        ressta[i].u_max = estructuras.u_max
                        ressta[i].u_static = estructuras.u_static
                        inteprintfstatic(estructuras,trenes[i],tra,ressta[i], project_name)
            else:
                if estructuras.nv == 1:
                    if estructuras.simply == 1:
                        mypool = multiprocessing.Pool(num_cores)
                        output = [mypool.apply_async(SringStatic, args=(estructuras,trenes[i])) for i in range(num_trains)]
                        ressta = [p.get() for p in output]
                        mypool.close()
                        #ressta = Parallel(n_jobs=num_cores)(delayed(SpringStatic) (estructuras,trenes[i]) for i in range(num_trains))
                        for i in range(num_trains):
                            inteprintfstatic(estructuras,trenes[i],tra,ressta[i], project_name)
                    else:
                        mypool = multiprocessing.Pool(num_cores)
                        output = [mypool.apply_async(SkewStatic, args=(estructuras,trenes[i])) for i in range(num_trains)]
                        ressta = [p.get() for p in output]
                        mypool.close()
                        #ressta = Parallel(n_jobs=num_cores)(delayed(SkewStatic) (estructuras,trenes[i]) for i in range(num_trains))
                        for i in range(num_trains):
                            inteprintfstatic(estructuras,trenes[i],tra,ressta[i], project_name)
                else:
                    mypool = multiprocessing.Pool(num_cores)
                    output = [mypool.apply_async(SkewHiperStatic, args=(estructuras,trenes[i])) for i in range(num_trains)]
                    ressta = [p.get() for p in output]
                    mypool.close()
                    #ressta = Parallel(n_jobs=num_cores)(delayed(SkewHiperStatic) (estructuras,trenes[i]) for i in range(num_trains))
                    for i in range(num_trains): 
                        inteprintfstatic(estructuras,trenes[i],tra,ressta[i], project_name)
        #------------
        # postprocess
        #------------
        n = 0
        bmax = np.zeros([len(v0),4])
        for i in range(num_trains):
            for j in range(num_struct):
                for k in range(len(v0)):
                    if isinstance(trenid,list):
                        if  v0[k] > trenes[i].vmax/3.6:
                            break
                    elif isinstance(trenid,int):
                        if  v0[k] > trenes.vmax/3.6:
                            break
                    n +=1
                    # maximum responses
                    # displacement
                    dmax = np.max(abs(res[n-1][:,1]))
                    id1 = np.argmax(abs(res[n-1][:,1]))
                    tdmax = res[n-1][id1,0]
                    # velocity
                    vmax = np.max(abs(res[n-1][:,2]))
                    id2 = np.argmax(abs(res[n-1][:,2]))
                    tvmax = res[n-1][id2,0]
                    # acceleration
                    amax = np.max(abs(res[n-1][:,3]))
                    id3 = np.argmax(abs(res[n-1][:,3]))
                    tamax = res[n-1][id3,0]
                    bmax[k,0] = v0[k]*3.6
                    bmax[k,1] = dmax 
                    bmax[k,2] = vmax
                    bmax[k,3] = amax
                    if  vb:
                        inteprintfhistoria(argumentos[n-1].estructura,argumentos[n-1].tren,v0[k]*3.6,tra,res[n-1],tdmax,tvmax,tamax,dmax,vmax,amax, project_name)
                # 
                if k!= 0:
                    if k < len(v0)-1:
                        dmax = np.max(bmax[:k,1])
                        id1 = np.argmax(bmax[:k,1])
                        vdmax = bmax[id1,0]
                        vmax = np.max(bmax[:k,2])
                        id2 = np.argmax(bmax[:k,2])
                        vvmax = bmax[id2,0]
                        amax = np.max(bmax[:k,3])
                        id3 = np.argmax(bmax[:k,3])
                        vamax = bmax[id3,0]
                        if sta:
                            inteprintfbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k,:],ressta[i].u_max,vdmax,vvmax,vamax,dmax,vmax,amax,project_name)
                        else:
                            inteprintfbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k,:],0.0,vdmax,vvmax,vamax,dmax,vmax,amax,project_name)
                        if gr:
                            inteplotbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k,:], vdmax, vamax, dmax, amax,project_name)
                    else:
                        dmax = np.max(bmax[:k+1,1])
                        id1 = np.argmax(bmax[:k+1,1])
                        vdmax = bmax[id1,0]
                        vmax = np.max(bmax[:k+1,2])
                        id2 = np.argmax(bmax[:k+1,2])
                        vvmax = bmax[id2,0]
                        amax = np.max(bmax[:k+1,3])
                        id3 = np.argmax(bmax[:k+1,3])
                        vamax = bmax[id3,0]
                        if sta:
                            inteprintfbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k+1,:],ressta[i].u_max,vdmax,vvmax,vamax,dmax,vmax,amax,project_name)
                        else:
                            inteprintfbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k+1,:],0.0,vdmax,vvmax,vamax,dmax,vmax,amax,project_name)
                        if gr:
                            inteplotbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k+1,:], vdmax, vamax, dmax, amax,project_name)

                else: 
                    dmax = np.max(bmax[:k+1,1])
                    id1 = np.argmax(bmax[:k+1,1])
                    vdmax = bmax[id1,0]
                    vmax = np.max(bmax[:k+1,2])
                    id2 = np.argmax(bmax[:k+1,2])
                    vvmax = bmax[id2,0]
                    amax = np.max(bmax[:k+1,3])
                    id3 = np.argmax(bmax[:k+1,3])
                    vamax = bmax[id3,0]
                    if sta:
                        inteprintfbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k,:],ressta[i].u_max,vdmax,vvmax,vamax,dmax,vmax,amax,project_name)
                    else:
                        inteprintfbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k,:],0.0,vdmax,vvmax,vamax,dmax,vmax,amax,project_name)
                    if gr:
                        inteplotbarrido(argumentos[n-1].estructura,argumentos[n-1].tren,velo,tra,bmax[:k,:], vdmax, vamax, dmax, amax,project_name)
    #------------------------------------
    return
# some additional functions for saving and plotting data
#-----------------
def inteprintfhistoria(estructura,tren,vel,tra,res,tdmax,tvmax,tamax, dmax,vmax,amax,project_name):
    path = getcwd()
    if  tra != 0:
        name = '-imd-'
    else:
        name = '-im-'
    datfile = path + '/' + project_name + '/' + tren.nombre+ name + 'skew'+str(estructura.alpha)+'-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-m'+str(estructura.m/1.e03)+'t-nm'+str(estructura.nmod) +'-v'+str(vel)+'.dat'
    fid = open(datfile,'w')
    fid.write("# Case %s\n" % (datfile))
    fid.write("# Train = %s\n" % (tren.nombre))
    fid.write("# v = %.1f km/h\n" % (vel))
    fid.write("# L = %s m\n" % (str(estructura.L)))
    fid.write("# EI = %s N.m2\n" % (str(estructura.EI)))
    fid.write("# GJ = %s N.m2\n" % (str(estructura.GJ)))
    fid.write("# rho = %s t/m\n" % (str(estructura.m/1.e03)))
    fid.write("# r = %s m\n" % (str(estructura.r)))
    fid.write("# skew angle = %.2f\n" % (estructura.alpha))
    fid.write("# zeta = %.2f \n" % (estructura.xi))
    fid.write("# f0 = %.3f Hz\n" % (estructura.wn[0]/2/np.pi))
    fid.write("# Number of modes = %i\n" % (estructura.nmod))
    fid.write("# Time increment = %.3f (s)\n" % (estructura.dt))
    fid.write("# \n")
    fid.write("# dmax = %12.5g at %12.5g s\n" % (dmax,tdmax))
    fid.write("# vmax = %12.5g at %12.5g s\n" % (vmax,tvmax))
    fid.write("# amax = %12.5g at %12.5g s\n" % (amax,tamax))
    fid.write("# \n")
    fid.write("#    time      d      v      a      b      db      ab     \n")
    for i in range(len(res)):
        fid.write("%5.4g  %5.4g   %5.4g   %5.4g   %5.4g   %5.4g    %5.4g\n" % (res[i,0],res[i,1],res[i,2],res[i,3],res[i,7],res[i,8],res[i,9]))
    fid.close()
    return
#----------
def inteprintfstatic(estructura,tren,tra,ressta, project_name):
    path = getcwd()
    if  tra != 0:
        name = '-imd-'
    else:
        name = '-im-'
    datfile = path + '/' + project_name + '/' + tren.nombre+ name + 'skew'+str(estructura.alpha)+'-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-m'+str(estructura.m/1.e03)+'t-nm'+str(estructura.nmod) +'-static.dat'
    fid = open(datfile,'w')
    fid.write("# Case %s\n" % (datfile))
    fid.write("# Train = %s\n" % (tren.nombre))
    fid.write("# L = %s m\n" % (str(estructura.L)))
    fid.write("# EI = %s N.m2\n" % (str(estructura.EI)))
    fid.write("# GJ = %s N.m2\n" % (str(estructura.GJ)))
    fid.write("# rho = %s t/m\n" % (str(estructura.m/1.e03)))
    fid.write("# r = %s m\n" % (str(estructura.r)))
    fid.write("# skew angle = %.2f\n" % (estructura.alpha))
    fid.write("# zeta = %.2f \n" % (estructura.xi))
    fid.write("# f0 = %.3f Hz\n" % (estructura.wn[0]/2/np.pi))
    for i in range(len(ressta.u_static)):
        fid.write('%i\t%5.4g\n' % (i, ressta.u_static[i]))
    fid.close()
    return
#----------
def inteprintfbarrido(estructura,tren,velo,tra,res,sta,vdmax,vvmax,vamax,dmax,vmax,amax,project_name):
    path = getcwd()
    if  tra != 0:
        name = '-imd-'
    else:
        name = '-im-'
    if  velo.final <= tren.vmax:
        datfile = path + '/' + project_name + '/' + tren.nombre+'-envelope'+name+'skew'+str(estructura.alpha)+'-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-m'+str(estructura.m/1.e03)+'t-nm'+str(estructura.nmod) +'-v'+str(velo.ini)+'-'+str(velo.final) + '.dat'
    else:
        datfile = path + '/' + project_name + '/' + tren.nombre+name+'-envelope'+name+'skew'+str(estructura.alpha)+'-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-m'+str(estructura.m/1.e03)+'t-nm'+str(estructura.nmod) +'-v'+str(velo.ini)+'-'+str(tren.vmax) + '.dat'
    #
    fid = open(datfile,'w')
    fid.write("# Envelope: %s\n" %(datfile))
    fid.write("# Train = %s\n" % (tren.nombre))
    fid.write("# v initial = %.2f km/h \n" % (velo.ini))
    if  velo.final <= tren.vmax:
        fid.write("# v final = %.2f km/h \n" % (velo.final))
    else:
        fid.write("# v final = %.2f km/h \n" % (tren.vmax))
    fid.write("# speed increment = %.2f km/h \n" %(velo.inc))
    fid.write("# rho = %s t/m\n" % (str(estructura.m/1.e03)))
    fid.write("# r = %s m\n" % (str(estructura.r)))
    fid.write("# skew angle = %.2f\n" % (estructura.alpha))
    fid.write("# zeta = %.2f \n" % (estructura.xi))
    fid.write("# f0 = %.3f Hz\n" % (estructura.wn[0]/2/np.pi))
    fid.write("# Number of modes = %i\n" % (estructura.nmod))
    fid.write("# Time increment = %.3f (s)\n" % (estructura.dt))
    fid.write("# \n")
    fid.write("# dmax = %12.5g at %12.5g km/h\n" % (dmax,vdmax))
    fid.write("# vmax = %12.5g at %12.5g km/h\n" % (vmax,vvmax))
    fid.write("# amax = %12.5g at %12.5g km/h\n" % (amax,vamax))
    if sta != 0.:
        fid.write("# dmax static = %12.5g m\n" % (sta))
    fid.write("# \n")
    fid.write("# vel    dmax    vmax    amax    DAF\n")
    for i in range(len(res)):
       fid.write("%8.4g    %8.4g   %8.4g   %8.4g    %8.4g\n" % (res[i,0],res[i,1],res[i,2],res[i,3],res[i,1]/sta))
    fid.close()
    return
#----------
def inteprintfstatic(estructura,tren,tra,ressta, project_name):
    path = getcwd()
    if  tra != 0:
        name = '-imd-'
    else:
        name = '-im-'
    datfile = path + '/' + project_name + '/' + tren.nombre+ name + 'skew'+str(estructura.alpha)+'-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-m'+str(estructura.m/1.e03)+'t-nm'+str(estructura.nmod) +'-static.dat'
    fid = open(datfile,'w')
    fid.write("# Case %s\n" % (datfile))
    fid.write("# Train = %s\n" % (tren.nombre))
    fid.write("# L = %s m\n" % (str(estructura.L)))
    fid.write("# EI = %s N.m2\n" % (str(estructura.EI)))
    fid.write("# GJ = %s N.m2\n" % (str(estructura.GJ)))
    fid.write("# rho = %s t/m\n" % (str(estructura.m/1.e03)))
    fid.write("# r = %s m\n" % (str(estructura.r)))
    fid.write("# skew angle = %.2f\n" % (estructura.alpha))
    fid.write("# zeta = %.2f \n" % (estructura.xi))
    fid.write("# f0 = %.3f Hz\n" % (estructura.wn[0]/2/np.pi))
    for i in range(len(ressta.u_static)):
        fid.write('%i\t%5.4g\n' % (i, ressta.u_static[i]))
    fid.close()
    return
#-----------
def inteplothistoria(estructura,tren,vel,tra,res,tdmax,tamax,dmax,amax,project_name):
    #
    path = getcwd()
    if  tra != 0:
        name = '-imd-'
    else:
        name = '-im-'
    basename = path + '/' + project_name + '/' +tren.nombre+name+'-skew'+str(estructura.alpha)+'-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-m'+str(estructura.m/1.e03)+'t-nm'+str(estructura.nmod) +'-v'+str(vel)
    titlename = tren.nombre +name +'-skew'+str(estructura.alpha) + '-L' + str(estructura.L)+ '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + '-v' + str(vel)
    epsfile = basename + '.eps'
    plt.figure()
    plt.subplot(211)
    plt.plot(res[:,0],res[:,3])
    if amax == max(res[:,3]):
        plt.plot(tamax,amax,'or',ms = 5.)
    else:
        plt.plot(tamax,-amax,'or',ms = 5.)
    plt.title(titlename)
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'Acceleration (m/s$^2$)')
    leyenda = r'amax = %.3f m/s$^2$ at t = %.3f s' % (amax,tamax)
    tmedio = 0.5*(res[0,0]+res[-1,0])
    if tamax > tmedio:
        if amax == max(res[:,3]):
            plt.text(tamax, amax,leyenda, horizontalalignment='right')
        else:
            plt.text(tamax,-amax,leyenda, horizontalalignment='right')
    else:
        if amax == max(res[:,3]):
            plt.text(tamax, amax,leyenda, horizontalalignment='left')
        else:
            plt.text(tamax,-amax,leyenda, horizontalalignment='left')
    #---            
    plt.subplot(212)
    plt.plot(res[:,0],1000*res[:,1])
    if  dmax == max(res[:,1]):
        plt.plot(tdmax,1000*dmax,'or',ms=5.)
    else:
        plt.plot(tdmax,-1000*dmax,'or',ms=5.)
    # plt.title(titlename + ': desplazamiento en el centro del vano')
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'Displacement (mm)')
    leyenda = r'dmax = %.3f mm at t = %.3f s' % (1000*dmax,tdmax)
    if tdmax > tmedio:
        if dmax == max(res[:,1]):
            plt.text(tdmax, dmax*1000,leyenda, horizontalalignment='right')
        else:
            plt.text(tdmax,-dmax*1000,leyenda, horizontalalignment='right')
    else:
        if dmax == max(res[:,1]):
            plt.text(tdmax, dmax*1000,leyenda, horizontalalignment='left')
        else:
            plt.text(tdmax,-dmax*1000,leyenda, horizontalalignment='left')
    # 
    plt.savefig(epsfile)  
    return
#------------
def inteplotbarrido(estructura,tren,velo,tra,res, vdmax, vamax, dmax, amax,project_name):
    path = getcwd()
    if  tra != 0:
        name = '-imd-'
    else:
        name = '-im-'
    if velo.final <= tren.vmax:
        basename = path + '/' + project_name + '/' + tren.nombre+'-envelope'+ name + 'skew'+str(estructura.alpha)+'-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-m'+str(estructura.m/1.e03)+'t-nm'+str(estructura.nmod) +'-v'+str(velo.ini)+'-'+str(velo.final) 
        titlename = tren.nombre+'-envelope'+name+'skew' + str(estructura.alpha) + '-L' + str(estructura.L)+ '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi))  + '-v' + str(velo.ini) + '-' + str(velo.final)
    else:
        basename = path + '/' + project_name + '/' + tren.nombre+'-envelope'+ name + 'skew'+str(estructura.alpha)+'-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + 'Hz-m'+str(estructura.m/1.e03)+'t-nm'+str(estructura.nmod) +'-v'+str(velo.ini)+'-'+str(tren.vmax) 
        titlename = tren.nombre+'-envelope'+name+'skew' + str(estructura.alpha) + '-L' + str(estructura.L)+ '-f'+ str('%.2f' % (estructura.wn[0]/2/np.pi)) + '-v' + str(velo.ini) + '-' + str(tren.vmax)
    #
    epsfile = basename + '.eps'
    plt.figure()
    plt.subplot(211)
    plt.plot(res[:,0],res[:,3])
    plt.plot(vamax,amax,'or',ms=10.)
    plt.xlabel(r'Velocity (km/h)')
    plt.ylabel(r'Acceleration (m/s$^2$)')
    plt.title(titlename)
    leyenda = r'amax = %.3f m/s$^2$ at v = %.1f km/h' %(amax,vamax)
    vmedio = 0.5*(res[0,0]+res[-1,0])
    if  vamax > vmedio:
        plt.text(vamax,amax,leyenda,horizontalalignment='right')
    else:
        plt.text(vamax,amax,leyenda,horizontalalignment='left')
    plt.subplot(212)
    plt.plot(res[:,0],1.e03*res[:,1])
    plt.plot(vdmax,dmax*1e03,'or',ms=10.)
    plt.xlabel(r'Velocity (km/h)')
    plt.ylabel(r'Displacement (mm)')
    # plt.title(titlename+': maximo desplazamiento')
    leyenda = r'dmax = %.3f mm at v = %.1f km/h' %(dmax*1.e3,vdmax)
    if vdmax > vmedio:
        plt.text(vdmax,dmax*1000,leyenda,horizontalalignment='right')
    else:
        plt.text(vdmax,dmax*1000,leyenda,horizontalalignment='left')
    plt.savefig(epsfile)
    return

#--------
#--- LIR Method ----
def LIR(estructura, tren, velo, sta, project_name):
    """
    Function that determines the maximum dynamic responses of the bridge
    based on the LIR method
    #----------------------
    Input:
        estructura
        tren
        velo
    Output:
        res: ndarray 

    """
    # vector of velocity
    v = np.arange(velo.ini,velo.final+velo.inc,velo.inc)
    # fundamental frequency of bridge
    w0 = np.pi**2*np.sqrt(estructura.EI/estructura.m)/estructura.L**2 
    f0 = np.pi*np.sqrt(estructura.EI/estructura.m)/2./estructura.L**2 
    # wavelength of exitation
    l = v/(3.6*f0)
    # adimensional parameter
    K = l/2./estructura.L
    zeta = estructura.xi
    A = K*np.sqrt(np.exp(-2*zeta*np.pi/K)+1+2*np.cos(np.pi/K)*np.exp(-zeta*np.pi/K))/(1-K**2)
    # Coefficient for acceleration
    Ca = 2./(estructura.m*estructura.L)
    # 
    Cd = 2./(estructura.m*estructura.L*w0**2)
    # Train spectrum
    G = np.zeros(len(l))
    for j in range(len(l)):
        Gaux = []
        for k in range(len(tren.dist)):
            aux1 = 0;  aux2 = 0;
            for i  in range(k+1):
                delta = (tren.dist[k] - tren.dist[i])/l[j]
                aux1 += tren.peso[i]*np.cos(2*np.pi*delta)*np.exp(-2*np.pi*zeta*delta)
                aux2 += tren.peso[i]*np.sin(2*np.pi*delta)*np.exp(-2*np.pi*zeta*delta)
            Gaux.append(np.sqrt(aux1**2+aux2**2))
        G[j] = np.max(Gaux)
    amax = Ca*A*G
    dmax = Cd*A*G
    if sta:
        ressta = SkewStatic(estructura,tren)
    # save file
    path = getcwd() 
    if  velo.final <= tren.vmax:
        datfile = path + '/' + project_name + '/' + tren.nombre+'-LIR-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (f0)) + 'Hz-m'+str(estructura.m/1.e03)+'t-v'+str(velo.ini)+'-'+str(velo.final) + '.dat'
    else:
        datfile = path + '/' + project_name + '/' + tren.nombre+'-LIR-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (f0)) + 'Hz-m'+str(estructura.m/1.e03)+'t-v'+str(velo.ini)+'-'+str(tren.vmax) + '.dat'
    fid = open(datfile,'w')
    fid.write("# Case %s\n" % (datfile))
    fid.write("# Train = %s\n" % (tren.nombre))
    fid.write("# L = %s m\n" % (str(estructura.L)))
    fid.write("# EI = %s N.m2\n" % (str(estructura.EI)))
    fid.write("# GJ = %s N.m2\n" % (str(estructura.GJ)))
    fid.write("# rho = %s t/m\n" % (str(estructura.m/1.e03)))
    fid.write("# zeta = %.2f \n" % (estructura.xi))
    fid.write("# f0 = %.3f Hz\n" % (f0))
    fid.write("# \n")
    fid.write("# \n")
    fid.write("#    vel   dmax   amax   DAF\n")
    for i in range(len(v)):
        if v[i] > tren.vmax:
            break
        fid.write("%8.4g   %8.4g   %8.4g %8.4g \n" % (v[i], dmax[i],amax[i], dmax[i]/ressta.u_max))
    return  
#---
def clir(estructura, trenid, velo, tra, sta, project_name, add_train, progressbar):
    num_trains = len(trenid)
    trenes = [train() for i in range(num_trains)]
    progressbar.setMinimum(0)
    progressbar.setMaximum(num_trains)
    for i in range(num_trains):
        trenes[i].cmovtrain(trenid[i], tra, add_train)
        LIR(estructura, trenes[i], velo, sta, project_name)
        progressbar.setValue(float(i+1))
        QApplication.processEvents()
    return

#--- DER Method ----
def DER(estructura, tren, velo, sta, project_name):
    """
    Function that determines the maximum dynamic responses of the bridge
    based on the DER method
    #----------------------
    Input:
        estructura
        tren
        velo
    Output:
        res: ndarray 

    """
    # vector of velocity
    v = np.arange(velo.ini,velo.final+velo.inc,velo.inc)
    # fundamental frequency of bridge
    w0 = np.pi**2*np.sqrt(estructura.EI/estructura.m)/estructura.L**2 
    f0 = np.pi*np.sqrt(estructura.EI/estructura.m)/2./estructura.L**2 
    # wavelength of exitation
    l = v/(3.6*f0)
    # adimensional parameter
    zeta = estructura.xi
    A = np.abs(np.cos(np.pi*estructura.L/l)/((2*estructura.L/l)**2-1))
    # Coefficient for acceleration
    Ca = 4./(estructura.m*np.pi)
    # 
    # Train spectrum
    G = np.zeros(len(l))
    if zeta == 0:
        for j in range(len(l)):
            u = (np.cumsum(tren.peso*np.cos(2*np.pi*tren.dist/l[j])))**2
            s = (np.cumsum(tren.peso*np.sin(2*np.pi*tren.dist/l[j])))**2
            w = np.max((u+s)**(0.5))
            G[j] = w*(2*np.pi/l[j])
    else:
        for j in range(len(l)):
            Gaux = []
            for k in range(len(tren.dist)):
                if tren.dist[k] == 0:
                    Gaux.append(2*np.pi/l[j]*np.sqrt(np.sum(tren.peso[:k+1]*np.cos(2*np.pi*tren.dist[:k+1]/l[j]))**2 + np.sum(tren.peso[:k+1]*np.sin(2*np.pi*tren.dist[:k+1]/l[j]))**2))
                else:
                    Gaux.append((1-np.exp(-2*np.pi*zeta*tren.dist[k]/l[j]))*np.sqrt(np.sum(tren.peso[:k+1]*np.cos(2*np.pi*tren.dist[:k+1]/l[j]))**2 + np.sum(tren.peso[:k+1]*np.sin(2*np.pi*tren.dist[:k+1]/l[j]))**2)/(zeta*tren.dist[k]))
            G[j] = np.max(Gaux)
    # maximum acceleration
    amax = Ca*A*G
    # maximum displacement
    dmax = amax/w0**2
    if sta:
        ressta = SkewStatic(estructura,tren)
    # save file
    path = getcwd() 
    if  velo.final <= tren.vmax:
        datfile = path + '/' + project_name + '/' + tren.nombre+'-DER-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (f0)) + 'Hz-m'+str(estructura.m/1.e03)+'t-v'+str(velo.ini)+'-'+str(velo.final) + '.dat'
    else:
        datfile = path + '/' + project_name + '/' + tren.nombre+'-DER-L'+str(estructura.L) + '-z'+str(estructura.xi*100) + '-f'+ str('%.2f' % (f0)) + 'Hz-m'+str(estructura.m/1.e03)+'t-v'+str(velo.ini)+'-'+str(tren.vmax) + '.dat'
    fid = open(datfile,'w')
    fid.write("# Case %s\n" % (datfile))
    fid.write("# Train = %s\n" % (tren.nombre))
    fid.write("# L = %s m\n" % (str(estructura.L)))
    fid.write("# EI = %s N.m2\n" % (str(estructura.EI)))
    fid.write("# GJ = %s N.m2\n" % (str(estructura.GJ)))
    fid.write("# rho = %s t/m\n" % (str(estructura.m/1.e03)))
    fid.write("# zeta = %.2f \n" % (estructura.xi))
    fid.write("# f0 = %.3f Hz\n" % (f0))
    fid.write("# \n")
    fid.write("# \n")
    fid.write("#    vel   dmax   amax   DAF\n")
    for i in range(len(v)):
        if v[i] > tren.vmax:
            break
        fid.write("%8.4g   %8.4g   %8.4g %8.4g \n" % (v[i], dmax[i],amax[i], dmax[i]/ressta.u_max))
    return  
#---
def cder(estructura, trenid, velo, tra, sta, project_name, add_train, progressbar):
    num_trains = len(trenid)
    trenes = [train() for i in range(num_trains)]
    progressbar.setMinimum(0)
    progressbar.setMaximum(num_trains)
    for i in range(num_trains):
        trenes[i].cmovtrain(trenid[i], tra, add_train)
        DER(estructura, trenes[i], velo, sta, project_name)
        progressbar.setValue(float(i+1))
        QApplication.processEvents()
    return
        


