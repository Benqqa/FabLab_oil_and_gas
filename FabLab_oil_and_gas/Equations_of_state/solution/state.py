# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 23:18:48 2021

@author: Benqqa

1) пееписать все под единственый корень в пределах 0,000001 до 1,00000001
2) пределы изменение p от 3 до 7 
"""
import math
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Solution():
    def __init__(self,components_table,p,T):
        self.components_table=components_table
        self.R=8.314
        self.eps=10**(-5)
        self.p=p #давление
        self.T=T #температура
        self.TableOfPairCollisionRate=self.getTableOfPairCollisionRate()
        for component in self.components_table:
            component.f_K(self.T, self.p)
    def main(self):
        self.V=self.F_V()
        print(self.F_v())
        self.z_y=self.SupercompressibilityCoefficient(self.a_m(self.val),self.b_m(self.val),self.c_m(self.val),self.d_m(self.val),'max')
        self.z_x=self.SupercompressibilityCoefficient(self.x_a_m(self.val),self.x_b_m(self.val),self.x_c_m(self.val),self.x_d_m(self.val),'min')
        print("z_y",self.z_y)
        print("z_x",self.z_x)
        if(len(self.val) !=0):
            print("rooot",self.val)
            self.K_2(self.val)
                    
    def getTableOfPairCollisionRate(self):#коэф парного столкновения
        table = pd.read_excel('./коэф парного взаимодействия.xlsx',index_col='c(ij)')
        return table
       
    def F_V(self):
        
        V=sp.symbols('V',real=True)
        _sum=0
        for component in self.components_table:
            # el=((component.z*(component.K))/(V*(component.K)-V*(1)+1))-((component.z*(1))/(V*(component.K)-V*(1)+1))
            el=((component.z*(component.K-1))/(V*(component.K-1)+1))
            _sum+=el
        # print(_sum,"каща")
        res=list(sp.solveset(sp.Eq((_sum),0),V,domain = sp.S.Reals))
        root=[]
        print(res)
        r_gr = 1.2
        l_gr = -0.2
        if(len(res)>1):
            for item in res:
                if(item<r_gr and item>l_gr):
                    root.append(item)
            print(len(root)  )
        else:
            for item in res:
                root.append(item)
        if(len(root)>1):
            new_r=[]
            old_r=[]
            for item in root:
                old_r.append(item)
            while(True):
                new_r.clear()
                l_gr+=0.00001
                r_gr-=0.00001
                for item in old_r:
                    if(item<r_gr and item>l_gr):
                        new_r.append(item)
                if(len(new_r)>1):
                    old_r.clear()
                    for item in new_r:
                        old_r.append(item)
                elif(len(new_r)==0):
                    print(old_r,"qqqqqqqqq")
                    print(new_r)
                    l_gr-=0.00001 #откат
                    r_gr+=0.00001
                    # l_gr=0
                    # r_gr=1
                    ii=0
                    while(True):
                        ii+=1
                        print(ii)
                        new_r.clear()
                        
                        
                        for item in old_r:
                            if(item<=r_gr and item>=l_gr):
                                # print(item)
                                new_r.append(item)
                        if(len(new_r)>1):
                            old_r.clear()
                            for item in new_r:
                                old_r.append(item)
                        elif(len(new_r)==1):
                            root.clear()
                            for item in new_r:
                                root.append(item)
                            break
                        l_gr+=0.0000001
                        r_gr-=0.0000001
                        if(ii>100):
                            # old_r.clear()
                            # old_r=[min(old_r)]
                            root.clear()
                            root.append(min(old_r))
                            break
                    break
                        
                else:
                    root.clear()
                    for item in new_r:
                        root.append(item)
                    break              
        print("f_v",root)
        
        return root
    
    def F_v(self):
        flag=False
        val=[]
        for value in self.V:
            print("value== ",value)
            if(value<0):
                print("V ненасыщенном жидком состоянии")
                val.append(value)
            elif(value==0):
                print("V насыщенное жидкое состояние (точка кипения).")
                val.append(value)
            elif(value>0 and value<1):
                print("V двухфазное парожидкостное состояние.")
                flag=True
                val.append(value)
            elif(value==1):
                print("V однофазное насыщенное паровое (газовое) состояние (точка росы).")
                val.append(value)
            elif(value>1):
                print("V однофазное ненасыщенное газовое состояние.")
                val.append(value)
        _sum1=0
        for component in self.components_table:
            _sum1+=component.z*component.K
        _sum2=0
        for component in self.components_table:
            _sum2+=+component.z/component.K
        # print(_sum1)
        # print(_sum2)
        self.val=val
        if(_sum1<1):
            return "В ненасыщенном жидком состоянии"
        elif(_sum1==1):
            return "В насыщенном жидком состоянии (точка кипения)." 
        elif(_sum2>1 and _sum1>1 ):
            # if(flag):
                
                # self.K_2(val)
            return "В двухфазном состоянии." 
        elif(_sum2==1):
            return "В однофазное насыщенное паровое (газовое) состояние (точка росы)." 
        elif(_sum2<1):
            return "однофазное ненасыщенное газовом состояние."
    def C(self, row_name,column_name): # коэффициентов парного взаимодействия
        table = self.TableOfPairCollisionRate
        i=table.columns.values.tolist().index(row_name)
        return table[column_name].iloc[i]
    #какие то коэфициенты
    def a_m(self,val):
        res=[]
        for value in val:    
            _sum=0
            for component_i in self.components_table:
                for component_j in self.components_table:
                    _sum+=component_i.y(self.T, self.p, value)*component_j.y(self.T, self.p, value)*(1-self.C(component_i.name,component_j.name))*math.sqrt(component_i.a(self.T)*component_j.a(self.T))
            res.append(_sum)
        return res
    def b_m(self,val):
        res=[]
        for value in val:    
            _sum=0
            for component_i in self.components_table:
                    _sum+=component_i.y(self.T, self.p, value)*component_i.b()
            res.append(_sum)
        return res
    def c_m(self,val):
        res=[]
        for value in val:    
            _sum=0
            for component_i in self.components_table:
                    _sum+=component_i.y(self.T, self.p, value)*component_i.c()
            res.append(_sum)
        return res
    def d_m(self,val):
        res=[]
        for value in val:    
            _sum=0
            for component_i in self.components_table:
                    _sum+=component_i.y(self.T, self.p, value)*component_i.d()
            res.append(_sum)
        return res
    #
    #какие то коэфициенты2
    def x_a_m(self,val):
        res=[]
        for value in val:    
            _sum=0
            for component_i in self.components_table:
                for component_j in self.components_table:
                    _sum+=component_i.x(self.T, self.p, value)*component_j.x(self.T, self.p, value)*(1-self.C(component_i.name,component_j.name))*math.sqrt(component_i.a(self.T)*component_j.a(self.T))
            res.append(_sum)
        return res
    def x_b_m(self, val):
        res=[]
        for value in val:    
            _sum=0
            for component_i in self.components_table:
                    _sum+=component_i.x(self.T, self.p, value)*component_i.b()
            res.append(_sum)
        return res
    def x_c_m(self,val):
        res=[]
        for value in val:    
            _sum=0
            for component_i in self.components_table:
                    _sum+=component_i.x(self.T, self.p, value)*component_i.c()
            res.append(_sum)
        return res
    def x_d_m(self,val):
        res=[]
        for value in val:    
            _sum=0
            for component_i in self.components_table:
                    _sum+=component_i.x(self.T, self.p, value)*component_i.d()
            res.append(_sum)
        return res
    #
   # поиск -Коэффициенту сверхсжимаемости
    
    #что-то вспомогательное
    def A_m(self,a_m):
        res=[]
        for value in a_m:  
            res.append(value*self.p/((self.R**2)*(self.T**2)))
        return res
    def B_m(self, b_m):
        res=[]
        for value in b_m:  
            res.append(value*self.p/((self.R)*(self.T)))
        return res
    def C_m(self,c_m):
        res=[]
        for value in c_m:  
            res.append(value*self.p/((self.R)*(self.T)))
        return res
    def D_m(self,d_m):
        res=[]
        for value in d_m:  
            res.append(value*self.p/((self.R)*(self.T)))
        return res
   
    #конец чего-то вспомогательного
    def SupercompressibilityCoefficient(self,a_m,b_m,c_m,d_m,Flag):
        mass=[]
        A1=self.A_m(a_m)
        B1=self.B_m(b_m)
        C1=self.C_m(c_m)
        D1=self.D_m(d_m)
        a1=a_m
        b1=b_m
        c1=c_m
        d1=d_m
        print("susuususuusus",len(A1))
        for i in range(len(A1)):  
            z=sp.symbols('z',real=True)
            A=A1[i]
            B=B1[i]
            C=C1[i]
            D=D1[i]
            
            a=a1[i]
            b=b1[i]
            c=c1[i]
            d=d1[i]
            # _sum=(z**3)*self.p/(self.R*self.T)+(c*self.p/(self.R*self.T)+d*self.p/(self.R*self.T)-b*self.p/(self.R*self.T)-1)*(z**2)+(a/(self.R*self.T)-b*c*self.p/(self.R*self.T)+c*d*self.p/(self.R*self.T)-b*d*self.p/(self.R*self.T)-d-c)*z-(c*b*d*self.p/(self.R*self.T)+c*d+a*b/(self.R*self.T))
            _sum=z**3+(C+D-B-1)*(z**2)+(A-B*C+C*D-B*D-D-C)*z-(B*C*D+C*D+A*B)
            Equ=sp.Eq((_sum),0)
            res=list(sp.solveset(Equ,z,sp.Reals))
            print("resssotttto", res)
            if(len(res) != 1):    
                res=[i for i in res if i > 0] or 100000
                if(Flag == "max"):
                    mass.append(max(res))    
                else:
                    mass.append(min(res))
            else:
                (res,)=res
                # if(res < 0):
                #     continue
                    # res=10000
                mass.append(res)
            
            
        if(Flag == "max"):
            print(max(mass))
            return max(mass)
        else:
            print(min(mass))
            return min(mass)
        
   #    
   #логарифм летучести компонентов в паровой фазе
    def ln_f(self,component,a_m,b_m,c_m,d_m,z,Flag):
        A1=self.A_m(a_m)
        B1=self.B_m(b_m)
        C1=self.C_m(c_m)
        D1=self.D_m(d_m)
        a1=a_m
        b1=b_m
        c1=c_m
        d1=d_m
        res=[]
        
        for i in range(len(A1)):  
            value=self.V[i]
            
            A=A1[i]
            B=B1[i]
            C=C1[i]
            D=D1[i]
            a=a1[i]
            b=b1[i]
            c=c1[i]
            d=d1[i]
            _sum=0
            if(Flag=='max'):
                for component_j in self.components_table:
                    _sum+=component_j.y(self.T,self.p,value)*(1-self.C(component.name,component_j.name))*math.sqrt(component.a(self.T)*component_j.a(self.T))
                # print(component.y(self.T,self.p,value)<0)
                # print( z-B<0)
                # print((z+C)/(z+D) < 0)
                if(component.y(self.T,self.p,value)>=0 and z-B>=0 and (z+C)/(z+D)>=0):
                    res.append( math.log1p (component.y(self.T,self.p,value)*self.p) -math.log1p(z-B)-(A)*((((2*_sum)/a))-((component.c()-component.d())/(c-d)))*math.log1p((z+C)/(z+D))/(C-D) +component.B(self.p, self.T)/(z-B)-A*((component.C(self.p, self.T)/(z+C))-(component.D(self.p, self.T)/(z+D)))/(C-D) )# кто такой Bi

            else:
                for component_j in self.components_table:
                    _sum+=component_j.x(self.T,self.p,value)*(1-self.C(component.name,component_j.name))*math.sqrt(component.a(self.T)*component_j.a(self.T))
                # print(component.x(self.T,self.p,value)<0)
                # print( z-B<0)
                # print((z+C)/(z+D) < 0)
                if(component.x(self.T,self.p,value)>=0 and z-B>=0 and (z+C)/(z+D)>=0):
                    res.append( math.log1p(component.x(self.T,self.p,value)*self.p) - math.log1p(z-B)-(A)*((((2*_sum)/a))-((component.c()-component.d())/(c-d)))*math.log1p((z+C)/(z+D))/(C-D) +component.B(self.p, self.T)/(z-B)-A*((component.C(self.p, self.T)/(z+C))-(component.D(self.p, self.T)/(z+D)))/(C-D) )# кто такой Bi

           
        return res
   #
   #Корректируют значения коэффициентов распределения
    def K_2(self, val):
        flag=True
        while(flag):
            flag= False
            for component in self.components_table:
                # print(component.name)
                res1=self.ln_f(component, self.a_m(val), self.b_m(val), self.c_m(val), self.d_m(val), self.z_y,'max')
                res2=self.ln_f(component, self.x_a_m(val), self.x_b_m(val), self.x_c_m(val), self.x_d_m(val), self.z_x,'min')
                if(len(res1)==0 or len(res2) == 0):
                    print("Empty")
                    continue
                ln_f_L=max(res1)
                ln_f_V=max(res2)
                print('ln_f_L',ln_f_L)
                print('ln_f_V',ln_f_V)
                try:
                    f_L=math.exp(ln_f_L)
                except OverflowError:
                    f_L = float('inf')
                try:
                    f_V=math.exp(ln_f_V)
                except OverflowError:
                    f_V = float('inf')
                print(f_L)
                print(f_V)
                if(f_V != 0):
                    print("solution",math.fabs((f_L/f_V)-1),math.fabs((f_L/f_V)))
                    if(math.fabs((f_L/f_V)-1)>self.eps):
                        print("Бегом пересчитывать!!!!!!")  
                        flag =True
                        component.K=component.K*f_L/f_V
                        self.V=self.F_V()
                        print(self.F_v())
                        print("new_val",self.val)
                        self.z_y=self.SupercompressibilityCoefficient(self.a_m(self.val),self.b_m(self.val),self.c_m(self.val),self.d_m(self.val),'max')
                        self.z_x=self.SupercompressibilityCoefficient(self.x_a_m(self.val),self.x_b_m(self.val),self.x_c_m(self.val),self.x_d_m(self.val),'min')
                        print("z_y",self.z_y)
                        print("z_x",self.z_x)
        print("Результаты:")            
        summ_x=0
        summ_y=0
        summ_z=0
        for component in self.components_table:
            print(component.name, " x = ", component.x_value)
            print(component.name, " y = ", component.y_value)
            print(component.name, " z = ", component.z)
            summ_x+=component.x_value
            summ_y+=component.y_value
            summ_z+=component.z
        print("Проверка 1 :")   
        print("summ_x",summ_x )
        print("summ_y",summ_y )
        print("summ_z",summ_z )
        self.summ_x=summ_x
        self.summ_y=summ_y
        self.summ_z=summ_z
        print("Проверка 2 :")
        summF=0
        for component in self.components_table:
            summF+=component.z*(component.K-1)/(self.val[0]*(component.K-1)+1)
        print("summ F(V)=",summF)
        print("Проверка 3 :")
        summXLYV=0
        for component in self.components_table:
            summXLYV+=component.x_value*(1-self.val[0])+component.y_value*(self.val[0])-component.z
        print("summXLYV =",summXLYV)
        #return self.components_table
class Component():
    def __init__(self,name,T_c,p_c,w,z,Z_c,Omega_c,Psy):
        self.T_c=T_c #критические температуру 
        self.p_c=p_c #давление
        self.w=w #ацентрический фактор 
        self.R=8.31446261815324 # а что это?
        self.z=z #мольная доля компонента в смеси
        self.name=name
        self.K=1
        self.Z_c=self.calc_Z_c(Z_c)
        self.Omega_c=self.calc_Omega_c(Omega_c)
        self.Psy=self.calc_Psy(Psy)
        self.x_value =1
        self.y_value =1
    #self.T=T # а это откуда?
    def calc_Z_c(self,Z_c):
        if(Z_c==1):
            return 0.3357-0.0294*self.w
        else:
            return Z_c
        
    def calc_Omega_c(self,Omega_c):
        if(Omega_c==1):
            return 0.75001
        else:
            return Omega_c
        
    def calc_Psy(self,Psy):
        if(Psy==1):
            if(self.w < 0.4489):
                return 1.050+0.105*self.w+0.482*(self.w**2)
            elif(self.w>0.4489):
                return 0.429+1.004*self.w+1.561*(self.w**2)
            else:
                return 1.194
        else:
            return Psy
        
   #coefficients of the equation of state
    def alfa(self):
        return self.Omega_c**3
    def betta(self):
        return self.Z_c+self.Omega_c-1
    def sigma(self):
        return -1*self.Z_c+self.Omega_c*(0.5+(self.Omega_c-0.75)**(0.5))
    def delta(self):
        return -1*self.Z_c+self.Omega_c*(0.5-(self.Omega_c-0.75)**(0.5))
   #
   #values
    def a(self,T):
        a_c=(self.alfa()*((self.R)**2)*((self.T_c)**2))/self.p_c
        calc=(1+self.Psy*(1-((T/self.T_c)**(0.5))))**2
        return (a_c*calc)
    def b(self):
        return self.betta()*self.R*self.T_c/self.p_c
    def c(self):
        return self.sigma()*self.R*self.T_c/self.p_c
    def d(self):
        return self.delta()*self.R*self.T_c/self.p_c
   #
   #что-то вспомогательное

    def B(self, p,T):
        return self.b()*p/(self.R*T)
    def C(self,p,T):
        return self.c()*p/(self.R*T)
    def D(self,p,T):
        return self.d()*p/(self.R*T)
    
   #distribution coefficients of components
    def p_Si(self,T):
        return np.exp(5.373*(1+self.w)*(1-self.T_c/T))*self.p_c
    def f_K(self,T,p):
        self.K=self.p_Si(T)/p
        #return self.p_Si(T)/p
   #
   #уравнением фазовой концентрации компонентов смеси.
    def y(self,T,p,V):
        # print("y",(self.z*self.K)/(V*(self.K-1)+1))
        res =(self.z*self.K)/(V*(self.K-1)+1)
        self.y_value = res
        return res
   
   #
   #мольные доли компонентов смеси в жидкой фазе. 
    def x(self,T,p,V):
        # print("x",self.z/(V*(self.K-1)+1))
        res=self.z/(V*(self.K-1)+1)
        self.x_value = res
        return res
        
   #
    
class Mix():
     def __init__(self,mixes,consts,params):
         self.mixes=mixes
         self.table_consts=self.ReadFile(consts)
         self.table_params=self.ReadFile(params)
     def ReadFile(self,file_name):
         table=pd.read_excel('./'+file_name,index_col="c")
         return table
     def getTableValue(self,table,value_name,name):
         res =table[value_name][name]
         return res
     def CreateMix(self,mix_number):
        table_mix = pd.read_excel('./'+self.mixes)
        headers_mix=table_mix.columns.values.tolist()
        values_mix=table_mix.iloc[mix_number]
        mass=[]
        
        #табличка с означениями температуры и давления + вырезать из класса части R
        for i in range(len(values_mix)):
            name=headers_mix[i]
            if(values_mix[i]!=0):
                # print("name",name)
                mass.append(Component(name,self.getTableValue(self.table_consts,'T_c',name),self.getTableValue(self.table_consts,'P_c',name),self.getTableValue(self.table_consts,'w',name),values_mix[i],self.getTableValue(self.table_params,'Z_c',name),self.getTableValue(self.table_params,'Omega_c',name),self.getTableValue(self.table_params,'Psy',name)))
        return mass
    
# mass=[Component("CH4",0.0001,1,0.0001,0.0001,0.05),Component("CH4",1,0.0001,1,2,0.03),Component("CH4",1,0.0001,1,2,0.03),Component("CH4",1,0.0001,1,2,0.43),Component("CH4",1,1,0.0001,1,0.15)]
# Test=Solution(mass,1,5,5)
# TEst_C=Component("CH4",0.0001,1,0.0001,0.0001,0.05)
# Test.K_2()
Mix1=Mix("mix.xlsx","consts.xlsx","params.xlsx")
# for ind in range(4,6):
new_mix=Mix1.CreateMix(0)
Test=Solution(new_mix,15,273.15+5)
Test.main()
####################
# построение 2д графика
massX=[]
massY=[]
len_p=np.linspace(3,7,20)
for i in len_p:
    # i=k/10
    print(i,"New P")
    Test=Solution(new_mix,i,273.15+5)
    Test.main()
    massX.append(Test.val[0]*100)
    massY.append((1-Test.val[0])*100)
    print("last V",Test.val)
fig, ax = plt.subplots()
# ax.plot(len_p,massX,label="X")
ax.plot(len_p,massY,label="Y")
# ax.plot(X, res2[:,5],label="явная схема")
ax.set_xlabel("p")
ax.set_ylabel("%")
# ax.set_title("сумма молей X")
ax.legend()
plt.show()
fig, ax = plt.subplots()
ax.plot(len_p,massX,label="X")
ax.set_xlabel("p")
ax.set_ylabel("%")
ax.legend()
plt.show()
#####################
# #построение поверхности
# len_p=range(7,7+10)
# len_T=range(273,273+10)
# T=np.zeros((len(len_p), len(len_T)))
# for i in len_p:
#     for j in len_T:
#         Test=Solution(new_mix,i,j)
#         Test.main()
#         T[i-7,j-273]=Test.summ_x
#         print(Test.summ_x)
# X=np.linspace(7,7+10,10)
# Y=np.linspace(273,273+10,10)
# X,Y = np.meshgrid(X,Y)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Z=T
# surf = ax.plot_surface(X, Y, Z, cmap='coolwarm',linewidth=0)
# ax.set_xlabel("p")
# ax.set_ylabel("T")
# ax.set_zlabel("%")
# ax.set_title("% от Температуры и Давленя")
#####################

#Test.K_2()
# print("rfif",TEst_C.y(1,1,Test.V[1]))
#Test.F_V()
# print(Test.ln_f(TEst_C,Test.a_m(),Test.b_m(),Test.c_m(),Test.d_m(),Test.z_y))
# print(Test.ln_f(TEst_C,Test.x_a_m(),Test.x_b_m(),Test.x_c_m(),Test.x_d_m(),Test.z_x))