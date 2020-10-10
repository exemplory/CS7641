# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 21:59:50 2020

@author: vivek
"""

import pandas as pd 			  		 			 	 	 		 		 	  		   	  			  	
import matplotlib.pyplot as plt 			  		 			 	 	 		 		 	  		   	  			  	
import numpy as np 			  		 			 	 	 		 		 	  		   	  			  			   	  			    		  		  		    	 		 		   		 		  


def plot_rhc_score(df_test, df_train, title, xlabel, ylabel, iterations, fileName): 
    
    lw = 2	
    plt.grid(True, linestyle = "--")
    plt.plot(np.arange(1, len(iterations) + 1), df_test, label="Testing Accuracy", color="g", lw=lw)
    plt.plot(np.arange(1, len(iterations) + 1), df_train, label="Training Accuracy", color="b", lw=lw)
   
    
    plt.xticks(np.arange(1, len(iterations) + 1), iterations)
    plt.xlabel(xlabel)  		   	  			    		  		  		    	 		 		   		 		  
    plt.ylabel(ylabel)  
    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=2)

    plt.title(title)
    #plt.show()  		   	  
    plt.savefig(fileName, bbox_inches='tight',pad_inches=0.2)
    plt.close() 

def plot_rhc_time(df_train, title, xlabel, ylabel, iterations, fileName): 
    
    lw = 2	
    plt.grid(True, linestyle = "--")
    plt.plot(np.arange(1, len(iterations) + 1), df_train, label="Training Time", color="b", lw=lw)
   
    plt.xticks(np.arange(1, len(iterations) + 1), iterations)
    plt.xlabel(xlabel)  		   	  			    		  		  		    	 		 		   		 		  
    plt.ylabel(ylabel)  
    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=2)

    plt.title(title)
    #plt.show()  		   	  
    plt.savefig(fileName, bbox_inches='tight',pad_inches=0.2)
    plt.close()
  

def plot_sa_score(df_test, labels, title, xlabel, ylabel, iterations, fileName): 

    lw = 2	
    plt.grid(True, linestyle = "--")
    plt.plot(np.arange(1, len(iterations) + 1), df_test[0], label=labels[0], color="tab:blue", lw=lw)
    plt.plot(np.arange(1, len(iterations) + 1), df_test[1], label=labels[1], color="tab:orange", lw=lw)
    plt.plot(np.arange(1, len(iterations) + 1), df_test[2], label=labels[2], color="tab:green", lw=lw)
    plt.plot(np.arange(1, len(iterations) + 1), df_test[3], label=labels[3], color="tab:red", lw=lw)
    plt.plot(np.arange(1, len(iterations) + 1), df_test[4], label=labels[4], color="tab:purple", lw=lw)
    plt.plot(np.arange(1, len(iterations) + 1), df_test[5], label=labels[5], color="tab:brown", lw=lw)
    
    
    plt.xticks(np.arange(1, len(iterations) + 1), iterations)
    plt.xlabel(xlabel)  		   	  			    		  		  		    	 		 		   		 		  
    plt.ylabel(ylabel)  
    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=3)

    plt.title(title)
    #plt.show()  		   	  
    plt.savefig(fileName, bbox_inches='tight',pad_inches=0.2)
    plt.close() 

def plot_ga_score(df_test, labels, title, xlabel, ylabel, iterations, fileName): 

    lw = 2	
    plt.grid(True, linestyle = "--")
    plt.plot(np.arange(1, len(iterations) + 1), df_test[0], label=labels[0], color="tab:blue", lw=lw)
    plt.plot(np.arange(1, len(iterations) + 1), df_test[1], label=labels[1], color="tab:orange", lw=lw)
    plt.plot(np.arange(1, len(iterations) + 1), df_test[2], label=labels[2], color="tab:green", lw=lw)
    plt.plot(np.arange(1, len(iterations) + 1), df_test[3], label=labels[3], color="tab:red", lw=lw)
    plt.plot(np.arange(1, len(iterations) + 1), df_test[4], label=labels[4], color="tab:purple", lw=lw)
    plt.plot(np.arange(1, len(iterations) + 1), df_test[5], label=labels[5], color="tab:brown", lw=lw)
    plt.plot(np.arange(1, len(iterations) + 1), df_test[6], label=labels[6], color="tab:pink", lw=lw)
    plt.plot(np.arange(1, len(iterations) + 1), df_test[7], label=labels[7], color="tab:gray", lw=lw)
    plt.plot(np.arange(1, len(iterations) + 1), df_test[8], label=labels[8], color="tab:olive", lw=lw)
    plt.plot(np.arange(1, len(iterations) + 1), df_test[9], label=labels[9], color="tab:cyan", lw=lw)

    
    plt.xticks(np.arange(1, len(iterations) + 1), iterations)
    plt.xlabel(xlabel)  		   	  			    		  		  		    	 		 		   		 		  
    plt.ylabel(ylabel)  
    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=3)

    plt.title(title)
    #plt.show()  		   	  
    plt.savefig(fileName, bbox_inches='tight',pad_inches=0.2)
    plt.close() 

def plot_rhc():
    plt.rc('figure', figsize=(10, 5))

    df_test  = pd.read_csv("WineNN_RHC_Test.csv", delimiter = ",", header="infer")
    df_train  = pd.read_csv("WineNN_RHC_Train.csv", delimiter = ",", header="infer")
    
    plot_rhc_score(df_test['Test_Accuracy'], df_train['Train_Accuracy'], "Figure 4.3: Wine Quality \n Random Hill Climbing Learning Curve", "Iterations", "Accuracy", df_test['Iterations'], "Figure 4.3.jpg")
    plot_rhc_time(df_train['Train_Time'],  "Figure 4.4: Wine Quality \n Random Hill Climbing Training Time", "Iterations", "Training Time (in seconds)", df_train['Iterations'], "Figure 4.4.jpg")

def plot_sa():
    plt.rc('figure', figsize=(10, 5))
    df_test  = pd.read_csv("WineNN_SA_Test.csv", delimiter = ",", header="infer")
    df_train  = pd.read_csv("WineNN_SA_Train.csv", delimiter = ",", header="infer")
    
    ## 1 ONE
    labels = ["T=1E9, CR=0.45", "T=1E9, CR=0.55", "T=1E9, CR=0.65", "T=1E9, CR=0.75", "T=1E9, CR=0.85", "T=1E9, CR=0.95"]
    iterations = df_test['Iterations'][df_test['Hyper'] == '(1.0E9 - 0.45)'].tolist()

    a = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E9 - 0.45)'].tolist()
    b = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E9 - 0.55)'].tolist()
    c = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E9 - 0.65)'].tolist()
    d = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E9 - 0.75)'].tolist()
    e = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E9 - 0.85)'].tolist()
    f = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E9 - 0.95)'].tolist()
    plot_sa_score([a,b,c,d,e,f], labels, "Figure 4.5.1: Wine Quality - Simulated Annealing Training Accuracy\nVarying Initial Temperature and Cooling Rate", "Iterations", "Accuracy", iterations, "Figure 4.5.1.jpg")
    
    a = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E9 - 0.45)'].tolist()
    b = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E9 - 0.55)'].tolist()
    c = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E9 - 0.65)'].tolist()
    d = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E9 - 0.75)'].tolist()
    e = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E9 - 0.85)'].tolist()
    f = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E9 - 0.95)'].tolist()
    plot_sa_score([a,b,c,d,e,f], labels, "Figure 4.5.2: Wine Quality - Simulated Annealing Testing Accuracy\nVarying Initial Temperature and Cooling Rate", "Iterations", "Accuracy", iterations, "Figure 4.5.2.jpg")

    a = df_train['Train_Time'][df_test['Hyper'] == '(1.0E9 - 0.45)'].tolist()
    b = df_train['Train_Time'][df_test['Hyper'] == '(1.0E9 - 0.55)'].tolist()
    c = df_train['Train_Time'][df_test['Hyper'] == '(1.0E9 - 0.65)'].tolist()
    d = df_train['Train_Time'][df_test['Hyper'] == '(1.0E9 - 0.75)'].tolist()
    e = df_train['Train_Time'][df_test['Hyper'] == '(1.0E9 - 0.85)'].tolist()
    f = df_train['Train_Time'][df_test['Hyper'] == '(1.0E9 - 0.95)'].tolist()
    plot_sa_score([a,b,c,d,e,f], labels, "Figure 4.5.3: Wine Quality - Simulated Annealing Training Time\nVarying Initial Temperature and Cooling Rate", "Iterations", "Training Time (in seconds)", iterations, "Figure 4.5.3.jpg")


    ## 2 TWO
    labels = ["T=1E10, CR=0.45", "T=1E10, CR=0.55", "T=1E10, CR=0.65", "T=1E10, CR=0.75", "T=1E10, CR=0.85", "T=1E10, CR=0.95"]
    iterations = df_test['Iterations'][df_test['Hyper'] == '(1.0E10 - 0.45)'].tolist()
    
    a = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E10 - 0.45)'].tolist()
    b = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E10 - 0.55)'].tolist()
    c = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E10 - 0.65)'].tolist()
    d = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E10 - 0.75)'].tolist()
    e = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E10 - 0.85)'].tolist()
    f = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E10 - 0.95)'].tolist()
    plot_sa_score([a,b,c,d,e,f], labels, "Figure 4.6.1: Wine Quality - Simulated Annealing Training Accuracy\nVarying Initial Temperature and Cooling Rate", "Iterations", "Accuracy", iterations, "Figure 4.6.1.jpg")

    a = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E10 - 0.45)'].tolist()
    b = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E10 - 0.55)'].tolist()
    c = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E10 - 0.65)'].tolist()
    d = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E10 - 0.75)'].tolist()
    e = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E10 - 0.85)'].tolist()
    f = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E10 - 0.95)'].tolist()
    plot_sa_score([a,b,c,d,e,f], labels, "Figure 4.6.2: Wine Quality - Simulated Annealing Testing Accuracy\nVarying Initial Temperature and Cooling Rate", "Iterations", "Accuracy", iterations, "Figure 4.6.2.jpg")

    a = df_train['Train_Time'][df_test['Hyper'] == '(1.0E10 - 0.45)'].tolist()
    b = df_train['Train_Time'][df_test['Hyper'] == '(1.0E10 - 0.55)'].tolist()
    c = df_train['Train_Time'][df_test['Hyper'] == '(1.0E10 - 0.65)'].tolist()
    d = df_train['Train_Time'][df_test['Hyper'] == '(1.0E10 - 0.75)'].tolist()
    e = df_train['Train_Time'][df_test['Hyper'] == '(1.0E10 - 0.85)'].tolist()
    f = df_train['Train_Time'][df_test['Hyper'] == '(1.0E10 - 0.95)'].tolist()
    plot_sa_score([a,b,c,d,e,f], labels, "Figure 4.6.3: Wine Quality - Simulated Annealing Training Time\nVarying Initial Temperature and Cooling Rate", "Iterations", "Training Time (in seconds)", iterations, "Figure 4.6.3.jpg")


    ## 3 THREE
    labels = ["T=1E11, CR=0.45", "T=1E11, CR=0.55", "T=1E11, CR=0.65", "T=1E11, CR=0.75", "T=1E11, CR=0.85", "T=1E11, CR=0.95"]
    iterations = df_test['Iterations'][df_test['Hyper'] == '(1.0E11 - 0.45)'].tolist()
    
    a = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E11 - 0.45)'].tolist()
    b = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E11 - 0.55)'].tolist()
    c = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E11 - 0.65)'].tolist()
    d = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E11 - 0.75)'].tolist()
    e = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E11 - 0.85)'].tolist()
    f = df_train['Train_Accuracy'][df_test['Hyper'] == '(1.0E11 - 0.95)'].tolist()
    plot_sa_score([a,b,c,d,e,f], labels, "Figure 4.7.1: Wine Quality - Simulated Annealing Training Accuracy\nVarying Initial Temperature and Cooling Rate", "Iterations", "Accuracy", iterations, "Figure 4.7.1.jpg")

    a = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E11 - 0.45)'].tolist()
    b = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E11 - 0.55)'].tolist()
    c = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E11 - 0.65)'].tolist()
    d = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E11 - 0.75)'].tolist()
    e = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E11 - 0.85)'].tolist()
    f = df_test['Test_Accuracy'][df_test['Hyper'] == '(1.0E11 - 0.95)'].tolist()
    plot_sa_score([a,b,c,d,e,f], labels, "Figure 4.7.2: Wine Quality - Simulated Annealing Testing Accuracy\nVarying Initial Temperature and Cooling Rate", "Iterations", "Accuracy", iterations, "Figure 4.7.2.jpg")

    a = df_train['Train_Time'][df_test['Hyper'] == '(1.0E11 - 0.45)'].tolist()
    b = df_train['Train_Time'][df_test['Hyper'] == '(1.0E11 - 0.55)'].tolist()
    c = df_train['Train_Time'][df_test['Hyper'] == '(1.0E11 - 0.65)'].tolist()
    d = df_train['Train_Time'][df_test['Hyper'] == '(1.0E11 - 0.75)'].tolist()
    e = df_train['Train_Time'][df_test['Hyper'] == '(1.0E11 - 0.85)'].tolist()
    f = df_train['Train_Time'][df_test['Hyper'] == '(1.0E11 - 0.95)'].tolist()
    plot_sa_score([a,b,c,d,e,f], labels, "Figure 4.7.3: Wine Quality - Simulated Annealing Training Time\nVarying Initial Temperature and Cooling Rate", "Iterations", "Training Time (in seconds)", iterations, "Figure 4.7.3.jpg")

def plot_ga():
    plt.rc('figure', figsize=(10, 5))
    df_test  = pd.read_csv("WineNN_GA_Test.csv", delimiter = ",", header="infer")
    df_train  = pd.read_csv("WineNN_GA_Train.csv", delimiter = ",", header="infer")
    
    ## 1 ONE
    labels = ['(25-10-20)','(25-25-10)','(50-50-2)','(50-10-10)','(100-100-2)','(100-100-10)','(250-25-2)','(250-150-10)','(500-10-10)', '(500-50-25)']
    iterations = df_test['Iterations'][df_test['Hyper'] == '[25 - 10 - 2]'].tolist()

    a = df_train['Train_Accuracy'][df_test['Hyper'] == '[25 - 10 - 2]'].tolist()
    b = df_train['Train_Accuracy'][df_test['Hyper'] == '[25 - 25 - 10]'].tolist()
    c = df_train['Train_Accuracy'][df_test['Hyper'] == '[50 - 50 - 2]'].tolist()
    d = df_train['Train_Accuracy'][df_test['Hyper'] == '[50 - 10 - 10]'].tolist()
    e = df_train['Train_Accuracy'][df_test['Hyper'] == '[100 - 100 - 2]'].tolist()
    f = df_train['Train_Accuracy'][df_test['Hyper'] == '[100 - 100 - 10]'].tolist()
    g = df_train['Train_Accuracy'][df_test['Hyper'] == '[250 - 25 - 2]'].tolist()
    h = df_train['Train_Accuracy'][df_test['Hyper'] == '[250 - 150 - 10]'].tolist()
    i = df_train['Train_Accuracy'][df_test['Hyper'] == '[500 - 10 - 10]'].tolist()
    j = df_train['Train_Accuracy'][df_test['Hyper'] == '[500 - 50 - 25]'].tolist()
    plot_ga_score([a,b,c,d,e,f,g,h,i,j], labels, "Figure 4.8.1: Wine Quality - Genetic Algorithm Training Accuracy\nVarying Population Size, To Mate and To Mutate", "Iterations", "Accuracy", iterations, "Figure 4.8.1.jpg")
    
    a = df_test['Test_Accuracy'][df_test['Hyper'] == '[25 - 10 - 2]'].tolist()
    b = df_test['Test_Accuracy'][df_test['Hyper'] == '[25 - 25 - 10]'].tolist()
    c = df_test['Test_Accuracy'][df_test['Hyper'] == '[50 - 50 - 2]'].tolist()
    d = df_test['Test_Accuracy'][df_test['Hyper'] == '[50 - 10 - 10]'].tolist()
    e = df_test['Test_Accuracy'][df_test['Hyper'] == '[100 - 100 - 2]'].tolist()
    f = df_test['Test_Accuracy'][df_test['Hyper'] == '[100 - 100 - 10]'].tolist()
    g = df_test['Test_Accuracy'][df_test['Hyper'] == '[250 - 25 - 2]'].tolist()
    h = df_test['Test_Accuracy'][df_test['Hyper'] == '[250 - 150 - 10]'].tolist()
    i = df_test['Test_Accuracy'][df_test['Hyper'] == '[500 - 10 - 10]'].tolist()
    j = df_test['Test_Accuracy'][df_test['Hyper'] == '[500 - 50 - 25]'].tolist()
    plot_ga_score([a,b,c,d,e,f,g,h,i,j], labels, "Figure 4.8.2: Wine Quality - Genetic Algorithm Testing Accuracy\nVarying Population Size, To Mate and To Mutate", "Iterations", "Accuracy", iterations, "Figure 4.8.2.jpg")

    a = df_train['Train_Time'][df_test['Hyper'] == '[25 - 10 - 2]'].tolist()
    b = df_train['Train_Time'][df_test['Hyper'] == '[25 - 25 - 10]'].tolist()
    c = df_train['Train_Time'][df_test['Hyper'] == '[50 - 50 - 2]'].tolist()
    d = df_train['Train_Time'][df_test['Hyper'] == '[50 - 10 - 10]'].tolist()
    e = df_train['Train_Time'][df_test['Hyper'] == '[100 - 100 - 2]'].tolist()
    f = df_train['Train_Time'][df_test['Hyper'] == '[100 - 100 - 10]'].tolist()
    g = df_train['Train_Time'][df_test['Hyper'] == '[250 - 25 - 2]'].tolist()
    h = df_train['Train_Time'][df_test['Hyper'] == '[250 - 150 - 10]'].tolist()
    i = df_train['Train_Time'][df_test['Hyper'] == '[500 - 10 - 10]'].tolist()
    j = df_train['Train_Time'][df_test['Hyper'] == '[500 - 50 - 25]'].tolist()
    plot_ga_score([a,b,c,d,e,f,g,h,i,j], labels, "Figure 4.8.3: Wine Quality - Genetic Algorithm Training Time\nVarying Population Size, To Mate and To Mutate", "Iterations", "Training Time (in seconds)", iterations, "Figure 4.8.3.jpg")


 

if __name__ == "__main__": 	
    #plt.rcParams.update({'font.size': 14})	
    #plot_nn()	  		 			 	 	 		 		 	  		   	  			  	
    #plot_travelling_salesman_charts()
    #plot_continous_peaks_charts()
    #plot_flip_flop_charts()
    
    #plot_rhc()
    #plot_sa()
    plot_ga()
    
    #plot_final_nn()
