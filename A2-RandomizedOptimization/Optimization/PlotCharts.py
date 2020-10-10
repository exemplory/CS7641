# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 21:51:06 2020

@author: vivek
"""

import pandas as pd 			  		 			 	 	 		 		 	  		   	  			  	
import matplotlib.pyplot as plt 			  		 			 	 	 		 		 	  		   	  			  	
import numpy as np 			  		 			 	 	 		 		 	  		   	  			  			   	  			    		  		  		    	 		 		   		 		  

def plot_score(df, title, xlabel, ylabel, filename):  		   	  			    		  		  		    	 		 		   		 		  
    iterations = [20, 100, 500, 1000, 2500, 5000, 7500, 10000, 25000, 50000]
 
    lw = 2	
    plt.grid(True, linestyle = "--")
    plt.plot(np.arange(1,11), df['RHC_Score'], label="RHC", color="b", lw=lw)
    plt.plot(np.arange(1,11), df['SA_Score'], label="SA", color="g", lw=lw)
    plt.plot(np.arange(1,11), df['GA_Score'], label="GA", color="r", lw=lw)
    plt.plot(np.arange(1,11), df['MIMIC_Score'], label="MIMIC", color="c", lw=lw)
    
    plt.xticks(np.arange(1,11), iterations)
    plt.xlabel(xlabel)  		   	  			    		  		  		    	 		 		   		 		  
    plt.ylabel(ylabel)  
    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=2)

    plt.title(title)
    #plt.show()  		   	  
    plt.savefig(filename, bbox_inches='tight',pad_inches=0.2)
    plt.close()

def plot_opt_score(size, df, title, xlabel, ylabel, filename):  		   	  			    		  		  		    	 		 		   		 		  
    #iterations = [20, 100, 500, 1000, 2500, 5000, 7500, 10000, 25000, 50000]
 
    lw = 2	
    plt.grid(True, linestyle = "--")
    plt.plot(np.arange(1,size), df['Score'], color="b", lw=lw)
    
    plt.xticks(np.arange(1,size), df['Params'], rotation='vertical')
    plt.xlabel(xlabel)  		   	  			    		  		  		    	 		 		   		 		  
    plt.ylabel(ylabel)  

    plt.title(title)
    #plt.show()  		   	  
    plt.savefig(filename, bbox_inches='tight',pad_inches=0.2)
    plt.close()

    
def plot_time(df, title, xlabel, ylabel, filename):  		   	  			    		  		  		    	 		 		   		 		  
    iterations = [20, 100, 500, 1000, 2500, 5000, 7500, 10000, 25000, 50000]
 
    lw = 2	
    plt.grid(True, linestyle = "--")
    #plt.yscale('log')
    plt.plot(np.arange(1,11), df['RHC_Time'], label="RHC", color="b", lw=lw)
    plt.plot(np.arange(1,11), df['SA_Time'], label="SA", color="g", lw=lw)
    plt.plot(np.arange(1,11), df['GA_Time'], label="GA", color="r", lw=lw)
    plt.plot(np.arange(1,11), df['MIMIC_Time'], label="MIMIC", color="c", lw=lw)
    
    plt.xticks(np.arange(1,11), iterations)
    plt.xlabel(xlabel)  		   	  			    		  		  		    	 		 		   		 		  
    plt.ylabel(ylabel)  
    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=2)

    plt.title(title)
    #plt.show()  		   	  
    plt.savefig(filename, bbox_inches='tight',pad_inches=0.2)
    plt.close()    
    
def plot_opt_time(size, df, title, xlabel, ylabel, filename):  		   	  			    		  		  		    	 		 		   		 		  
 
    lw = 2	
    plt.grid(True, linestyle = "--")
    #plt.yscale('log')
    plt.plot(np.arange(1,size), df['Time'], color="b", lw=lw)
    
    plt.xticks(np.arange(1,size), df['Params'], rotation='vertical')
    plt.xlabel(xlabel)  		   	  			    		  		  		    	 		 		   		 		  
    plt.ylabel(ylabel)  

    plt.title(title)
    #plt.show()  		   	  
    plt.savefig(filename, bbox_inches='tight',pad_inches=0.2)
    plt.close()    

def plot_cp_chart():
    plt.rc('figure', figsize=(10, 5))
    df  = pd.read_csv("ContinuousPeaksResult.csv", delimiter = ",", header="infer")
    plot_score(df, "Figure 1.1: Continuous Peaks\n Optimal Fitness Values", "Iterations", "Optimal Fitness", "Figure 1.1.jpg")
    plot_time(df, "Figure 1.2: Continuous Peaks\n Computation Time", "Iterations", "Computation Time (in secs)", "Figure 1.2.jpg")

def plot_ts_chart():
    plt.rc('figure', figsize=(10, 5))
    df  = pd.read_csv("TravellingSalesmanResult.csv", delimiter = ",", header="infer")
    plot_score(df, "Figure 2.1: Travelling Salesman\n Optimal Fitness Values", "Iterations", "Optimal Fitness", "Figure 2.1.jpg")
    plot_time(df, "Figure 2.2: Travelling Salesman\n Computation Time", "Iterations", "Computation Time (in secs)", "Figure 2.2.jpg")
    
def plot_4p_chart():
    plt.rc('figure', figsize=(10, 5))
    df  = pd.read_csv("FourPeaksResult.csv", delimiter = ",", header="infer")
    plot_score(df, "Figure 3.1: Four Peaks\n Optimal Fitness Values", "Iterations", "Optimal Fitness", "Figure 3.1.jpg")
    plot_time(df, "Figure 3.2: Four Peaks\n Computation Time", "Iterations", "Computation Time (in secs)", "Figure 3.2.jpg")


def plot_cp_opt_chart():
    plt.rc('figure', figsize=(20, 5))
    df  = pd.read_csv("ContinuousPeaksOptimizationResult.csv", delimiter = ",", header="infer")
    plot_opt_score(51, df, "Figure 1.3: Continuous Peaks - Optimize Simulated Annealing\n Optimal Fitness Values", "(Temperature - Cooling Rate)", "Optimal Fitness", "Figure 1.3.jpg")
    plot_opt_time(51, df, "Figure 1.4: Continuous Peaks - Optimize Simulated Annealing\n Computation Time", "(Temperature - Cooling Rate)", "Computation Time (in secs)", "Figure 1.4.jpg")
    
def plot_ts_opt_chart():
    plt.rc('figure', figsize=(20, 5))
    df  = pd.read_csv("TravellingSalesmanOptimizationResult.csv", delimiter = ",", header="infer")
    #print(df)
    plot_opt_score(67, df, "Figure 2.3: Travelling Salesman - Optimize Genetic Algorithm\n Optimal Fitness Values", "(Population Size - To Mate Size - To Mutate Size)", "Optimal Fitness", "Figure 2.3.jpg")
    plot_opt_time(67, df, "Figure 2.4: Travelling Salesman - Optimize Genetic Algorithm\n Computation Time", "(Population Size - To Mate Size - To Mutate Size)", "Computation Time (in secs)", "Figure 2.4.jpg")

def plot_4p_opt_chart():
    plt.rc('figure', figsize=(20, 5))
    df  = pd.read_csv("FourPeaksOptimizationResult.csv", delimiter = ",", header="infer")
    plot_opt_score(31, df, "Figure 3.3: Four Peaks - Optimize MIMIC\n Optimal Fitness Values", "(Population Size - To Keep Size)", "Optimal Fitness", "Figure 3.3.jpg")
    plot_opt_time(31, df, "Figure 3.4: Four Peaks - Optimize MIMIC\n Computation Time", "(Population Size - To Keep Size)", "Computation Time (in secs)", "Figure 3.4.jpg")
    
    
if __name__ == "__main__": 			  		 			 	 	 		 		 	  		   	  			  	
    plot_cp_chart()
    plot_ts_chart()
    plot_4p_chart()
    plot_cp_opt_chart()
    plot_ts_opt_chart()
    plot_4p_opt_chart()