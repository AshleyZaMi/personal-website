""" Ashley Zapata Minero
    Project03
    Description: This project is about creating a penguin matrix using a nested for Loop
    and if the columns equals to the rows then that's when you know it's diagnol and should be a histogram. Then for each scatter plot
    do data the one on top vs data on the side to see them in the plot either above or below the histogram. The second part is about filtering the data and then
    displaying that data in the chart in different scatter calls but in the same plot. different scatter calls allows them to have different colors. then of course
    for both of them doing the legend, title, and for the second part x, and y axis labeling
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    # write your code here
    
    #Accessing the file
    file_name = "penguins.csv"
    df_penguin = pd.read_csv(file_name)
    
    #1.Create a Scatterplot Matrix of all of the numeric attributes
    #(1)Make an area to plot in
    fig, ax = plt.subplots(4,4, figsize=(16,9))
    
    #filtering out the data we need
    df_penguins = df_penguin[['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']]
    columns = df_penguins.columns
    
    
    for outside in range(4): 
        for inside in range(4):
            if outside == inside: #outside == inside would be across the histogram
                if outside == 0: #giving the name of the column to print depending on the index
                    ax[inside,outside].hist(df_penguins[columns[0]], bins = 10, label=columns[0])
                    ax[inside,outside].set(xlabel = columns[0]) #setting the label of the graph
                elif outside == 1:
                    ax[inside,outside].hist(df_penguins[columns[1]], bins = 10, label=columns[1])
                    ax[inside,outside].set(xlabel = columns[1])
                elif outside == 2:
                    ax[inside,outside].hist(df_penguins[columns[2]], bins = 10, label=columns[2])
                    ax[inside,outside].set(xlabel = columns[2])
                elif outside == 3:
                    ax[inside,outside].hist(df_penguins[columns[3]], bins = 10, label=columns[3])
                    ax[inside,outside].set(xlabel = columns[3])
                   
                       
            else: #outside -> column, inside-> row
                #versus part
                if outside == 0: # this is the columns its how pandas is percieving it
                    if inside == 1: #row 1
                        ax[inside,outside].scatter(df_penguins[columns[1]], df_penguins[columns[0]])
                    elif inside == 2: #row 2
                        ax[inside,outside].scatter(df_penguins[columns[2]], df_penguins[columns[0]])
                    elif inside == 3: #row 3
                        ax[inside,outside].scatter(df_penguins[columns[3]], df_penguins[columns[0]])
                if outside == 1: #column
                    if inside == 0: #row
                        ax[inside,outside].scatter(df_penguins[columns[0]], df_penguins[columns[1]])
                    elif inside == 2: #row
                        ax[inside,outside].scatter(df_penguins[columns[2]], df_penguins[columns[1]])
                    elif inside == 3: #row
                        ax[inside,outside].scatter(df_penguins[columns[3]], df_penguins[columns[1]])
                if outside == 2: #column
                    if inside == 0: #row 0
                        ax[inside,outside].scatter(df_penguins[columns[0]], df_penguins[columns[2]])
                    elif inside == 1:
                        ax[inside,outside].scatter(df_penguins[columns[1]], df_penguins[columns[2]])
                    elif inside == 3:
                        ax[inside,outside].scatter(df_penguins[columns[3]], df_penguins[columns[2]])
                if outside == 3:#format seems to be column, row
                    if inside == 0:
                        ax[inside,outside].scatter(df_penguins[columns[0]], df_penguins[columns[3]])
                    elif inside == 1:#row 1
                        ax[inside,outside].scatter(df_penguins[columns[1]], df_penguins[columns[3]])
                    elif inside == 2:
                        ax[inside,outside].scatter(df_penguins[columns[2]], df_penguins[columns[3]])
                    
    
    fig.suptitle('Palmer Penguins Attributes Scatterplot Matrix') 
    fig.tight_layout()#title for the entire figure
    fig.savefig('penguins_attributes_scplama.png')


    #2.Create a single scatterplot which visualizes bill length vs flipper length. 
    #  Each datapoint should be color-coded based on its species
    
  
    fig,ax=plt.subplots(1,1,figsize=(10,6))
    df_penguin_adelie_filtered= df_penguin.loc[df_penguin['species'] == "Adelie", ["bill_length_mm", "flipper_length_mm"]] #filteering the data by loc which is rows, then using columns to filter it and keep the columns we want
    df_penguin_chinstrap_filtered = df_penguin.loc[df_penguin['species'] == "Chinstrap", ["bill_length_mm", "flipper_length_mm"]]
    df_penguin_gentoo_filtered = df_penguin.loc[df_penguin['species'] == "Gentoo",["bill_length_mm", "flipper_length_mm"]]
    
    #converting to columns for easier access
    ad_columns = df_penguin_adelie_filtered.columns 
    chin_columns = df_penguin_chinstrap_filtered.columns
    gent_columns = df_penguin_gentoo_filtered.columns



    ax.scatter(df_penguin_adelie_filtered[ad_columns[0]], df_penguin_adelie_filtered[ad_columns[1]], label = 'Adelie')
    ax.scatter(df_penguin_chinstrap_filtered[chin_columns[0]], df_penguin_chinstrap_filtered[chin_columns[1]], label = "Chinstrap")
    ax.scatter(df_penguin_gentoo_filtered[gent_columns[0]], df_penguin_gentoo_filtered[gent_columns[1]], label = "Gentoo")

    fig.suptitle("bill_length_mm vs flipper_length_mm")
    ax.set(xlabel = "bill_length_mm", ylabel ="flipper_length_mm")
    ax.legend() #displays the legend
    fig.tight_layout()
    
    fig.savefig('penguins_bill_flipper_by_species_sc.png')

    # Print the image file path or base64 string to pass to JavaScript
    print('penguins_attributes_scplama.png\npenguins_bill_flipper_by_species_sc.png')  # This will be passed to JavaScript


if __name__ == '__main__':
    main()
