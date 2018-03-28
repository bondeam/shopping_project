import pandas as pd
import glob, os  


def get_pick_drop(directory):
    path = directory # './data'
    drop_Files = glob.glob(path + "/*_shelf_wood_drop*.csv")
    pick_Files = glob.glob(path + "/*_shelf_wood_pick*.csv")

    drop = []
    pick = []

    for file_ in drop_Files:
        # remove the first line, starting with "shopping"
        df = pd.read_csv(file_,index_col=None, header=None).iloc[1:]
        drop.append(df)
        
    for file_ in pick_Files:
        df = pd.read_csv(file_,index_col=None, header=None).iloc[1:] 
        pick.append(df)

return pick, drop