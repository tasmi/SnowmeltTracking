"""
Detect_Melt.py

Functions for determining the start and end of the snowmelt season from passive microwave data
Author: Taylor Smith (tasmith@uni-potsdam.de)
Date, v0: 20.1.2017

"""

import numpy as np
import pandas as pd
from detect_peaks import detect_peaks #This function can be found at https://github.com/demotu/BMC
import traceback, datetime

def daterangelist(start_date, end_date, skip):
    """ 
    Helper Function to return a list of dates
    given a start_date, end_date, skip value in days
    Returns: A Python List of datetime64 objects    
    """
    l = np.array([])
    for n in range(0, int((end_date - start_date).days), skip):
        l = np.append(l, (start_date + datetime.timedelta(n)))
    return l.astype('datetime64')
    
def Melt_Dates(XPGR, SWE, Tb37V, END):
    """Classify the start and end of melt based upon XPGR, SWE, 37V PM data, and the long-term average end date.
    For classifying the long-term average end date, use 'NULL' in place of END."""
    def start(XPGR):
        """Find the start date of melt from a given XPGR time series"""
        peaks = detect_peaks(XPGR.values) #Identify peaks in XPGR data
        aoi_sum = -100 #Set initial condition
        for p in peaks:
            peak = XPGR.index[p]
            aoi = XPGR[np.logical_and(XPGR.index < peak + pd.Timedelta('3 days'), XPGR.index > peak - pd.Timedelta('3 days'))] #Select values within 3 days
            if np.nanmean(aoi.values) > aoi_sum: 
                aoi_sum = np.nanmean(aoi.values) #Replace the initial condition if the peak is higher/thicker
                current_max = peak
        try:
            return current_max
        except:
            return XPGR.idxmax() #If no peaks are above inital condition (error), return the date of yearly max XPGR
    
    def end(Tb37V, SWE):
        """Find the end date of melt from given 37V and SWE time series"""
        peaks = detect_peaks(Tb37V.values, mph=0) #Find all positive peaks in 37V data
        aoi_sum = 0 #Set initial condition
        for p in peaks:
            peak = Tb37V.index[p]
            aoi = Tb37V[np.logical_and(Tb37V.index < peak + pd.Timedelta('3 days'), Tb37V.index > peak - pd.Timedelta('3 days'))] #Select values within 3 days
            if np.nanmean(aoi.values) > aoi_sum: 
                aoi_sum = np.nanmean(aoi.values) #Replace initial condition if the peak is higher/thicker
                current_max = peak
        RollSWE = SWE.copy() #Create a copy of the SWE data
        minswe = np.nanmin(SWE.values) #Find yearly minimum SWE
        RollSWE[SWE.values <= minswe + 2] = 1 #Classify the SWE series into 1s and 0s based on distance from min swe
        RollSWE[SWE.values > minswe + 2] = 0
        rm = pd.rolling_sum(RollSWE, 5) #Calculate a rolling sum on a 5 day window
        snow_clear = rm[rm.values >= 4].index.min() - pd.Timedelta('4 days') #Declare snow clearance if 4/5 days are 'snow clear'
        try:
            return pd.Series((snow_clear, current_max)).min() #Choose the minimum between the SWE clearance and the Tb37 max dates
        except:
            try:
                return pd.Series(snow_clear, Tb37V.idxmax()).min() #If no data was returned from the Tb37 peak find, use the maximum yearly value instead
            except:
                return snow_clear #If this also fails, use only the snow clearance date to determine date of snow melt off
                
    if END != 'NULL': #Main classifier
        endmelt = end(Tb37V, SWE)
        if np.abs((END - endmelt).days) > 60: #If there is high deviation from long-term average end dates, reclassify using a threshold
            m = start(XPGR)
            endmelt = end(Tb37V[Tb37V.index > m], SWE[SWE.index > m])
            m = start(XPGR[XPGR.index < endmelt])
        else:
            m = start(XPGR[XPGR.index < endmelt])
    elif END == 'NULL': #Initial use, when long-term average time series is used instead of yearly data
        m = start(XPGR) #First guess of melt onset
        endmelt = end(Tb37V, SWE) #First guess of melt end
        if m > endmelt: #If the date of melt onset is later than the date of melt end, reclassify the melt end date using only the data following the onset of melt
            endmelt = end(Tb37V[Tb37V.index > m], SWE[SWE.index > m])
        else:
            m = start(XPGR[XPGR.index < endmelt]) #If the date of melt onset is before the date of melt end, reclassify the start of the melt season only with those data before the end of melt
    
    return m, endmelt

def Average_Year(ser):
    """Generate an average, long-term year from a given pandas time series"""
    Average_Year = ser.groupby([ser.index.month, ser.index.day]).mean() #Get average value at each day of year over whole series length
    try:
        Average_Year.index = daterangelist(datetime.datetime(2000,1,1), datetime.datetime(2001,1,1),1) #Fit these days onto a 366 day series (to account for leap years)
    except:
        minus = 366 - Average_Year.index.shape[0]
        Average_Year.index = daterangelist(datetime.datetime(2000,1,1), datetime.datetime(2001,1,1) - pd.Timedelta(str(minus) + ' days'),1) #If your series doesn't contain each day of the year, shrink the series
    av = Average_Year.index.copy()
    idx = np.where(av > pd.Timestamp('2000-09-30'))
    idx2 = np.where(av <= pd.Timestamp('2000-09-30'))
    new = av[idx].map(lambda t: t.replace(year=1999, month=t.month, day=t.day))
    new2 = av[idx2].map(lambda t: t.replace(year=2000, month=t.month, day=t.day))
    Av = pd.Series(np.concatenate((Average_Year.values[idx2],Average_Year.values[idx])), index=np.concatenate((new2,new)))
    Av = Av.sort_index() #Reshape the matrix so that it is as a 'water year', running from Oct 1 to Sept 30
    
    return Av
    
def WaterYear(series):
    """ 
    Takes a pandas series and splits it into water-year segments.
    Returns a list of pandas series
    """
    chunked = []
    for yr in np.unique(series.index.year):
        if yr == np.min(np.unique(series.index.year)):
            S = series[np.logical_and(series.index.year == np.min(np.unique(series.index.year)), series.index.month < 10)]
        else:
            S = pd.concat([series[np.logical_and(series.index.year == (yr - 1), series.index.month > 9)], series[np.logical_and(series.index.year == yr, series.index.month < 10)]])
        chunked.append(S)
    return chunked
        
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Takes input of series, window_size, order, deriv, and rate
    DEFAULT: savitzky_golay(SWE.values, 21, 1, deriv=0, rate=1)
    window_size must be ODD
    """
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
    
#Example Usage:
def MeltDates(XPGR, SWE, Tb37):
    """ Example function to take XPGR, SWE, and Tb37V data to track the onset
    and end of melt on a year-by-year basis.
    Inputs are pandas Series obbjects.
    """
    Dict = {}
    #Compute Average Year
    Av = Average_Year(XPGR) 
    AveSWE = Average_Year(SWE)
    Ave37 = Average_Year(Tb37)
    avem, aveend = Melt_Dates(Av, AveSWE, Ave37, 'NULL')
    avem = pd.Timestamp(avem, tz='UTC') - pd.Timestamp('1999-10-01', tz='UTC') #Change the average date to a 'distance from Oct1 metric'
    aveend = pd.Timestamp(aveend, tz='UTC') - pd.Timestamp('1999-10-01', tz='UTC')
    
    longavstart = avem.days
    longavend = aveend.days

    chunk_XPGR = WaterYear(XPGR)
    chunk_37 = WaterYear(Tb37)
    chunk_swe = WaterYear(SWE)
    try:
        num = len(chunk_XPGR)
    except:
        num = 1
    for ii in range(0,num):
        #For each year of data... 
        XPGR = chunk_XPGR[ii]
        Tb37V = chunk_37[ii]
        SWE = chunk_swe[ii]
        if not XPGR.index.year.max() in baddict[inst]:
            year = XPGR.index.year.max()
            #Generate expected start dates from long-term averages...
            START = avem + XPGR.index.min()
            END = aveend + XPGR.index.min()
            
            m, endmelt = Melt_Dates(XPGR, SWE, Tb37V, END)
            try:
                if (endmelt - m).days > 0:
                    oct1 = pd.Timestamp('%s-10-01' % XPGR.index.year.min()) #Oct 1, start of water year
                    distfromstart = (m - oct1).days
                    distfromend = (endmelt - oct1).days
                    period = (endmelt - m).days
                    startdeviation = (m - START).days
                    enddeviation = (endmelt - END).days
                    #Clean unreasonable values...
                    if period > 366:
                        period = np.nan
                    if distfromstart > 366:
                        distfromstart = np.nan
                    if distfromend > 366:
                        distfromend = np.nan
                    if period < 0:
                        period = np.nan
                    if distfromstart < 0:
                        distfromstart = np.nan
                    if distfromend < 0:
                        distfromend = np.nan
                else:
                    distfromstart, distfromend, period = (np.nan,)*3
            except:
                distfromstart, distfromend, period = (np.nan,)*3
                traceback.print_exc()
            storedata = [period, distfromstart, distfromend]
            Dict[str(year)] = storedata #Save the data to a by-year dictionary
            del XPGR, Tb37V, SWE
    DictOut = {}
    for yr in range(1988,2017):
        l1, l2, l3 = [], [], []
        for i in insts:
            try:
                data = Dict[i + str(yr)]
                l1.append(data[0])
                l2.append(data[1])
                l3.append(data[2])
            except:
                pass
        #If there are multiple values for one year (for example, coming from multiple sensors), average them if they are close, or throw out bad ones if some are 
        #far from the long-term average
        l_list = [l1, l2, l3]
        indlist = [longavend - longavstart, longavstart, longavend]
        outlist = []
        for i in [0,1,2]:
            try:
                l = l_list[i]
                if len(l) > 1:
                    try:
                        if np.nanmax(l) - np.nanmin(l) > 14:
                            ind = indlist[i]
                            finder = np.abs(np.array(l) - ind)
                            l = l[np.where(finder == np.nanmin(finder))[0]]
                        else:
                            l = np.nanmean(l)
                    except:
                        l = np.nanmean(l)
                else:
                    l = float(l[0])
            except:
                l = np.nan
            outlist.append(l)
        DictOut[str(yr)] = outlist
        
    #return a dictionary with year: [start, end, period] pairs. 
    return DictOut 
    
    
    